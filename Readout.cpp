#include "Readout.h"
#include "HCNN.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>


Readout::Readout(const ReadoutConfig& cfg)
    : config_(cfg)
    , num_outputs_(static_cast<size_t>(cfg.num_outputs))
{
    // Build the network eagerly. build_architecture() needs only the config
    // (no data, no warm-up), so there is nothing to defer: net_ is a non-null
    // invariant from construction on. This removes the lazy-init dance (the
    // old InitOnline ceremony, the trained_-as-"is-built" flag, the triplicated
    // build+optimizer+buffers sequence) that existed only because warm-up was
    // historically treated as a readout-disengaged phase.
    num_features_ = 1ULL << config_.dim;
    build_architecture();
    net_->SetOptimizer(hcnn::OptimizerType::ADAM);
    net_->PrepareBuffers();
}

Readout::~Readout() = default;
Readout::Readout(Readout&&) noexcept = default;
Readout& Readout::operator=(Readout&&) noexcept = default;

// ---------------------------------------------------------------------------
//  Architecture
// ---------------------------------------------------------------------------

static hcnn::Activation map_activation(ReadoutActivation a)
{
    switch (a) {
        case ReadoutActivation::TANH:       return hcnn::Activation::TANH;
        case ReadoutActivation::RELU:       return hcnn::Activation::RELU;
        case ReadoutActivation::LEAKY_RELU: return hcnn::Activation::LEAKY_RELU;
        case ReadoutActivation::NONE:       return hcnn::Activation::NONE;
    }
    return hcnn::Activation::TANH;
}

void Readout::build_architecture()
{
    assert(config_.dim >= 5);
    const size_t n = 1ULL << config_.dim;
    const int d = static_cast<int>(config_.dim);

    // Auto-size layers: min(DIM - 2, 2), at least 1.
    int layers = (config_.num_layers > 0)
                     ? config_.num_layers
                     : std::min(d - 2, 2);
    layers = std::max(layers, 1);
    assert(layers <= d - 2);

    auto task_type = (config_.task == ReadoutTask::Classification)
                         ? hcnn::TaskType::Classification
                         : hcnn::TaskType::Regression;
    net_ = std::make_unique<hcnn::HCNN>(
        d, config_.num_outputs, /*input_channels=*/1,
        task_type);

    const hcnn::Activation act = map_activation(config_.activation);
    int ch = config_.conv_channels;
    for (int i = 0; i < layers; ++i) {
        net_->AddConv(ch, act, /*use_bias=*/true);
        net_->AddPool(hcnn::PoolType::MAX);
        ch *= 2;
    }

    net_->RandomizeWeights(0.0f, config_.seed);

    scratch_embedded_.resize(n);
    scratch_pred_.resize(num_outputs_);
}

// ---------------------------------------------------------------------------
//  Training
// ---------------------------------------------------------------------------

void Readout::Train(const float* states, const float* targets,
                       size_t num_samples)
{
    // net_ is already built (ctor). Train fits the existing network in place;
    // a second Train() continues from the current weights rather than
    // re-randomizing — reconstruct the Readout for a fresh fit.
    const size_t n = num_features_;
    const bool is_classification = (config_.task == ReadoutTask::Classification);

    const float lr_min = config_.lr_max * config_.lr_min_frac;
    const int horizon = (config_.lr_decay_epochs > 0)
                            ? config_.lr_decay_epochs
                            : config_.epochs;

    std::vector<int> int_targets;
    if (is_classification) {
        int_targets.resize(num_samples);
        for (size_t s = 0; s < num_samples; ++s)
            int_targets[s] = static_cast<int>(targets[s]);
    }

    for (int e = 0; e < config_.epochs; ++e) {
        float lr = CosineLR(static_cast<float>(e) / static_cast<float>(horizon),
                            config_.lr_max, lr_min);

        if (is_classification) {
            net_->TrainEpoch(
                states, static_cast<int>(n),
                int_targets.data(),
                static_cast<int>(num_samples), config_.batch_size,
                lr, config_.momentum, config_.weight_decay,
                /*class_weights=*/nullptr,
                /*shuffle_seed=*/static_cast<unsigned>(e + 1));
        } else {
            net_->TrainEpochRegression(
                states, static_cast<int>(n),
                targets,
                static_cast<int>(num_samples), config_.batch_size,
                lr, config_.momentum, config_.weight_decay,
                /*shuffle_seed=*/static_cast<unsigned>(e + 1));
        }
    }
}

// ---------------------------------------------------------------------------
//  Streaming training (dispatches on config_.task)
// ---------------------------------------------------------------------------

void Readout::TrainStep(const float* state, const float* target,
                        float lr, float weight_decay)
{
    assert(net_);
    const int n = static_cast<int>(num_features_);

    if (config_.task == ReadoutTask::Classification)
        net_->TrainStep(state, n, static_cast<int>(target[0]),
                        lr, config_.momentum, weight_decay);
    else
        net_->TrainStepRegression(state, n, target,
                                  lr, config_.momentum, weight_decay);
}

void Readout::TrainStepBatch(const float* states, const float* targets,
                             size_t count, float lr, float weight_decay)
{
    assert(net_);
    const int n = static_cast<int>(num_features_);

    if (config_.task == ReadoutTask::Classification) {
        // HCNN's classification path takes integer class labels; the unified
        // float* target carries each class index as a float, so narrow here.
        std::vector<int> labels(count);
        for (size_t i = 0; i < count; ++i)
            labels[i] = static_cast<int>(targets[i]);
        net_->TrainBatch(states, n, labels.data(), static_cast<int>(count),
                         lr, config_.momentum, weight_decay);
    } else {
        net_->TrainBatchRegression(states, n, targets, static_cast<int>(count),
                                   lr, config_.momentum, weight_decay);
    }
}

// ---------------------------------------------------------------------------
//  Prediction
// ---------------------------------------------------------------------------

void Readout::PredictRaw(const float* state, float* output) const
{
    assert(net_);
    const size_t n = num_features_;

    net_->Embed(state, static_cast<int>(n), scratch_embedded_.data());
    net_->Forward(scratch_embedded_.data(), scratch_pred_.data());

    for (size_t k = 0; k < num_outputs_; ++k)
        output[k] = scratch_pred_[k];
}

int Readout::PredictClass(const float* state) const
{
    assert(net_);
    const size_t n = num_features_;

    net_->Embed(state, static_cast<int>(n), scratch_embedded_.data());
    net_->Forward(scratch_embedded_.data(), scratch_pred_.data());

    return static_cast<int>(
        std::max_element(scratch_pred_.begin(),
                         scratch_pred_.begin() + num_outputs_) -
        scratch_pred_.begin());
}

// ---------------------------------------------------------------------------
//  Evaluation
// ---------------------------------------------------------------------------

double Readout::R2(const float* states, const float* targets,
                      const size_t num_samples) const
{
    if (num_samples == 0) return 0.0;
    const size_t n = num_features_;
    const size_t K = num_outputs_;

    // Predict all samples once, cache results.
    std::vector<float> preds(num_samples * K);
    for (size_t s = 0; s < num_samples; ++s)
        PredictRaw(states + s * n, preds.data() + s * K);

    // Average R2 across outputs.
    double r2_sum = 0.0;
    for (size_t k = 0; k < K; ++k) {
        double tgt_mean = 0.0;
        for (size_t s = 0; s < num_samples; ++s)
            tgt_mean += targets[s * K + k];
        tgt_mean /= static_cast<double>(num_samples);

        double ss_res = 0.0, ss_tot = 0.0;
        for (size_t s = 0; s < num_samples; ++s) {
            double y  = targets[s * K + k];
            double yh = preds[s * K + k];
            ss_res += (y - yh) * (y - yh);
            ss_tot += (y - tgt_mean) * (y - tgt_mean);
        }
        r2_sum += (ss_tot < 1e-12) ? 0.0 : (1.0 - ss_res / ss_tot);
    }
    return r2_sum / static_cast<double>(K);
}

double Readout::Accuracy(const float* states, const float* labels,
                            const size_t num_samples) const
{
    if (num_samples == 0) return 0.0;
    const size_t n = num_features_;
    size_t correct = 0;

    if (num_outputs_ > 1) {
        // Multi-class: argmax vs label.
        for (size_t s = 0; s < num_samples; ++s) {
            int pred = PredictClass(states + s * n);
            if (pred == static_cast<int>(labels[s])) ++correct;
        }
    } else {
        // Binary: threshold at 0.
        for (size_t s = 0; s < num_samples; ++s) {
            float pred_val;
            PredictRaw(states + s * n, &pred_val);
            if ((pred_val > 0.0f) == (labels[s] > 0.0f)) ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(num_samples);
}

// ---------------------------------------------------------------------------
//  Serialization
// ---------------------------------------------------------------------------

std::vector<double> Readout::Weights() const
{
    // Snapshot the live network's weights on demand, by value — a returned copy
    // can't go stale behind a later TrainStep* call (streaming training mutates
    // net_ in place).
    const std::vector<float> fw = net_->GetWeights();
    return std::vector<double>(fw.begin(), fw.end());
}

void Readout::SetState(std::vector<double> weights)
{
    // net_ is built (optimizer + buffers prepared) in the ctor — load the saved
    // weights straight into the existing, ready-to-train network.
    if (weights.empty()) return;
    const std::vector<float> fw(weights.begin(), weights.end());
    net_->SetWeights(fw);
}
