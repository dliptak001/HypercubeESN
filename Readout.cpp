#include "Readout.h"
#include "HCNN.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>


Readout::Readout(const ReadoutConfig& cfg)
    : config_(cfg)
    , num_outputs_(static_cast<size_t>(cfg.num_outputs))
{}

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
    assert(config_.dim >= 5);
    const size_t n = 1ULL << config_.dim;
    num_features_ = n;
    num_outputs_ = static_cast<size_t>(config_.num_outputs);
    const bool is_classification = (config_.task == ReadoutTask::Classification);

    build_architecture();
    net_->SetOptimizer(hcnn::OptimizerType::ADAM);
    trained_ = true;

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

    std::vector<float> verbose_logits;
    std::vector<float> verbose_preds;
    if (config_.verbose && config_.verbose_train_acc) {
        if (is_classification)
            verbose_logits.resize(num_samples * num_outputs_);
        else
            verbose_preds.resize(num_samples * num_outputs_);
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

        if (config_.verbose) {
            if (config_.verbose_train_acc) {
                if (is_classification) {
                    net_->ForwardBatch(states, static_cast<int>(n),
                                       static_cast<int>(num_samples),
                                       verbose_logits.data());
                    size_t correct = 0;
                    for (size_t s = 0; s < num_samples; ++s) {
                        const float* row = verbose_logits.data() + s * num_outputs_;
                        size_t pred = 0;
                        float best = row[0];
                        for (size_t k = 1; k < num_outputs_; ++k)
                            if (row[k] > best) { best = row[k]; pred = k; }
                        if (static_cast<int>(pred) == int_targets[s]) ++correct;
                    }
                    double acc = 100.0 * correct / num_samples;
                    std::printf("  epoch %3d/%d  lr=%.5f  train_acc=%.2f%%\n",
                                e + 1, config_.epochs, lr, acc);
                } else {
                    net_->ForwardBatch(states, static_cast<int>(n),
                                       static_cast<int>(num_samples),
                                       verbose_preds.data());
                    double mse = 0.0;
                    for (size_t i = 0; i < num_samples * num_outputs_; ++i) {
                        double d = verbose_preds[i] - targets[i];
                        mse += d * d;
                    }
                    mse /= static_cast<double>(num_samples * num_outputs_);
                    std::printf("  epoch %3d/%d  lr=%.5f  train_mse=%.6f\n",
                                e + 1, config_.epochs, lr, mse);
                }
            } else {
                std::printf("  epoch %3d/%d  lr=%.5f\n",
                            e + 1, config_.epochs, lr);
            }
            std::fflush(stdout);
        }
    }

    flatten_weights();
}

// ---------------------------------------------------------------------------
//  Online (streaming) training
// ---------------------------------------------------------------------------

void Readout::InitOnline()
{
    assert(config_.dim >= 5);
    const size_t n = 1ULL << config_.dim;
    num_features_ = n;
    num_outputs_ = static_cast<size_t>(config_.num_outputs);

    build_architecture();
    net_->SetOptimizer(hcnn::OptimizerType::ADAM);
    net_->PrepareBuffers();
    trained_ = true;
}

void Readout::TrainOnlineStep(const float* state, int target_class,
                                 float lr, float weight_decay)
{
    assert(trained_ && net_);
    const size_t n = num_features_;

    net_->TrainStep(state, static_cast<int>(n), target_class,
                    lr, config_.momentum, weight_decay);
}

void Readout::TrainOnlineBatch(const float* states, const int* targets,
                                  size_t count, float lr, float weight_decay)
{
    assert(trained_ && net_);
    const size_t n = num_features_;

    net_->TrainBatch(states, static_cast<int>(n),
                     targets, static_cast<int>(count),
                     lr, config_.momentum, weight_decay);
}

void Readout::TrainOnlineStepRegression(const float* state, const float* target,
                                           float lr, float weight_decay)
{
    assert(trained_ && net_);
    const size_t n = num_features_;

    net_->TrainStepRegression(state, static_cast<int>(n), target,
                              lr, config_.momentum, weight_decay);
}

void Readout::TrainOnlineBatchRegression(const float* states, const float* targets,
                                            size_t count, float lr, float weight_decay)
{
    assert(trained_ && net_);
    const size_t n = num_features_;

    net_->TrainBatchRegression(states, static_cast<int>(n),
                               targets, static_cast<int>(count),
                               lr, config_.momentum, weight_decay);
}

// ---------------------------------------------------------------------------
//  Prediction
// ---------------------------------------------------------------------------

void Readout::PredictRaw(const float* state, float* output) const
{
    assert(trained_ && net_);
    const size_t n = num_features_;

    net_->Embed(state, static_cast<int>(n), scratch_embedded_.data());
    net_->Forward(scratch_embedded_.data(), scratch_pred_.data());

    for (size_t k = 0; k < num_outputs_; ++k)
        output[k] = scratch_pred_[k];
}

float Readout::PredictRaw(const float* state) const
{
    assert(num_outputs_ == 1);
    // Write into the correctly-sized scratch buffer, not a single stack float:
    // the float* overload writes num_outputs_ values, so a &float target would
    // overflow the stack for a multi-output readout in release builds (where the
    // assert above is compiled out). Callers needing every channel must use the
    // float* overload. scratch_pred_ is sized to num_outputs_ in build_architecture().
    PredictRaw(state, scratch_pred_.data());
    return scratch_pred_[0];
}

int Readout::PredictClass(const float* state) const
{
    assert(trained_ && net_);
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

const std::vector<double>& Readout::Weights() const
{
    if (weights_blob_.empty() && net_) {
        auto fw = net_->GetWeights();
        weights_blob_.assign(fw.begin(), fw.end());
    }
    return weights_blob_;
}

void Readout::flatten_weights()
{
    if (!net_) { weights_blob_.clear(); return; }
    auto fw = net_->GetWeights();
    weights_blob_.assign(fw.begin(), fw.end());
}

void Readout::rebuild_from_blob()
{
    if (weights_blob_.empty() || config_.dim == 0) return;

    // Reconstruct the network from stored config if needed.
    if (!net_) {
        build_architecture();
    }

    std::vector<float> fw(weights_blob_.begin(), weights_blob_.end());
    net_->SetWeights(fw);
}

void Readout::SetState(std::vector<double> weights)
{
    weights_blob_ = std::move(weights);
    num_features_ = (config_.dim >= 5) ? (1ULL << config_.dim) : 0;

    if (!weights_blob_.empty()) {
        rebuild_from_blob();
        trained_ = true;
    }
}
