// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#include "HCNN.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace hcnn {

HCNN::HCNN(int start_dim, int num_outputs, int input_channels,
           TaskType task_type, LossType loss_type,
           size_t num_threads)
    : net_(std::make_unique<HCNNNetwork>(start_dim, num_outputs, input_channels,
                                         task_type, loss_type,
                                         num_threads)) {}

HCNN::~HCNN() = default;

// ---------------------------------------------------------------------------
//  Architecture
// ---------------------------------------------------------------------------
void HCNN::AddConv(int c_out, Activation activation,
                   bool use_bias, bool use_batchnorm) {
    net_->add_conv(c_out, activation, use_bias, use_batchnorm);
}

void HCNN::AddPool(PoolType type) {
    net_->add_pool(type);
}

void HCNN::RandomizeWeights(float scale, unsigned seed) {
    net_->randomize_all_weights(scale, seed);
}

// ---------------------------------------------------------------------------
//  Mode / optimizer
// ---------------------------------------------------------------------------
void HCNN::SetTraining(bool training) {
    net_->set_training(training);
}

void HCNN::SetOptimizer(OptimizerType type, float beta1, float beta2, float eps) {
    net_->set_optimizer(type, beta1, beta2, eps);
}

void HCNN::PrepareBuffers() {
    net_->prepare_all_buffers();
}

// ---------------------------------------------------------------------------
//  Inference
// ---------------------------------------------------------------------------
void HCNN::Embed(const float* raw_input, int input_length,
                 float* embedded_out) const {
    net_->embed_input(raw_input, input_length, embedded_out);
}

void HCNN::Forward(const float* embedded, float* logits) const {
    net_->forward(embedded, logits);
}

void HCNN::ForwardBatch(const float* flat_inputs, int input_length,
                        int batch_size, float* logits_out) {
    net_->forward_batch(flat_inputs, input_length, batch_size, logits_out);
}

// ---------------------------------------------------------------------------
//  Training — classification
// ---------------------------------------------------------------------------
void HCNN::TrainStep(const float* raw_input, int input_length, int target_class,
                     float learning_rate, float momentum, float weight_decay,
                     const float* class_weights) {
    net_->train_step(raw_input, input_length, target_class, learning_rate,
                     momentum, weight_decay, class_weights);
}

void HCNN::TrainBatch(const float* flat_inputs, int input_length,
                      const int* targets, int batch_size,
                      float learning_rate, float momentum, float weight_decay,
                      const float* class_weights) {
    net_->train_batch(flat_inputs, input_length, targets, batch_size,
                      learning_rate, momentum, weight_decay, class_weights);
}

void HCNN::TrainEpoch(const float* flat_inputs, int input_length,
                      const int* targets, int sample_count, int batch_size,
                      float learning_rate, float momentum, float weight_decay,
                      const float* class_weights, unsigned shuffle_seed) {
    if (batch_size <= 0) {
        throw std::invalid_argument("HCNN::TrainEpoch: batch_size must be > 0");
    }
    if (sample_count < 0) {
        throw std::invalid_argument("HCNN::TrainEpoch: sample_count must be >= 0");
    }
    if (sample_count == 0) return;

    const auto n  = static_cast<size_t>(sample_count);
    const auto il = static_cast<size_t>(input_length);
    const auto bs = static_cast<size_t>(batch_size);

    // Build shuffle permutation if requested.
    if (shuffle_seed != 0) {
        if (shuffle_idx_.size() < n) shuffle_idx_.resize(n);
        std::iota(shuffle_idx_.begin(), shuffle_idx_.begin() + n, 0);
        std::mt19937 rng(shuffle_seed);
        std::shuffle(shuffle_idx_.begin(), shuffle_idx_.begin() + n, rng);
    }

    for (int start = 0; start < sample_count; start += batch_size) {
        int chunk = std::min(batch_size, sample_count - start);

        if (shuffle_seed != 0) {
            // Gather this chunk into contiguous scratch buffers.
            if (shuffle_inputs_.size() < bs * il)
                shuffle_inputs_.resize(bs * il);
            if (shuffle_targets_.size() < bs)
                shuffle_targets_.resize(bs);

            for (int i = 0; i < chunk; ++i) {
                int j = shuffle_idx_[start + i];
                std::memcpy(shuffle_inputs_.data() + i * il,
                            flat_inputs + j * il, il * sizeof(float));
                shuffle_targets_[i] = targets[j];
            }
            net_->train_batch(shuffle_inputs_.data(), input_length,
                              shuffle_targets_.data(), chunk,
                              learning_rate, momentum, weight_decay,
                              class_weights);
        } else {
            net_->train_batch(flat_inputs + start * il, input_length,
                              targets + start, chunk,
                              learning_rate, momentum, weight_decay,
                              class_weights);
        }
    }
}

// ---------------------------------------------------------------------------
//  Training — regression
// ---------------------------------------------------------------------------
void HCNN::TrainStepRegression(const float* raw_input, int input_length,
                               const float* target, float learning_rate,
                               float momentum, float weight_decay) {
    net_->train_step_regression(raw_input, input_length, target, learning_rate,
                                momentum, weight_decay);
}

void HCNN::TrainBatchRegression(const float* flat_inputs, int input_length,
                                const float* flat_targets, int batch_size,
                                float learning_rate, float momentum,
                                float weight_decay) {
    net_->train_batch_regression(flat_inputs, input_length, flat_targets,
                                 batch_size, learning_rate, momentum,
                                 weight_decay);
}

void HCNN::TrainEpochRegression(const float* flat_inputs, int input_length,
                                const float* flat_targets,
                                int sample_count, int batch_size,
                                float learning_rate, float momentum,
                                float weight_decay, unsigned shuffle_seed) {
    if (batch_size <= 0) {
        throw std::invalid_argument("HCNN::TrainEpochRegression: batch_size must be > 0");
    }
    if (sample_count < 0) {
        throw std::invalid_argument("HCNN::TrainEpochRegression: sample_count must be >= 0");
    }
    if (sample_count == 0) return;

    const auto n   = static_cast<size_t>(sample_count);
    const auto il  = static_cast<size_t>(input_length);
    const auto bs  = static_cast<size_t>(batch_size);
    const auto K   = static_cast<size_t>(net_->get_num_outputs());

    // Build shuffle permutation if requested.
    if (shuffle_seed != 0) {
        if (shuffle_idx_.size() < n) shuffle_idx_.resize(n);
        std::iota(shuffle_idx_.begin(), shuffle_idx_.begin() + n, 0);
        std::mt19937 rng(shuffle_seed);
        std::shuffle(shuffle_idx_.begin(), shuffle_idx_.begin() + n, rng);
    }

    for (int start = 0; start < sample_count; start += batch_size) {
        int chunk = std::min(batch_size, sample_count - start);

        if (shuffle_seed != 0) {
            // Gather this chunk into contiguous scratch buffers.
            if (shuffle_inputs_.size() < bs * il)
                shuffle_inputs_.resize(bs * il);
            if (shuffle_targets_f_.size() < bs * K)
                shuffle_targets_f_.resize(bs * K);

            for (int i = 0; i < chunk; ++i) {
                int j = shuffle_idx_[start + i];
                std::memcpy(shuffle_inputs_.data() + i * il,
                            flat_inputs + j * il, il * sizeof(float));
                std::memcpy(shuffle_targets_f_.data() + i * K,
                            flat_targets + j * K, K * sizeof(float));
            }
            net_->train_batch_regression(shuffle_inputs_.data(), input_length,
                                         shuffle_targets_f_.data(), chunk,
                                         learning_rate, momentum, weight_decay);
        } else {
            net_->train_batch_regression(flat_inputs + start * il, input_length,
                                         flat_targets + start * K, chunk,
                                         learning_rate, momentum, weight_decay);
        }
    }
}

// ---------------------------------------------------------------------------
//  Sizing accessors
// ---------------------------------------------------------------------------
int HCNN::GetStartDim() const       { return net_->get_start_dim(); }
int HCNN::GetStartN() const         { return net_->get_start_N(); }
int HCNN::GetInputChannels() const  { return net_->get_input_channels(); }
int HCNN::GetNumOutputs() const     { return net_->get_num_outputs(); }
TaskType HCNN::GetTaskType() const  { return net_->get_task_type(); }
LossType HCNN::GetLossType() const  { return net_->get_loss_type(); }

// ---------------------------------------------------------------------------
//  Weight serialization
// ---------------------------------------------------------------------------

size_t HCNN::GetWeightCount() const {
    size_t total = 0;
    for (size_t i = 0; i < net_->get_num_conv(); ++i) {
        const auto& conv = net_->get_conv(i);
        total += static_cast<size_t>(conv.get_kernel_size());
        total += static_cast<size_t>(conv.get_bias_size());
    }
    const auto& ro = net_->get_readout();
    total += static_cast<size_t>(ro.get_weight_size());
    total += static_cast<size_t>(ro.get_bias_size());
    return total;
}

std::vector<float> HCNN::GetWeights() const {
    std::vector<float> blob;
    blob.reserve(GetWeightCount());

    for (size_t i = 0; i < net_->get_num_conv(); ++i) {
        const auto& conv = net_->get_conv(i);
        const float* k = conv.get_kernel_data();
        blob.insert(blob.end(), k, k + conv.get_kernel_size());
        const float* b = conv.get_bias_data();
        blob.insert(blob.end(), b, b + conv.get_bias_size());
    }

    const auto& ro = net_->get_readout();
    const float* w = ro.get_weight_data();
    blob.insert(blob.end(), w, w + ro.get_weight_size());
    const float* b = ro.get_bias_data();
    blob.insert(blob.end(), b, b + ro.get_bias_size());

    return blob;
}

void HCNN::SetWeights(const std::vector<float>& blob) {
    if (blob.size() != GetWeightCount()) {
        throw std::invalid_argument(
            "HCNN::SetWeights: blob size " + std::to_string(blob.size()) +
            " != weight count " + std::to_string(GetWeightCount()));
    }

    size_t offset = 0;

    for (size_t i = 0; i < net_->get_num_conv(); ++i) {
        auto& conv = net_->get_conv(i);
        int ks = conv.get_kernel_size();
        std::memcpy(conv.get_kernel_data(), blob.data() + offset, ks * sizeof(float));
        offset += static_cast<size_t>(ks);
        int bs = conv.get_bias_size();
        std::memcpy(conv.get_bias_data(), blob.data() + offset, bs * sizeof(float));
        offset += static_cast<size_t>(bs);
    }

    auto& ro = net_->get_readout();
    int ws = ro.get_weight_size();
    std::memcpy(ro.get_weight_data(), blob.data() + offset, ws * sizeof(float));
    offset += static_cast<size_t>(ws);
    int bs = ro.get_bias_size();
    std::memcpy(ro.get_bias_data(), blob.data() + offset, bs * sizeof(float));
    offset += static_cast<size_t>(bs);

    assert(offset == blob.size());
}

} // namespace hcnn
