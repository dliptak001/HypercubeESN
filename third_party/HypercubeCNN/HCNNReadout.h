// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

#include <vector>
#include <cstdint>
#include <random>

// OptimizerType enum — defined in HCNNConv.h (no circular dependency)
#include "HCNNConv.h"

namespace hcnn {

/**
 * @class HCNNReadout
 * @brief Final pipeline stage: collapses the hypercube activations into
 *        `num_outputs` real-valued scalars.
 *
 * The readout is loss-agnostic and task-agnostic.  It outputs
 * `num_outputs` raw scalars which downstream code interprets either as
 * classification logits (fed through softmax + cross-entropy) or as
 * regression predictions (fed through MSE or other continuous losses).
 * No activation, no softmax — the linear layer output is final.
 *
 * The constructor receives `input_channels = c_final * N_final` and
 * the network passes `N = 1` to forward(), so the internal channel-wise
 * average is a no-op and the linear layer sees every (channel, vertex)
 * activation as an independent input (FLATTEN readout).
 *
 * Owns: weight matrix + bias + matching first / second moment buffers
 * (Adam allocates the second moments on demand via set_optimizer).
 *
 * Two backward paths mirror HCNNConv:
 *   - backward(): apply gradients in-place via the configured optimizer.
 *   - compute_gradients() + apply_gradients(): write raw gradients into
 *     caller buffers, then apply once after batch reduction.  The caller
 *     is responsible for computing the upstream loss gradient with
 *     respect to the outputs (`grad_logits`) — the readout itself has no
 *     notion of loss.
 *
 * Power-user class: ordinary SDK consumers should use HCNN.
 */
class HCNNReadout {
public:
    HCNNReadout(int num_outputs, int input_channels);

    void randomize_weights(float scale, std::mt19937& rng);

    // work_buf: optional pre-allocated buffer of at least input_channels floats.
    void forward(const float* in, float* out, int N,
                 float* work_buf = nullptr) const;

    // Backward: computes grad_in (if non-null) and updates weights via SGD with optional momentum.
    // work_buf: optional pre-allocated buffer of at least input_channels floats.
    void backward(const float* grad_logits, const float* in, int N,
                  float* grad_in, float learning_rate, float momentum = 0.0f,
                  float weight_decay = 0.0f, int timestep = 0,
                  float* work_buf = nullptr);

    // Compute gradients without applying SGD update.
    // work_buf: optional pre-allocated buffer of at least input_channels floats.
    void compute_gradients(const float* grad_logits, const float* in, int N,
                           float* grad_in, float* weight_grad, float* bias_grad,
                           float* work_buf = nullptr) const;

    // Apply externally computed (averaged) gradients via momentum SGD.
    void apply_gradients(const float* weight_grad, const float* bias_grad,
                         float learning_rate, float momentum, float weight_decay = 0.0f,
                         int timestep = 0);

    /// Configure the optimizer. Allocates second-moment buffers for Adam.
    void set_optimizer(OptimizerType type, float beta1 = 0.9f,
                       float beta2 = 0.999f, float eps = 1e-8f);

    int get_num_outputs() const { return num_outputs; }
    int get_input_channels() const { return input_channels; }

    float* get_weight_data() { return weights.data(); }
    const float* get_weight_data() const { return weights.data(); }
    int get_weight_size() const { return static_cast<int>(weights.size()); }
    float* get_bias_data() { return bias.data(); }
    const float* get_bias_data() const { return bias.data(); }
    int get_bias_size() const { return static_cast<int>(bias.size()); }

private:
    int num_outputs;
    int input_channels;
    std::vector<float> weights;
    std::vector<float> bias;
    std::vector<float> weight_m;    // first moment (SGD velocity / Adam m)
    std::vector<float> bias_m;      // first moment for bias
    std::vector<float> weight_m2;   // second moment (Adam only)
    std::vector<float> bias_m2;     // second moment (Adam only)
    OptimizerType optimizer_type_ = OptimizerType::SGD;
    float adam_beta1_ = 0.9f, adam_beta2_ = 0.999f, adam_eps_ = 1e-8f;
};

} // namespace hcnn
