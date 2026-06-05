// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

#include "HCNNNetwork.h"   // re-exports HCNNConv, HCNNPool, HCNNReadout, all enums
#include <memory>
#include <vector>

namespace hcnn {

/// @class HCNN
/// @brief Top-level HypercubeCNN SDK front door.  One class wraps the entire
///        pipeline: input embedding -> conv/pool stack -> readout, plus
///        single-sample inference, batch inference, single-sample training,
///        and mini-batch / full-epoch training (classification and regression).
///
/// HCNN owns a `HCNNNetwork` (the internal orchestrator) and forwards every
/// public call to it through a thin PIMPL-style wrapper.  Use this class for
/// virtually all SDK consumption -- the underlying layer classes
/// (`HCNNNetwork`, `HCNNConv`, `HCNNPool`, `HCNNReadout`, `ThreadPool`) are
/// re-exported transitively via this header for power users who need direct
/// weight access (serialization, gradient checking, custom training loops),
/// but ordinary code should never need to reach for them.
///
/// Build the architecture incrementally with AddConv()/AddPool(), then call
/// RandomizeWeights() before training.  Classification (the default):
///
///     hcnn::HCNN net(10, /*num_outputs=*/10);   // DIM=10, N=1024, 10 classes
///     net.AddConv(32);
///     net.AddPool(hcnn::PoolType::MAX);
///     net.AddConv(64);
///     net.AddPool(hcnn::PoolType::MAX);
///     net.RandomizeWeights();
///
///     // Single-sample inference: caller owns and reuses both buffers.
///     std::vector<float> embedded(net.GetStartN());
///     std::vector<float> logits(net.GetNumOutputs());
///     net.Embed(raw, raw_len, embedded.data());
///     net.Forward(embedded.data(), logits.data());
///
/// Regression: pass `TaskType::Regression` at construction and use the
/// `*Regression` training API with `const float*` targets of length
/// `num_outputs`.  The readout layer and forward path are unchanged — the
/// raw output vector is simply interpreted as predictions instead of
/// logits, and the loss becomes MSE instead of softmax + cross-entropy:
///
///     hcnn::HCNN net(6, /*num_outputs=*/1, /*input_channels=*/1,
///                    hcnn::TaskType::Regression);
///     net.AddConv(16);
///     net.AddPool();
///     net.RandomizeWeights();
///
///     float target = 0.5f;
///     net.TrainStepRegression(raw, raw_len, &target, /*lr=*/0.01f);
///
/// **Contiguous data model.**  All batch and epoch methods accept data as
/// contiguous row-major `const float*` buffers — one base pointer plus a
/// uniform `int input_length`.  This eliminates a class of bugs inherent
/// in pointer-per-sample interfaces (off-by-one stride, mismatched lengths,
/// dangling pointers after reallocation) that produce silent data
/// corruption rather than compiler errors.
///
/// All methods that take raw inputs avoid hidden per-call allocations:
/// single-sample inference reuses persistent ping-pong scratch on the
/// network; batch inference and batch training reuse lazily-allocated
/// per-thread buffers.
///
/// **Enums** consumed by this API are defined alongside their owning
/// internal headers (all re-exported transitively via HCNN.h):
///   - `hcnn::PoolType`      (HCNNPool.h)     — MAX, AVG
///   - `hcnn::TaskType`      (HCNNNetwork.h)  — Classification, Regression
///   - `hcnn::LossType`      (HCNNNetwork.h)  — Default, CrossEntropy, MSE
///   - `hcnn::Activation`    (HCNNConv.h)     — NONE, RELU, LEAKY_RELU, TANH
///   - `hcnn::OptimizerType` (HCNNConv.h)     — SGD, ADAM
///
/// **Non-copyable, non-movable.**  HCNN owns a HCNNNetwork (which in turn
/// owns a ThreadPool with live worker threads) and persistent scratch
/// vectors used by inference and training.  Move semantics would require
/// either teaching the worker threads to follow the moved-from object or
/// rebuilding the pool on the destination — both add complexity for no
/// real-world win, so move is deleted entirely.  Wrap in
/// `std::unique_ptr<HCNN>` if you need transfer-of-ownership semantics.
class HCNN {
public:
    explicit HCNN(int start_dim, int num_outputs = 10,
                  int input_channels = 1,
                  TaskType task_type = TaskType::Classification,
                  LossType loss_type = LossType::Default,
                  size_t num_threads = 0);
    ~HCNN();

    HCNN(const HCNN&) = delete;
    HCNN& operator=(const HCNN&) = delete;
    HCNN(HCNN&&) = delete;
    HCNN& operator=(HCNN&&) = delete;

    // -----------------------------------------------------------------
    //  Architecture (incremental builder)
    // -----------------------------------------------------------------

    /// Append a convolutional layer with `c_out` output channels.
    void AddConv(int c_out, Activation activation = Activation::RELU,
                 bool use_bias = true, bool use_batchnorm = false);

    /// Append an antipodal pooling layer.  Reduces DIM by 1.
    void AddPool(PoolType type = PoolType::MAX);

    /// Initialize all weights.  scale > 0: uniform [-scale, +scale].
    /// scale <= 0 (default): per-layer Xavier/He init based on activation.
    void RandomizeWeights(float scale = 0.0f, unsigned seed = 42);

    // -----------------------------------------------------------------
    //  Mode / optimizer
    // -----------------------------------------------------------------

    /// Switch all batch-norm layers between training and eval mode.
    void SetTraining(bool training);

    /// Configure the optimizer for all layers.  Resets the timestep.
    void SetOptimizer(OptimizerType type, float beta1 = 0.9f,
                      float beta2 = 0.999f, float eps = 1e-8f);

    /// Eagerly allocate all internal work buffers (single-step, batch,
    /// and inference).  Normally these are allocated lazily on first use;
    /// call this after architecture setup to move the cost to startup.
    void PrepareBuffers();

    // -----------------------------------------------------------------
    //  Inference
    // -----------------------------------------------------------------

    /// Map a raw scalar array onto N = 2^start_dim hypercube vertices via
    /// Direct Linear Assignment.  Values must be in [-1.0, 1.0].
    /// `embedded_out` must hold GetStartN() floats.  Caller-owned buffer
    /// (designed for reuse across calls -- no hidden allocation).
    void Embed(const float* raw_input, int input_length,
               float* embedded_out) const;

    /// Run conv/pool/readout from already-embedded activations.
    /// `embedded` is GetStartN() floats; `logits` is GetNumOutputs() floats.
    /// For Classification nets these are raw pre-softmax logits; for
    /// Regression nets they are raw real-valued predictions.  No
    /// allocation.
    void Forward(const float* embedded, float* logits) const;

    /// Batch inference (parallel via internal thread pool).
    /// `flat_inputs` is `batch_size * input_length` contiguous floats
    /// (row-major).  `logits_out` must hold `batch_size * GetNumOutputs()`
    /// floats.  Per-thread buffers are lazily allocated on first call and
    /// reused thereafter.
    void ForwardBatch(const float* flat_inputs, int input_length,
                      int batch_size, float* logits_out);

    // -----------------------------------------------------------------
    //  Training — classification
    // -----------------------------------------------------------------

    /// Single-sample SGD step (forward + backward + weight update).
    /// Classification only — throws `std::logic_error` if the network
    /// was built with `TaskType::Regression`.
    /// `class_weights` (optional, length GetNumOutputs()) scales the
    /// per-class loss; pass nullptr for uniform weighting.
    void TrainStep(const float* raw_input, int input_length, int target_class,
                   float learning_rate, float momentum = 0.0f,
                   float weight_decay = 0.0f,
                   const float* class_weights = nullptr);

    /// Mini-batch parallel step from contiguous data.
    /// `flat_inputs` is `batch_size * input_length` contiguous floats.
    /// Forward+backward run in parallel for each sample, gradients are
    /// reduced (averaged), then a single weight update is applied.
    /// Per-thread buffers are lazily allocated and reused.
    /// Classification only — throws `std::logic_error` on Regression nets.
    void TrainBatch(const float* flat_inputs, int input_length,
                    const int* targets, int batch_size,
                    float learning_rate, float momentum = 0.0f,
                    float weight_decay = 0.0f,
                    const float* class_weights = nullptr);

    /// Iterate `sample_count` samples and dispatch `TrainBatch` in chunks
    /// of `batch_size` (the final chunk may be smaller).
    /// `flat_inputs` is `sample_count * input_length` contiguous floats.
    /// Throws `std::invalid_argument` if `batch_size <= 0` or
    /// `sample_count < 0`.  Classification only — throws
    /// `std::logic_error` on Regression nets.
    ///
    /// `shuffle_seed`:
    ///   - 0 (default): no shuffle, samples are processed in input order.
    ///   - nonzero: deterministic shuffle for this call, seeded by this
    ///     value.  Pass a different seed each epoch (e.g. epoch index)
    ///     for a fresh reproducible permutation.  HCNN owns persistent
    ///     gather buffers used by the shuffle path — after the first
    ///     shuffled epoch, no further allocations occur as long as
    ///     `batch_size` and `sample_count` do not grow.
    void TrainEpoch(const float* flat_inputs, int input_length,
                    const int* targets, int sample_count, int batch_size,
                    float learning_rate, float momentum = 0.0f,
                    float weight_decay = 0.0f,
                    const float* class_weights = nullptr,
                    unsigned shuffle_seed = 0);

    // -----------------------------------------------------------------
    //  Training — regression
    // -----------------------------------------------------------------
    //
    // Counterparts of TrainStep / TrainBatch / TrainEpoch for networks
    // built with `TaskType::Regression`.  The only differences from
    // the classification counterparts are:
    //   - `target` / `flat_targets` are `const float*` pointers to
    //     contiguous real-valued target vectors instead of integer
    //     class indices.
    //   - The loss is MSE (default for Regression) instead of softmax +
    //     cross-entropy.
    //   - No `class_weights` parameter — per-sample reweighting for
    //     regression can be added later if needed.
    //
    // Each method throws `std::logic_error` if called on a Classification
    // network.

    /// Single-sample regression training step.  `target` must point to
    /// `GetNumOutputs()` floats.
    void TrainStepRegression(const float* raw_input, int input_length,
                             const float* target, float learning_rate,
                             float momentum = 0.0f,
                             float weight_decay = 0.0f);

    /// Mini-batch regression training step from contiguous data.
    /// `flat_inputs` is `batch_size * input_length` contiguous floats.
    /// `flat_targets` is `batch_size * GetNumOutputs()` contiguous floats.
    void TrainBatchRegression(const float* flat_inputs, int input_length,
                              const float* flat_targets, int batch_size,
                              float learning_rate, float momentum = 0.0f,
                              float weight_decay = 0.0f);

    /// Full-epoch regression training from contiguous data.
    /// `flat_inputs` is `sample_count * input_length` contiguous floats.
    /// `flat_targets` is `sample_count * GetNumOutputs()` contiguous floats.
    /// Same shuffle semantics as `TrainEpoch`.
    void TrainEpochRegression(const float* flat_inputs, int input_length,
                              const float* flat_targets,
                              int sample_count, int batch_size,
                              float learning_rate, float momentum = 0.0f,
                              float weight_decay = 0.0f,
                              unsigned shuffle_seed = 0);

    // -----------------------------------------------------------------
    //  Sizing accessors (everything a consumer needs to size buffers)
    // -----------------------------------------------------------------
    int GetStartDim() const;
    int GetStartN() const;
    int GetInputChannels() const;
    int GetNumOutputs() const;
    TaskType GetTaskType() const;
    LossType GetLossType() const;

    // -----------------------------------------------------------------
    //  Weight serialization
    // -----------------------------------------------------------------

    /// @brief Total number of trainable parameters across all layers.
    [[nodiscard]] size_t GetWeightCount() const;

    /// @brief Flatten all trainable parameters into a contiguous vector.
    ///
    /// Layout (contiguous, in order):
    ///   for each conv layer i = 0 .. num_conv-1:
    ///     conv[i] kernel   (c_out * c_in * K floats)
    ///     conv[i] bias     (c_out floats, or 0 if bias disabled)
    ///   readout weights    (num_outputs * input_features floats)
    ///   readout bias       (num_outputs floats)
    [[nodiscard]] std::vector<float> GetWeights() const;

    /// @brief Restore all trainable parameters from a contiguous vector.
    /// @throws std::invalid_argument if blob.size() != GetWeightCount().
    void SetWeights(const std::vector<float>& blob);


private:
    std::unique_ptr<HCNNNetwork> net_;

    // Persistent shuffle index used by TrainEpoch / TrainEpochRegression.
    // Sized to sample_count; grows on demand, never shrinks.
    std::vector<int> shuffle_idx_;

    // Per-batch gather buffers for the shuffle path.  Sized to one batch
    // (batch_size * input_length for inputs, batch_size for classification
    // targets, batch_size * num_outputs for regression targets).  Grown on
    // demand, never shrink.  The no-shuffle path is zero-copy (direct
    // pointer arithmetic into the caller's buffer).
    std::vector<float> shuffle_inputs_;       // batch_size * input_length
    std::vector<int>   shuffle_targets_;      // batch_size (classification)
    std::vector<float> shuffle_targets_f_;    // batch_size * num_outputs (regression)
};

} // namespace hcnn
