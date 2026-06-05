// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

#include "HCNNConv.h"
#include "HCNNPool.h"
#include "HCNNReadout.h"
#include <functional>
#include <memory>
#include <vector>
#include <stdexcept>

namespace hcnn {

class ThreadPool;

/// Task the network is being trained for.  Controls the training API
/// shape (integer class targets vs. float regression targets), the
/// interpretation of raw readout outputs, and — unless overridden via
/// `LossType` — the default loss function.
///
/// - `Classification`: raw readout outputs are logits, trained through
///   softmax + cross-entropy loss.  Use `train_step` / `train_batch` /
///   `TrainEpoch` with integer class targets.
/// - `Regression`: raw readout outputs are real-valued predictions,
///   trained through MSE loss (by default).  Use
///   `train_step_regression` / `train_batch_regression` /
///   `TrainEpochRegression` with `const float*` targets of length
///   `num_outputs`.
enum class TaskType { Classification, Regression };

/// Loss function used by training.
///
/// - `Default`: infer from `TaskType` — CrossEntropy for Classification,
///   MSE for Regression.  This is the normal path.
/// - `CrossEntropy`: softmax + cross-entropy.  Only valid when
///   `TaskType == Classification`.
/// - `MSE`: mean squared error.  Only valid when
///   `TaskType == Regression`.
///
/// Additional loss functions (Huber, L1, ...) will slot in here as
/// future enum values with no API break — the gradient dispatch in
/// `HCNNNetwork::compute_classification_grad` and
/// `HCNNNetwork::compute_regression_grad` is designed to accommodate
/// them with a single case statement.
enum class LossType { Default, CrossEntropy, MSE };

/**
 * @class HCNNNetwork
 * @brief Internal pipeline orchestrator: input embedding → conv/pool stack
 *        → readout, plus the inference, training, and batch-dispatch logic
 *        that drives them.
 *
 * Wrapped by `HCNN` (the canonical SDK front door).  Most consumers should
 * use `HCNN`; this class is re-exported for power users who need direct
 * weight access (e.g. for serialization or gradient checking) or who want
 * to bypass the wrapper for custom training loops.
 *
 * Owns:
 *   - `vector<HCNNConv>` and `vector<HCNNPool>` interleaved per `is_conv_layer`
 *   - `HCNNReadout` (sized lazily by `randomize_all_weights` based on
 *     readout type and final layer geometry)
 *   - A `ThreadPool` shared across all layers
 *   - Persistent per-thread inference buffers (`ibufs_`)
 *   - Persistent per-thread training buffers and gradient accumulators
 *     (`tbufs_`, `accum_`), allocated lazily on first `train_batch`
 *   - Persistent ping-pong scratch (`fwd_buf1_` / `fwd_buf2_`) used by
 *     single-sample `forward()`
 *
 * Threading: three strategies coexist but never nest.  `train_batch` and
 * `forward_batch` parallelize across samples; `HCNNConv::forward` /
 * `backward` parallelize across vertices when DIM is large enough.  The
 * RAII `LayerThreadGuard` disables per-layer vertex threading during batch
 * dispatch to keep the non-reentrant ThreadPool safe.  The RAII
 * `BNStatsGuard` suppresses per-sample running-stats EMA updates during
 * batch-parallel forward passes (race-free reduction happens after).  The
 * RAII `EvalModeGuard` makes `forward()` / `forward_batch()` observably
 * const w.r.t. batch-norm training state.
 *
 * Non-copyable, non-movable -- the owned ThreadPool has live worker
 * threads.
 */
class HCNNNetwork {
public:
    HCNNNetwork(int start_dim, int num_outputs = 10,
                int input_channels = 1,
                TaskType task_type = TaskType::Classification,
                LossType loss_type = LossType::Default,
                size_t num_threads = 0);
    ~HCNNNetwork();

    HCNNNetwork(const HCNNNetwork&) = delete;
    HCNNNetwork& operator=(const HCNNNetwork&) = delete;
    HCNNNetwork(HCNNNetwork&&) = delete;
    HCNNNetwork& operator=(HCNNNetwork&&) = delete;

    void add_conv(int c_out, Activation activation = Activation::RELU,
                  bool use_bias = true, bool use_batchnorm = false);
    void add_pool(PoolType type = PoolType::MAX);

    /// Set training mode (true) or eval mode (false) for all layers with BN.
    void set_training(bool training) const;

    /// Configure the optimizer for all layers. Resets timestep.
    void set_optimizer(OptimizerType type, float beta1 = 0.9f,
                       float beta2 = 0.999f, float eps = 1e-8f);

    /// Initialize all weights.  scale > 0: uniform [-scale, +scale].
    /// scale <= 0 (default): Xavier/Glorot uniform per layer.
    void randomize_all_weights(float scale = 0.0f, unsigned seed = 42);

    void embed_input(const float* raw_input, int input_length,
                     float* first_layer_activations) const;

    /// Single-sample forward pass.  Reads existing batch-norm mode (caller's
    /// responsibility to call set_training(false) before inference).  Reuses
    /// persistent ping-pong scratch buffers — no per-call allocation in steady
    /// state.  Not thread-safe with respect to other forward() / train_step()
    /// calls on the same network (the persistent scratch is shared).
    void forward(const float* first_layer_activations, float* logits) const;

    /// Batch inference: embed + forward for multiple samples in parallel.
    /// `flat_inputs` is `batch_size * input_length` floats (contiguous,
    /// row-major).  `logits_out` must have `batch_size * num_outputs` floats.
    /// For classification nets these are raw (pre-softmax) logits; for
    /// regression nets they are the raw real-valued predictions.
    void forward_batch(const float* flat_inputs, int input_length,
                       int batch_size, float* logits_out);

    void train_step(const float* raw_input, int input_length,
                    int target_class, float learning_rate, float momentum = 0.0f,
                    float weight_decay = 0.0f,
                    const float* class_weights = nullptr);

    /// Mini-batch training: process batch_size samples in parallel, average
    /// gradients, then apply a single weight update. Requires ThreadPool.
    /// `flat_inputs` is `batch_size * input_length` contiguous floats.
    /// class_weights: optional per-class loss scaling (length num_outputs).
    /// Classification only — throws std::logic_error if task_type_ is
    /// Regression.
    void train_batch(const float* flat_inputs, int input_length,
                     const int* targets, int batch_size,
                     float learning_rate, float momentum = 0.0f,
                     float weight_decay = 0.0f,
                     const float* class_weights = nullptr);

    /// Regression counterpart of train_step.  `target` is a pointer to
    /// `num_outputs` floats — the per-output regression targets for a
    /// single sample.  Throws std::logic_error if task_type_ is
    /// Classification.
    void train_step_regression(const float* raw_input, int input_length,
                               const float* target, float learning_rate,
                               float momentum = 0.0f,
                               float weight_decay = 0.0f);

    /// Regression counterpart of train_batch.  `flat_targets` is
    /// `batch_size * num_outputs` contiguous floats (row-major).
    /// Throws std::logic_error if task_type_ is Classification.
    void train_batch_regression(const float* flat_inputs, int input_length,
                                const float* flat_targets, int batch_size,
                                float learning_rate, float momentum = 0.0f,
                                float weight_decay = 0.0f);

    int get_start_dim() const { return start_dim; }
    int get_start_N() const { return 1 << start_dim; }
    int get_input_channels() const { return input_channels; }
    int get_num_outputs() const { return num_outputs; }
    TaskType get_task_type() const { return task_type_; }
    LossType get_loss_type() const { return loss_type_; }

    HCNNConv& get_conv(size_t i) { return conv_layers[i]; }
    const HCNNConv& get_conv(size_t i) const { return conv_layers[i]; }
    HCNNReadout& get_readout() { return readout; }
    const HCNNReadout& get_readout() const { return readout; }
    size_t get_num_conv() const { return conv_layers.size(); }
    size_t get_num_pool() const { return pool_layers.size(); }
    const std::vector<bool>& get_layer_types() const { return is_conv_layer; }
    const std::vector<int>& get_channel_counts() const { return channel_counts; }

    /// Eagerly allocate all internal work buffers (single-step, batch,
    /// inference).  Each is idempotent — safe to call multiple times.
    void prepare_all_buffers();

private:
    int start_dim;
    int current_dim;
    int num_outputs;
    int input_channels;
    TaskType task_type_;
    LossType loss_type_;
    int adam_timestep_{0};     // Global optimizer timestep (incremented per train_step/train_batch)
    std::vector<HCNNConv> conv_layers;
    std::vector<HCNNPool> pool_layers;
    std::vector<bool> is_conv_layer;
    std::vector<int> channel_counts;
    HCNNReadout readout;
    std::unique_ptr<ThreadPool> thread_pool;

    // --- Persistent batch-training buffers (allocated once, reused every train_batch) ---
    struct LayerInfo { int N; int channels; };

    struct ThreadAccum {
        std::vector<std::vector<float>> conv_kernel_grad;
        std::vector<std::vector<float>> conv_bias_grad;
        std::vector<std::vector<float>> conv_bn_gamma_grad;
        std::vector<std::vector<float>> conv_bn_beta_grad;
        std::vector<std::vector<float>> conv_bn_mean;   // per-conv BN mean accumulator
        std::vector<std::vector<float>> conv_bn_var;    // per-conv BN var accumulator
        std::vector<float> readout_weight_grad;
        std::vector<float> readout_bias_grad;
    };

    struct ThreadBuf {
        struct LayerCache {
            std::vector<float> activation;
            std::vector<float> pre_act;
            std::vector<int> max_indices;
        };
        std::vector<LayerCache> cache;
        std::vector<float> logits, probs, grad_logits;
        std::vector<float> grad_a, grad_b;
        std::vector<float> rw_grad, rb_grad;
        std::vector<std::vector<float>> kg, bg;
        std::vector<std::vector<float>> bn_gg, bn_bg;  // per-conv BN gamma/beta grads
        std::vector<std::vector<float>> bn_save;        // per-conv BN inv_std cache
        std::vector<float> conv_work;     // work buf for HCNNConv::compute_gradients
        std::vector<float> readout_work;  // work buf for HCNNReadout::compute_gradients
    };

    bool batch_bufs_ready{false};
    std::vector<LayerInfo> layer_info_;
    std::vector<ThreadAccum> accum_;
    std::vector<ThreadBuf> tbufs_;

    void prepare_batch_buffers();
    void zero_accumulators();

    // ----- Shared training cores -----------------------------------------
    //
    // train_step / train_step_regression and train_batch / train_batch_regression
    // share an identical forward / backward / weight-update pipeline.  The
    // only thing that differs between classification and regression is the
    // computation of dL/d(logits) for each sample.  We capture that one
    // step in a callable and let the wrappers (the public train_* methods)
    // build the appropriate lambda.
    //
    // The "loss-grad" callbacks receive a `probs_scratch` buffer of size
    // num_outputs floats.  Classification lambdas use it to hold the
    // softmax probabilities (a side effect of compute_classification_grad);
    // regression lambdas ignore it.
    using LossGradStepFn = std::function<void(const float* logits,
                                              float* grad_logits_out,
                                              float* probs_scratch)>;
    using LossGradBatchFn = std::function<void(int sample_idx,
                                               const float* logits,
                                               float* grad_logits_out,
                                               float* probs_scratch)>;

    // Single-sample training core.  Caller is responsible for any task /
    // target validation -- this method trusts its inputs.
    void train_step_impl(const float* raw_input, int input_length,
                         const LossGradStepFn& loss_grad,
                         float learning_rate, float momentum,
                         float weight_decay);

    // Mini-batch training core.  Caller is responsible for any task /
    // target / batch_size validation.
    void train_batch_impl(const float* flat_inputs, int input_length,
                          int batch_size,
                          const LossGradBatchFn& loss_grad,
                          float learning_rate, float momentum,
                          float weight_decay);

    // Compute dL/d(logits) for a single sample under a classification loss.
    // `probs_scratch` must point to at least `num_outputs` floats and
    // receives the softmax of `logits` as a side effect (callers that
    // care about the softmax, e.g. for loss reporting, can read it).
    // Dispatches internally on `loss_type_` so future classification
    // losses (e.g. focal loss) can slot in with no caller change.
    // Throws std::logic_error if called on a Regression task.
    void compute_classification_grad(const float* logits, int target_class,
                                     float class_weight,
                                     float* probs_scratch,
                                     float* grad_logits_out) const;

    // Compute dL/d(logits) for a single sample under a regression loss.
    // `target` is a pointer to `num_outputs` real-valued targets.
    // Dispatches internally on `loss_type_` so future regression losses
    // (Huber, L1, ...) can slot in with no caller change.  Throws
    // std::logic_error if called on a Classification task.
    void compute_regression_grad(const float* logits, const float* target,
                                 float* grad_logits_out) const;

    // --- Persistent inference buffers (allocated once, reused every forward_batch) ---
    struct InferenceBuf {
        std::vector<float> buf1, buf2;
        std::vector<float> embedded;
    };
    bool infer_bufs_ready{false};
    std::vector<InferenceBuf> ibufs_;
    int infer_max_layer_size_{0};

    void prepare_inference_buffers();

    // Persistent scratch for single-sample forward() — sized to the largest
    // layer in the network and reused across calls.
    mutable std::vector<float> fwd_buf1_;
    mutable std::vector<float> fwd_buf2_;
    mutable std::vector<float> fwd_readout_avg_;

    // --- Persistent single-step training buffers (allocated once, reused every train_step) ---
    struct StepCache {
        std::vector<float> activation;
        std::vector<float> pre_act;
        std::vector<float> bn_save;
        std::vector<int> max_indices;
    };
    struct StepBuf {
        std::vector<float> embedded;
        std::vector<StepCache> cache;
        std::vector<int> layer_N;
        std::vector<int> layer_ch;
        std::vector<float> logits, probs, grad_logits;
        std::vector<float> grad_a, grad_b;
        std::vector<float> readout_avg;
    };
    bool step_buf_ready_{false};
    StepBuf step_buf_;
    void prepare_step_buffers();

    // RAII guard to disable per-layer threading during batch dispatch
    // and restore it when the scope exits (including on exception).
    struct LayerThreadGuard {
        std::vector<HCNNConv>& conv;
        std::vector<HCNNPool>& pool_layers;
        ThreadPool* pool;
        LayerThreadGuard(std::vector<HCNNConv>& c, std::vector<HCNNPool>& p, ThreadPool* tp)
            : conv(c), pool_layers(p), pool(tp) {
            for (auto& layer : conv) layer.set_thread_pool(nullptr);
            for (auto& layer : pool_layers) layer.set_thread_pool(nullptr);
        }
        ~LayerThreadGuard() {
            for (auto& layer : conv) layer.set_thread_pool(pool);
            for (auto& layer : pool_layers) layer.set_thread_pool(pool);
        }
    };

    // RAII guard for inference: temporarily forces eval mode (so BN uses
    // running stats and never updates them), and restores the prior per-layer
    // training flag on scope exit (including on exception).  This makes
    // forward() / forward_batch() observably const w.r.t. BN training state.
    // Takes a const reference because HCNNConv::set_training is const-qualified
    // (the training flag is `mutable`).
    struct EvalModeGuard {
        const std::vector<HCNNConv>& layers;
        std::vector<bool> prev_training;
        explicit EvalModeGuard(const std::vector<HCNNConv>& l) : layers(l) {
            prev_training.reserve(layers.size());
            for (const auto& layer : layers) {
                prev_training.push_back(layer.is_training());
                layer.set_training(false);
            }
        }
        ~EvalModeGuard() {
            for (size_t i = 0; i < layers.size(); ++i)
                layers[i].set_training(prev_training[i]);
        }
    };

    // RAII guard to suppress per-sample running-stats EMA updates during
    // batch-parallel forward passes, and restore on scope exit (including on exception).
    struct BNStatsGuard {
        std::vector<HCNNConv>& layers;
        BNStatsGuard(std::vector<HCNNConv>& l) : layers(l) {
            for (auto& layer : layers)
                if (layer.has_batchnorm()) layer.set_skip_running_stats(true);
        }
        ~BNStatsGuard() {
            for (auto& layer : layers)
                if (layer.has_batchnorm()) layer.set_skip_running_stats(false);
        }
    };
};

} // namespace hcnn
