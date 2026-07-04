#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

namespace hcnn
{
    class HCNN;
}

/// Which kind of task the readout learns; fixed at construction. Regression
/// predicts continuous values; Classification predicts a discrete class label.
enum class ReadoutTask { Regression, Classification };

/// Activation applied after each Conv layer in the Readout's CNN stack.
/// Mirrors `hcnn::Activation` to keep HCNN.h out of this public header
/// (PIMPL discipline -- mapping lives in Readout.cpp).
enum class ReadoutActivation { TANH, RELU, LEAKY_RELU, NONE };

/// Cosine-annealing learning-rate schedule: eases the rate smoothly from
/// @p lr_max (at @p progress = 0) down to @p lr_min (at @p progress = 1).
/// Shared between the batch and streaming training paths so the schedule shape
/// is identical. @p progress is clamped to [0, 1].
inline float CosineLR(float progress, float lr_max, float lr_min)
{
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    constexpr float pi = 3.14159265358979323846f;
    return lr_min + 0.5f * (lr_max - lr_min) * (1.0f + std::cos(pi * progress));
}

/// Exponential-decay learning-rate schedule: geometric interpolation
/// lr_max * (lr_min/lr_max)^progress, from @p lr_max (at @p progress = 0) down
/// to exactly @p lr_min (at @p progress = 1). Where cosine holds the rate high
/// early and dives late, this drops by a constant *ratio* per unit progress —
/// linear in log-space. Drop-in alternative to @ref CosineLR (same signature
/// and clamping); requires @p lr_max and @p lr_min > 0.
inline float ExponentialDecayLR(float progress, float lr_max, float lr_min)
{
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    return lr_max * std::pow(lr_min / lr_max, progress);
}

/// @brief Architecture and training settings for the @ref Readout CNN.
///
/// The defaults are sensible for a first run; you mainly set @c dim,
/// @c num_outputs, and @c task. The fields split into two groups: the network
/// *shape* (dim, num_outputs, task, num_layers, conv_channels, activation) and
/// the *training* hyperparameters (epochs, batch_size, the learning-rate
/// schedule, weight_decay, momentum, seed).
///
/// Must stay trivially copyable (POD) so it can be written into a checkpoint.
struct ReadoutConfig
{
    size_t dim = 0; ///< Input feature dim: features per sample = 2^dim.
    int num_outputs = 1; ///< Classes (classification) or targets (regression).
    ReadoutTask task = ReadoutTask::Regression;
    int num_layers = 1; ///< Conv+Pool pairs. 0 = auto: min(DIM-2, 2).
    int conv_channels = 16; ///< Base channels (doubles per layer).
    int epochs = 200;
    int batch_size = 32;
    float lr_max = 0.0015f; ///< Cosine annealing peak. Keep <= 0.005 to avoid NaN.
    float lr_min_frac = 0.01f; ///< Floor = lr_max * lr_min_frac.
    int lr_decay_epochs = 0; ///< Cosine decay horizon. 0 = use `epochs`.
    float weight_decay = 0.0f;
    float momentum = 0.0f; ///< SGD momentum (heavy-ball). 0 = plain SGD. 0.9 typical for CNN.
    unsigned seed = 42; ///< CNN weight initialization seed.
    ReadoutActivation activation = ReadoutActivation::TANH; ///< Per-Conv-layer activation.
};

/// @brief The **trainable half of an @ref ESN**: a small convolutional network
/// (CNN) that maps one reservoir state — N = 2^dim floats — to the task output.
///
/// In an ESN the reservoir is fixed and only the readout learns (see @ref ESN
/// for the whole picture). This class *is* that learner. Each timestep it takes
/// the reservoir's N-number state as its input and produces either a regression
/// vector or class logits, depending on the @ref ReadoutTask chosen at construction.
///
/// ## Data path
/// ```
///   state[N] ──▶ Embed ──▶ [ Conv + Pool ] × L ──▶ Flatten ──▶ Linear ──▶ output
///                          channels double each layer
/// ```
/// The stack is auto-sized from @c dim: by default L = min(dim - 2, 2) Conv+Pool
/// pairs (override with @ref ReadoutConfig::num_layers), the first conv using
/// @ref ReadoutConfig::conv_channels channels and doubling each layer.
///
/// ## Lifecycle
/// Pick a training path:
///   - **Batch** — collect a set of states, then @ref Train once over all of them.
///   - **Streaming / online** — interleave @ref TrainStep (one state) or
///     @ref TrainStepBatch (a mini-batch) with whatever drives the reservoir.
///
/// Then @ref PredictRaw / @ref PredictClass to use it, and @ref R2 / @ref Accuracy
/// to score it. Save and reload the learned weights with @ref Weights / @ref SetState.
///
/// @note PIMPL: the underlying @c hcnn::HCNN is held by unique_ptr so HCNN.h
///       stays out of this public header (it is included only in Readout.cpp).
class Readout
{
public:
    explicit Readout(const ReadoutConfig& cfg);
    ~Readout();
    Readout(Readout&&) noexcept;
    Readout& operator=(Readout&&) noexcept;

    Readout(const Readout&) = delete;
    Readout& operator=(const Readout&) = delete;

    // ----- Batch training -----

    /// @brief Batch-train the CNN on @p num_samples reservoir states (row-major,
    /// 2^dim floats per sample) against their @p targets, using the epoch count,
    /// learning-rate schedule, and other hyperparameters from the construction-time
    /// @ref ReadoutConfig.
    ///
    /// @p targets layout: for regression, @p num_samples * num_outputs floats; for
    /// classification, @p num_samples floats (class indices). Calling @ref Train
    /// again **continues** from the current weights rather than starting fresh —
    /// construct a new Readout for an independent fit.
    void Train(const float* states, const float* targets, size_t num_samples);

    // ----- Streaming training -----
    //
    // One gradient step at a time, interleaved with whatever drives the
    // reservoir. Both methods dispatch on the construction-time task.

    /// @brief One streaming (online) gradient step on a single @p state. @p target
    /// is num_outputs floats for regression, or a single class-index float (cast
    /// to int) for classification.
    /// @param lr           learning rate for this step.
    /// @param weight_decay L2 regularization strength (0 = off).
    void TrainStep(const float* state, const float* target,
                   float lr, float weight_decay = 0.0f);

    /// @brief One streaming gradient step over @p count states (each num_features
    /// floats). @p targets is count * num_outputs floats for regression, or count
    /// class-index floats for classification. Runs the batch in parallel.
    void TrainStepBatch(const float* states, const float* targets,
                        size_t count, float lr, float weight_decay = 0.0f);

    // ----- Prediction -----

    /// @brief Run the network on one @p state and write num_outputs floats to
    /// @p output. Regression: the raw network output. Classification: the raw
    /// class logits (use @ref PredictClass for the argmax label).
    void PredictRaw(const float* state, float* output) const;

    /// @brief Predicted class index — the argmax over the classification logits.
    [[nodiscard]] int PredictClass(const float* state) const;

    // ----- Evaluation -----

    /// @brief R² (coefficient of determination) over @p num_samples (state, target)
    /// pairs, averaged across outputs for multi-output regression. 1.0 is a perfect
    /// fit, 0.0 is no better than always predicting the mean. (Regression metric.)
    [[nodiscard]] double R2(const float* states, const float* targets,
                            size_t num_samples) const;

    /// @brief Classification accuracy over @p num_samples (state, label) pairs —
    /// the fraction predicted correctly. Multi-class compares argmax to the label;
    /// a single output is thresholded at 0. (Classification metric.)
    [[nodiscard]] double Accuracy(const float* states, const float* labels,
                                  size_t num_samples) const;

    // ----- Accessors -----

    /// @brief Size of one prediction: regression targets, or number of classes.
    [[nodiscard]] size_t NumOutputs() const { return num_outputs_; }
    /// @brief Length of the input state vector the network expects, N = 2^dim.
    [[nodiscard]] size_t NumFeatures() const { return num_features_; }
    /// @brief Always true — reports that the network exists and has weights worth
    /// persisting (the net_ invariant), not "has been fed training data".
    [[nodiscard]] bool IsTrained() const { return net_ != nullptr; }
    [[nodiscard]] const ReadoutConfig& GetConfig() const { return config_; }

    // ----- Serialization -----

    /// @brief Snapshot the live CNN weights as an opaque blob, returned by value so
    /// the copy can't go stale behind a later TrainStep* call (streaming training
    /// mutates the network in place). Pair with @ref SetState.
    [[nodiscard]] std::vector<double> Weights() const;

    /// @brief Load a weight blob from @ref Weights back into the network.
    /// An empty blob is ignored.
    void SetState(std::vector<double> weights);

private:
    std::unique_ptr<hcnn::HCNN> net_;
    ReadoutConfig config_;
    size_t num_features_ = 0;
    size_t num_outputs_ = 1;

    mutable std::vector<float> scratch_embedded_;
    mutable std::vector<float> scratch_pred_;

    void build_architecture();
};
