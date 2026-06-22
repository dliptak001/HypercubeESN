#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

namespace hcnn
{
    class HCNN;
}

enum class ReadoutTask { Regression, Classification };

/// Activation applied after each Conv layer in the Readout's CNN stack.
/// Mirrors `hcnn::Activation` to keep HCNN.h out of this public header
/// (PIMPL discipline -- mapping lives in Readout.cpp).
enum class ReadoutActivation { TANH, RELU, LEAKY_RELU, NONE };

/// Cosine annealing LR for progress in [0, 1]. Shared between batch and
/// streaming training paths so the schedule shape is identical.
inline float CosineLR(float progress, float lr_max, float lr_min)
{
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    constexpr float pi = 3.14159265358979323846f;
    return lr_min + 0.5f * (lr_max - lr_min) * (1.0f + std::cos(pi * progress));
}

/// HCNN readout architecture and training parameters.
/// Must stay trivially copyable (POD) for checkpoint serialization.
struct ReadoutConfig
{
    size_t dim = 0;
    ///< Input feature dim: features per sample = 2^dim. Must be set (>= 5) at construction (the CNN is built in the ctor).
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

/// HypercubeCNN-based learned readout operating on raw reservoir state
/// (N = 2^DIM floats per timestep).
///
/// Data path: raw state -> Conv+Pool stack -> Flatten -> Linear -> output.
///
/// Architecture auto-sized from DIM: min(DIM-2, 2) Conv+Pool pairs,
/// channels doubling per layer. Override via ReadoutConfig::num_layers.
///
/// PIMPL: hcnn::HCNN held via unique_ptr; #include "HCNN.h" in .cpp only.
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

    /// Train on collected reservoir states (row-major, 2^config.dim floats
    /// per sample). Uses the ReadoutConfig supplied at construction.
    void Train(const float* states, const float* targets, size_t num_samples);

    // ----- Online (streaming) training -----
    //
    // The CNN is built eagerly in the constructor (no separate init step):
    // net_ is ready to predict and to train (Adam + prepared buffers) from
    // construction. Stream straight into the TrainOnline* methods.

    /// Single-sample online step (classification).
    void TrainOnlineStep(const float* state, int target_class,
                         float lr, float weight_decay = 0.0f);

    /// Mini-batch online step (classification). Parallelized via HCNN::TrainBatch.
    void TrainOnlineBatch(const float* states, const int* targets,
                          size_t count, float lr, float weight_decay = 0.0f);

    /// Single-sample online step (regression).
    void TrainOnlineStepRegression(const float* state, const float* target,
                                   float lr, float weight_decay = 0.0f);

    /// Mini-batch online step (regression).
    void TrainOnlineBatchRegression(const float* states, const float* targets,
                                    size_t count, float lr, float weight_decay = 0.0f);

    // ----- Prediction -----

    /// Multi-output: writes num_outputs floats. Regression: raw network output.
    /// Classification: logits.
    void PredictRaw(const float* state, float* output) const;

    /// Returns predicted class index (argmax over logits).
    [[nodiscard]] int PredictClass(const float* state) const;

    // ----- Evaluation -----

    /// R-squared (averaged across outputs for multi-output regression).
    [[nodiscard]] double R2(const float* states, const float* targets,
                            size_t num_samples) const;

    /// Classification accuracy (argmax vs label for multi-class).
    [[nodiscard]] double Accuracy(const float* states, const float* labels,
                                  size_t num_samples) const;

    // ----- Accessors -----

    [[nodiscard]] size_t NumOutputs() const { return num_outputs_; }
    [[nodiscard]] size_t NumFeatures() const { return num_features_; }
    /// The CNN is built in the ctor, so a Readout is always ready to predict and
    /// has weights worth persisting — net_ is the invariant this reports.
    [[nodiscard]] bool IsTrained() const { return net_ != nullptr; }
    [[nodiscard]] const ReadoutConfig& GetConfig() const { return config_; }

    // ----- Serialization -----

    /// Snapshot the live CNN weights as an opaque blob. Returned by value so the
    /// copy can't go stale behind a later TrainOnline* call.
    [[nodiscard]] std::vector<double> Weights() const;

    /// Load a previously saved weight blob into the (ctor-built) CNN.
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
