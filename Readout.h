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
/// Optimal seed for DIM >= 10, NARMA_20 -> best= 0.194107 (seed 66), best= 0.197295 (seed 119)
struct ReadoutConfig
{
    size_t dim = 0; ///< Input feature dim: features per sample = 2^dim. Must be set (>= 5) before Train/InitOnline.
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
    bool verbose = false; ///< Print per-epoch lr to stdout.
    bool verbose_train_acc = false; ///< Also print train accuracy/MSE each epoch.
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
    void Train(const float* states, const float* targets,
               size_t num_samples);

    // ----- Online (streaming) training -----

    /// Initialize for online training. Builds the CNN architecture and
    /// sets the Adam optimizer. Architecture and seed come from the
    /// ReadoutConfig passed at construction.
    void InitOnline();

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

    /// Scalar prediction. Asserts num_outputs == 1.
    [[nodiscard]] float PredictRaw(const float* state) const;

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
    [[nodiscard]] bool IsTrained() const { return trained_; }
    [[nodiscard]] const ReadoutConfig& GetConfig() const { return config_; }

    // ----- Serialization -----

    /// Flattened CNN weights (opaque blob). Lazily synced from live network.
    [[nodiscard]] const std::vector<double>& Weights() const;

    /// Restore a previously trained state. Rebuilds the CNN from config + weights.
    void SetState(std::vector<double> weights);

private:
    std::unique_ptr<hcnn::HCNN> net_;
    ReadoutConfig config_;
    bool trained_ = false;
    size_t num_features_ = 0;
    size_t num_outputs_ = 1;

    mutable std::vector<double> weights_blob_;

    mutable std::vector<float> scratch_embedded_;
    mutable std::vector<float> scratch_pred_;

    void build_architecture();
    void flatten_weights();
    void rebuild_from_blob();
};
