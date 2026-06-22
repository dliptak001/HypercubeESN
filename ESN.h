#pragma once

#include <memory>
#include <vector>
#include "Reservoir.h"
#include "Readout.h"

struct ESNConfig
{
    ReservoirConfig reservoir;
    ReadoutConfig readout;
};


/// @brief Echo-state network implementing the full pipeline:
///        Reservoir -> Readout. The readout sees all N reservoir vertices.
///
/// @note Not thread-safe: even the const prediction methods write to a shared
///       internal scratch buffer, so a single ESN instance must not be driven
///       from multiple threads concurrently. Use one ESN per thread.
///
/// The hypercube dimension is a runtime config field (@c cfg.reservoir.dim);
/// the vertex count is N = 2^dim.
class ESN
{
public:
    explicit ESN(const ESNConfig& cfg);

    // ---------------------------------------------------------------
    //  Reservoir driving
    // ---------------------------------------------------------------

    /// @brief One timestep on the live reservoir: inject this step's inputs
    /// (NumInputs() floats) and Step. No readout training occurs here.
    ///
    /// Open-loop by default (@p feedback == nullptr). To drive closed-loop, pass
    /// @p feedback — NumFeedbackChannels() floats staged RAW (no clamp) on the
    /// reservoir's dedicated feedback port (its own weight block +
    /// feedback_scaling, outside the spectral-radius rescale). This is the only
    /// feedback entry point; the ESN never generates or learns the feedback.
    /// @param inputs   NumInputs() floats (task input for this step).
    /// @param feedback nullptr for open-loop, else NumFeedbackChannels() floats.
    /// @throws std::invalid_argument if @p feedback is non-null but feedback is
    ///         not configured (num_feedback_channels == 0).
    void StepLive(const float* inputs, const float* feedback = nullptr);

    /// @brief Drive the reservoir for @p num_steps without recording states
    /// (washes out the initial transient). @p inputs is row-major,
    /// num_steps * NumInputs() floats (NumInputs() values per timestep, one per
    /// channel). Steps via @ref StepLive open-loop, so no feedback is injected
    /// even on a feedback-configured ESN. Closed-loop drive is the caller's
    /// responsibility via @ref StepLive (passing feedback).
    void Warmup(const float* inputs, size_t num_steps);

    /// @brief Drive the reservoir for @p num_steps and append the full state at
    /// each step to the recorded-states buffer (for batch Train / R2 / NRMSE /
    /// Accuracy). @p inputs has the same layout as @ref Warmup. Steps via
    /// @ref StepLive — strictly open-loop, so the recorded states carry no
    /// feedback drive regardless of num_feedback_channels.
    ///
    /// By default successive calls **accumulate** into one growing batch. Pass
    /// @p clear_recorded = true to first discard everything recorded so far (and
    /// reset the timestep count) so this call starts a fresh batch — the trained
    /// readout and the live reservoir state are left untouched either way.
    void Run(const float* inputs, size_t num_steps, bool clear_recorded = false);

    /// @brief Clear the live reservoir state (zero its activations + history) so
    /// a new input sequence starts from rest. The recorded states and the
    /// trained readout are preserved.
    void ClearReservoir();

    // ---------------------------------------------------------------
    //  Training
    // ---------------------------------------------------------------

    /// @brief Batch-train the readout on collected timesteps [0, train_size).
    /// Requires train_size <= NumCollected(). @p targets layout matches @ref R2.
    void Train(const float* targets, size_t train_size);

    /// @brief Prepare for online (streaming) training by warming up the
    /// reservoir on @p warmup_inputs (same layout as @ref Warmup) to wash out
    /// the initial transient. The readout's CNN is built eagerly at ESN
    /// construction, so no readout init happens here — this is purely the
    /// reservoir washout. (Single-ESN online-user sugar; the ensemble owns its
    /// own warm-up loop.)
    void InitOnline(const float* warmup_inputs, size_t warmup_count);

    /// @brief Single-step online classification training on the live reservoir
    /// state against @p target_class.
    void TrainLiveStep(float target_class, float lr, float weight_decay);

    /// @brief Copy the current live reservoir state (Size() floats = all N
    /// vertices) into @p out, for external mini-batch accumulation.
    void CopyLiveState(float* out) const;

    /// No-weight_decay overload: inherits `cfg.readout.weight_decay`.
    void TrainLiveBatch(const float* states, const int* targets, size_t count, float lr);

    /// @brief Mini-batch online classification training on pre-accumulated
    /// states (each Size() floats) with integer class @p targets.
    void TrainLiveBatch(const float* states, const int* targets, size_t count, float lr, float weight_decay);

    /// @brief Single-step online regression training on the live reservoir
    /// state against @p target (NumOutputs() floats).
    void TrainLiveStepRegression(const float* target, float lr, float weight_decay);

    /// @brief Mini-batch online regression training on pre-accumulated states
    /// (each Size() floats) with @p targets (count * NumOutputs()).
    void TrainLiveBatchRegression(const float* states, const float* targets,
                                  size_t count, float lr, float weight_decay);

    // ---------------------------------------------------------------
    //  Prediction & evaluation
    // ---------------------------------------------------------------

    [[nodiscard]] float PredictRaw(size_t timestep) const;
    void PredictRaw(size_t timestep, float* output) const;

    [[nodiscard]] float PredictLiveRaw() const;

    /// Sugar for: CopyLiveState(buf); PredictFromState(buf, output).
    /// Use the explicit two-call form when you need to modify the readout
    /// input (e.g. brand a side channel onto the first few slots).
    void PredictLiveRaw(float* output) const;

    /// Run the readout on a caller-supplied state buffer (Size() floats).
    /// Lets the caller modify the readout input -- e.g. brand a side channel
    /// onto the first few slots -- before prediction, without touching live
    /// reservoir state.
    void PredictFromState(const float* state, float* output) const;

    /// @brief R-squared on collected timesteps [start, start+count).
    /// @param targets  Must span timesteps [0, start+count): for regression,
    ///                 (start+count)*num_outputs floats (row-major); for
    ///                 classification, (start+count) floats.  The method
    ///                 indexes from targets[start*num_outputs].
    /// @param start
    /// @param count
    [[nodiscard]] double R2(const float* targets, size_t start, size_t count) const;

    /// @param targets  Same layout contract as R2. @param start @param count
    [[nodiscard]] double NRMSE(const float* targets, size_t start, size_t count) const;

    /// @param labels  Must span timesteps [0, start+count): (start+count)
    ///               floats (class indices).  Indexed from labels[start]. @param start @param count
    [[nodiscard]] double Accuracy(const float* labels, size_t start, size_t count) const;

    [[nodiscard]] size_t NumOutputs() const;

    // ---------------------------------------------------------------
    //  State access
    // ---------------------------------------------------------------

    /// @brief All collected states, row-major: NumCollected() * Size().
    [[nodiscard]] std::vector<float> CollectedStates() const;

    // ---------------------------------------------------------------
    //  Accessors
    // ---------------------------------------------------------------
    [[nodiscard]] size_t NumCollected() const { return num_collected_; }
    [[nodiscard]] size_t NumInputs() const { return num_inputs_; }

    /// Hypercube dimension of the underlying reservoir (cfg.reservoir.dim).
    [[nodiscard]] size_t Dim() const { return reservoir_->Dim(); }
    /// Reservoir neuron count N = 2^Dim().
    [[nodiscard]] size_t Size() const { return n_; }

    /// Number of external feedback channels D (= cfg.reservoir.num_feedback_channels).
    /// 0 means feedback is not configured; @ref StepLive expects this many
    /// feedback values per step when driven closed-loop (non-null feedback).
    [[nodiscard]] size_t NumFeedbackChannels() const
    {
        return esn_config_.reservoir.num_feedback_channels;
    }

    // --- Config & persistence ---

    [[nodiscard]] ESNConfig GetConfig() const;

    struct ReadoutState
    {
        std::vector<double> weights;
        bool is_trained = false;
    };

    [[nodiscard]] ReadoutState GetReadoutState() const;
    void SetReadoutState(const ReadoutState& state);

private:
    std::unique_ptr<Reservoir> reservoir_;
    Readout readout_;
    ESNConfig esn_config_;

    size_t n_ = 0; // reservoir neuron count N = 2^dim
    size_t num_inputs_ = 1;

    std::vector<float> states_;
    size_t num_collected_ = 0;

    static ReadoutConfig MakeReadoutConfig(const ESNConfig& cfg);

    const float* ReadoutInput(size_t timestep) const;
    [[nodiscard]] std::vector<float> ReadoutStates(size_t start, size_t count) const;

    mutable std::vector<float> scratch_state_;
};
