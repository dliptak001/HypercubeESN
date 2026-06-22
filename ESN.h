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

    /// @brief One timestep on the reservoir: inject this step's inputs
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
    void ReservoirStep(const float* inputs, const float* feedback = nullptr);

    /// @brief Drive the reservoir for @p num_steps without recording states
    /// (washes out the initial transient). @p inputs is row-major,
    /// num_steps * NumInputs() floats (NumInputs() values per timestep, one per
    /// channel). Steps via @ref ReservoirStep open-loop, so no feedback is injected
    /// even on a feedback-configured ESN. Closed-loop drive is the caller's
    /// responsibility via @ref ReservoirStep (passing feedback).
    ///
    /// This is also the warm-up step for online/streaming training: drive the
    /// reservoir here to wash out the x(0) = 0 transient before the first
    /// @ref TrainStep / @ref TrainStepBatch. The readout CNN is built
    /// eagerly at construction, so no separate readout-init call is needed.
    void ReservoirWarmup(const float* inputs, size_t num_steps);

    /// @brief Drive the reservoir for @p num_steps and append the full state at
    /// each step to the recorded-states buffer (for batch Train / R2 / NRMSE /
    /// Accuracy). @p inputs has the same layout as @ref ReservoirWarmup. Steps via
    /// @ref ReservoirStep — strictly open-loop, so the recorded states carry no
    /// feedback drive regardless of num_feedback_channels.
    ///
    /// By default successive calls **accumulate** into one growing batch. Pass
    /// @p clear_recorded = true to first discard everything recorded so far (and
    /// reset the timestep count) so this call starts a fresh batch — the trained
    /// readout and the reservoir state are left untouched either way.
    void ReservoirRun(const float* inputs, size_t num_steps, bool clear_recorded = false);

    /// @brief Clear the reservoir state (zero its activations + history) so
    /// a new input sequence starts from rest. The recorded states and the
    /// trained readout are preserved.
    void ReservoirClear();

    // ---------------------------------------------------------------
    //  Training
    // ---------------------------------------------------------------

    /// @brief Batch-train the readout on recorded timesteps [0, train_size).
    /// Requires train_size <= NumCollectedStates(). @p targets layout matches @ref R2.
    void Train(const float* targets, size_t train_size);

    /// @brief One streaming gradient step on the reservoir's current state
    /// toward @p target. The task is fixed at construction: for regression,
    /// @p target is NumOutputs() floats; for classification, a single float
    /// holding the class index.
    void TrainStep(const float* target, float lr, float weight_decay = 0.0f);

    /// @brief One streaming gradient step over a mini-batch of pre-accumulated
    /// states (each ReservoirNeuronCount() floats, e.g. from @ref CopyReservoirState). For
    /// regression, @p targets is count * NumOutputs() floats; for
    /// classification, @p targets is count floats (class indices).
    void TrainStepBatch(const float* states, const float* targets, size_t count,
                        float lr, float weight_decay = 0.0f);

    /// @brief Copy the current reservoir state (ReservoirNeuronCount() floats = all N
    /// vertices) into @p out, to accumulate a mini-batch for @ref TrainStepBatch.
    void CopyReservoirState(float* out) const;

    // ---------------------------------------------------------------
    //  Prediction & evaluation
    // ---------------------------------------------------------------

    /// @brief Predict from the reservoir's current state. Returns NumOutputs() floats.
    [[nodiscard]] std::vector<float> Predict() const;

    /// @brief Predict from a recorded timestep (after ReservoirRun). Returns NumOutputs() floats.
    [[nodiscard]] std::vector<float> PredictFromRecorded(size_t timestep) const;

    /// @brief Run the readout on a state buffer you pass in, instead of on the
    /// reservoir's current state.
    ///
    /// @p state is a reservoir state of ReservoirNeuronCount() floats -- usually one you saved
    /// earlier with CopyReservoirState(). Returns NumOutputs() floats.
    ///
    /// Use this when you want to predict from a stored state, or to adjust the
    /// state before the readout sees it (for example, overwriting the first few
    /// entries with an external signal). Unlike Predict(), it never reads the
    /// reservoir.
    [[nodiscard]] std::vector<float> PredictFromState(const float* state) const;

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

    // ---------------------------------------------------------------
    //  State access
    // ---------------------------------------------------------------

    /// @brief All collected states, row-major: NumCollectedStates() * ReservoirNeuronCount().
    [[nodiscard]] std::vector<float> CollectedStates() const;

    // ---------------------------------------------------------------
    //  Accessors
    // ---------------------------------------------------------------
    /// Number of recorded reservoir-state snapshots — one per timestep driven
    /// through @ref ReservoirRun. This is the row count of @ref CollectedStates
    /// and the valid index range [0, n) for @ref PredictFromRecorded.
    [[nodiscard]] size_t NumCollectedStates() const { return num_collected_; }
    [[nodiscard]] size_t NumInputs() const { return num_inputs_; }
    [[nodiscard]] size_t NumOutputs() const { return readout_.NumOutputs(); }

    /// Hypercube dimension of the underlying reservoir (cfg.reservoir.dim).
    [[nodiscard]] size_t ReservoirHypercubeDimension() const { return reservoir_->Dim(); }
    /// Reservoir neuron count N = 2^ReservoirHypercubeDimension().
    [[nodiscard]] size_t ReservoirNeuronCount() const { return n_; }

    /// Number of external feedback channels D (= cfg.reservoir.num_feedback_channels).
    /// 0 means feedback is not configured; @ref ReservoirStep expects this many
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
