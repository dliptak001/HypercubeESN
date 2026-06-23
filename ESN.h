#pragma once

#include <memory>
#include <vector>
#include "Reservoir.h"
#include "Readout.h"

/// @brief Everything needed to build an @ref ESN: the settings for the fixed
/// reservoir plus the settings for the trainable readout. The individual fields
/// are documented on @ref ReservoirConfig and @ref ReadoutConfig.
struct ESNConfig
{
    ReservoirConfig reservoir;
    ReadoutConfig readout;
};


/// @brief An **Echo-State Network (ESN)** —
/// a recurrent neural network for learning from sequences (time series).
///
/// ## The main idea
///
/// Training an ordinary recurrent neural network is hard: you have to tune
/// thousands of recurrent connections with backpropagation-through-time. An ESN
/// sidesteps that by splitting the network into two parts and only training one
/// of them:
///
///   1. A **reservoir** — a large, *fixed* (never-trained) recurrent network.
///      You feed your signal in and it echoes around inside, blending the present
///      input with a fading memory of recent inputs. Think of a still pool of
///      water: drop a stone in (the input) and the ripples (the reservoir's
///      *state*) carry traces of both this splash and the splashes just before it.
///
///   2. A **readout** — a small, *trainable* model that looks at the reservoir's
///      current state (a vector of N numbers, one per reservoir neuron) and turns
///      it into the answer you want. Only this part learns. Here the readout is a
///      small convolutional network (see @ref Readout).
///
/// Because only the readout trains, an ESN learns quickly.
///
/// ```
///   input(t) ──▶ [   Reservoir   ] ──state(t)──▶ [  Readout  ] ──▶ prediction(t)
///                  fixed weights                   trained weights
///                  N = 2^dim neurons               (a small CNN)
/// ```
///
/// ## The usual lifecycle
///
/// You almost always drive an ESN in this order:
///
/// ```
///   1. construct       ESN esn(cfg);
///   2. warm up         esn.ReservoirWarmup(...)    // settle the reservoir, discard early junk states
///   3. record states   esn.ReservoirRun(...)       // drive the reservoir and save the state at each step
///   4. train           esn.Train(targets, ...)     // fit the readout to those saved states
///   5. predict / score esn.Predict(),  esn.R2(...) // use it, and measure how good it is
/// ```
///
/// Steps 2–3 exist because the reservoir starts at all-zeros (x(0) = 0), a state
/// no real input would ever have produced. Driving it for a while first "washes
/// out" that artificial start (the *transient*), so the states you train on
/// reflect the input rather than the cold start.
///
/// ## Vocabulary used throughout this header
///   - **timestep** — one tick of the input sequence; one call to @ref ReservoirStep.
///   - **state**    — the reservoir's N-number snapshot at a timestep; this is the
///                    readout's input. The readout sees *all* N neurons.
///   - **N**        — the number of reservoir neurons, N = 2^dim (see
///                    @ref ReservoirNeuronCount).
///   - **open- vs closed-loop** — whether the reservoir is driven purely by your
///                    external input, or also by a signal fed back in. See
///                    @ref ReservoirStep.
///
/// @note **Not thread-safe.** Even the const prediction methods scribble on a
///       shared internal scratch buffer, so a single ESN instance must not be
///       driven from multiple threads at once. Use one ESN per thread.
class ESN
{
public:
    explicit ESN(const ESNConfig& cfg);

    // ---------------------------------------------------------------
    //  Reservoir driving
    // ---------------------------------------------------------------

    /// @brief Advance the reservoir by one timestep: inject this step's input,
    /// then let the reservoir update. No learning happens here — this only moves
    /// the reservoir forward in time.
    ///
    /// **Open-loop vs closed-loop.** Normally you run *open-loop*: the reservoir
    /// is driven purely by the external input you pass (@p feedback == nullptr).
    /// In *closed-loop* mode you additionally feed a signal back into the
    /// reservoir — typically the previous step's own prediction — which is how an
    /// ESN can generate a sequence by itself, free-running with no external input.
    /// Passing @p feedback switches to closed-loop.
    ///
    /// The ESN never invents or learns the feedback signal itself; *you* compute
    /// it (e.g. from the last @ref Predict) and hand it in here. The values are
    /// staged RAW (no clamp) on the reservoir's dedicated feedback port, which has
    /// its own weight block and feedback_scaling, separate from the input port.
    /// This is the one and only feedback entry point.
    ///
    /// @param inputs   NumInputs() floats — the task input for this step.
    /// @param feedback nullptr for open-loop; otherwise NumFeedbackChannels()
    ///                 floats for closed-loop drive.
    /// @throws std::invalid_argument if @p feedback is non-null but this ESN was
    ///         built without a feedback port (NumFeedbackChannels() == 0). Failing
    ///         loudly here stops a closed-loop call from silently running open-loop.
    void ReservoirStep(const float* inputs, const float* feedback = nullptr);

    /// @brief Drive the reservoir for @p num_steps **without saving any states** —
    /// the warm-up that washes out the cold-start transient.
    ///
    /// The reservoir begins at x(0) = 0, an all-zero state no real input would
    /// produce. Feeding it a stretch of input first lets it "forget" that
    /// artificial start, so the states you later record (@ref ReservoirRun) and
    /// train on genuinely reflect the input. Whatever you drive here is consumed
    /// and discarded — nothing is recorded.
    ///
    /// @p inputs is row-major: @p num_steps * NumInputs() floats, i.e. NumInputs()
    /// values per timestep laid end to end. Each step runs open-loop via
    /// @ref ReservoirStep, so no feedback is injected even on a feedback-capable
    /// ESN; for a closed-loop warm-up, call @ref ReservoirStep yourself.
    ///
    /// This is also the warm-up for online/streaming training: drive here to
    /// settle the reservoir before your first @ref TrainStep / @ref TrainStepBatch.
    void ReservoirWarmup(const float* inputs, size_t num_steps);

    /// @brief Drive the reservoir for @p num_steps and **save the full state after
    /// each step** into an internal buffer, ready for batch @ref Train and the
    /// scoring methods (@ref R2 / @ref NRMSE / @ref Accuracy).
    ///
    /// This is the data-collection step: each recorded state becomes one training
    /// row, to be paired with one target. @p inputs has the same row-major layout
    /// as @ref ReservoirWarmup. Each step runs open-loop, so the recorded states
    /// never carry feedback drive, regardless of NumFeedbackChannels().
    ///
    /// By default, successive calls **accumulate** into one growing batch (so you
    /// can build it up from several sequences). Pass @p clear_recorded = true to
    /// first discard everything recorded so far (and reset the timestep count),
    /// starting a fresh batch. Either way the trained readout and the live
    /// reservoir state are left untouched — only the recorded-state buffer changes.
    void ReservoirRun(const float* inputs, size_t num_steps, bool clear_recorded = false);

    /// @brief Reset the reservoir to rest (zero its neuron activations and
    /// history) so the next input sequence starts from a clean slate.
    ///
    /// Use this between independent sequences so leftover ripples from one don't
    /// bleed into the next. It clears only the live dynamics — your recorded
    /// states (@ref ReservoirRun) and the trained readout are preserved.
    void ReservoirClear();

    // ---------------------------------------------------------------
    //  Training
    // ---------------------------------------------------------------

    /// @brief Train the readout in one batch over recorded timesteps
    /// [0, @p train_size), fitting it to map each recorded state to its target.
    ///
    /// Call this after @ref ReservoirRun has collected the states. @p train_size
    /// must be <= NumCollectedStates(). @p targets uses the same layout as @ref R2
    /// (one target per timestep, indexed from timestep 0).
    /// @throws std::out_of_range if @p train_size > NumCollectedStates().
    void Train(const float* targets, size_t train_size);

    /// @brief Take **one** online (streaming) gradient step, nudging the readout
    /// toward @p target using the reservoir's *current* state.
    ///
    /// This is the streaming alternative to batch @ref Train: instead of
    /// collecting all states then fitting at once, you interleave drive-a-step
    /// (@ref ReservoirStep) with learn-a-step here. The task is fixed at
    /// construction — for regression, @p target is NumOutputs() floats; for
    /// classification, a single float holding the class index.
    /// @param lr           learning rate (the step size of this update).
    /// @param weight_decay optional L2 regularization strength (0 = off).
    void TrainStep(const float* target, float lr, float weight_decay = 0.0f);

    /// @brief Like @ref TrainStep, but takes one gradient step over a mini-batch
    /// of states you have collected yourself (each ReservoirNeuronCount() floats,
    /// e.g. saved with @ref CopyReservoirState).
    ///
    /// For regression, @p targets is @p count * NumOutputs() floats; for
    /// classification, @p targets is @p count floats (class indices).
    void TrainStepBatch(const float* states, const float* targets, size_t count,
                        float lr, float weight_decay = 0.0f);

    /// @brief Copy the reservoir's current state (ReservoirNeuronCount() floats —
    /// all N neurons) into @p out.
    ///
    /// Use it to retain a state for later: to accumulate a mini-batch for
    /// @ref TrainStepBatch, or to predict from a saved state via
    /// @ref PredictFromState. @p out must have room for ReservoirNeuronCount() floats.
    void CopyReservoirState(float* out) const;

    // ---------------------------------------------------------------
    //  Prediction & evaluation
    // ---------------------------------------------------------------

    /// @brief Run the readout on the reservoir's **current** state and return the
    /// prediction (NumOutputs() floats) — typically called right after a @ref ReservoirStep.
    [[nodiscard]] std::vector<float> Predict() const;

    /// @brief Like @ref Predict, but writes the NumOutputs() prediction values
    /// into caller-provided @p out instead of allocating a vector. For hot loops
    /// that reuse a buffer (e.g. an ensemble reading each member every tick).
    /// @p out must have room for NumOutputs() floats.
    void Predict(float* out) const;

    /// @brief Predict from a **recorded** state instead of the live one: runs the
    /// readout on the state saved at @p timestep during @ref ReservoirRun. Returns
    /// NumOutputs() floats.
    /// @throws std::out_of_range if @p timestep >= NumCollectedStates().
    [[nodiscard]] std::vector<float> PredictFromRecorded(size_t timestep) const;

    /// @brief Predict from a state **you supply**, rather than the reservoir's own.
    ///
    /// @p state is a reservoir state of ReservoirNeuronCount() floats — usually one
    /// you saved earlier with @ref CopyReservoirState. Returns NumOutputs() floats.
    ///
    /// Use it to predict from a stored state, or to modify a state before the
    /// readout sees it (e.g. overwriting the first few entries with an external
    /// signal). Unlike @ref Predict, it never touches the live reservoir.
    [[nodiscard]] std::vector<float> PredictFromState(const float* state) const;

    /// @brief Coefficient of determination (R²) over recorded timesteps
    /// [@p start, @p start + @p count) — a goodness-of-fit score where 1.0 is a
    /// perfect fit and 0.0 is no better than always guessing the mean.
    /// @param targets The ground-truth values. Must span timesteps
    ///                [0, @p start + @p count): for regression,
    ///                (start+count) * NumOutputs() floats (row-major); for
    ///                classification, (start+count) floats. The method indexes from
    ///                targets[start * NumOutputs()], so pass the whole array, not a
    ///                pre-sliced one.
    /// @param start   first recorded timestep to score.
    /// @param count   how many timesteps to score.
    /// @throws std::out_of_range if @p start + @p count > NumCollectedStates().
    [[nodiscard]] double R2(const float* targets, size_t start, size_t count) const;

    /// @brief Normalized root-mean-square error over recorded timesteps
    /// [@p start, @p start + @p count) — lower is better (0 = perfect). It is the
    /// RMSE divided by the target's standard deviation, making it scale-free so
    /// you can compare across signals of different magnitudes.
    /// @param targets Same layout contract as @ref R2.
    /// @param start   first recorded timestep to score.
    /// @param count   how many timesteps to score.
    /// @throws std::out_of_range if @p start + @p count > NumCollectedStates().
    [[nodiscard]] double NRMSE(const float* targets, size_t start, size_t count) const;

    /// @brief Classification accuracy over recorded timesteps
    /// [@p start, @p start + @p count) — the fraction predicted correctly (1.0 =
    /// all right). Use this for classification tasks (see @ref R2 / @ref NRMSE for
    /// regression).
    /// @param labels  The true class indices. Must span timesteps
    ///                [0, @p start + @p count): (start+count) floats, indexed from
    ///                labels[start].
    /// @param start   first recorded timestep to score.
    /// @param count   how many timesteps to score.
    /// @throws std::out_of_range if @p start + @p count > NumCollectedStates().
    [[nodiscard]] double Accuracy(const float* labels, size_t start, size_t count) const;

    // ---------------------------------------------------------------
    //  State access
    // ---------------------------------------------------------------

    /// @brief Return a copy of every recorded state, row-major:
    /// NumCollectedStates() rows of ReservoirNeuronCount() floats each.
    ///
    /// Mainly for inspection or for feeding states to other tools; training and
    /// scoring read the internal buffer directly, so you rarely need this.
    [[nodiscard]] std::vector<float> CollectedStates() const;

    // ---------------------------------------------------------------
    //  Accessors
    // ---------------------------------------------------------------
    /// @brief How many reservoir-state snapshots are currently recorded — one per
    /// timestep driven through @ref ReservoirRun. This is the number of rows in
    /// @ref CollectedStates and the valid index range [0, n) for
    /// @ref PredictFromRecorded.
    [[nodiscard]] size_t NumCollectedStates() const { return num_collected_; }
    /// @brief Number of input channels the reservoir expects per timestep — the
    /// length of each @p inputs row passed to the driving methods.
    [[nodiscard]] size_t NumInputs() const { return num_inputs_; }
    /// @brief Size of a single prediction: targets per step (regression) or number
    /// of classes (classification).
    [[nodiscard]] size_t NumOutputs() const { return readout_.NumOutputs(); }

    /// @brief The reservoir's hypercube dimension (cfg.reservoir.dim). The neuron
    /// count is N = 2^dim — see @ref ReservoirNeuronCount.
    [[nodiscard]] size_t ReservoirHypercubeDimension() const { return reservoir_->Dim(); }
    /// @brief Number of reservoir neurons, N = 2^ReservoirHypercubeDimension().
    /// This is the length of every reservoir state vector (and the readout's input
    /// width).
    [[nodiscard]] size_t ReservoirNeuronCount() const { return n_; }

    /// @brief Number of external feedback channels D (cfg.reservoir.num_feedback_channels).
    ///
    /// 0 means this ESN has no feedback port and can only run open-loop. When
    /// D > 0, a closed-loop @ref ReservoirStep expects exactly this many feedback
    /// values per step.
    [[nodiscard]] size_t NumFeedbackChannels() const
    {
        return esn_config_.reservoir.num_feedback_channels;
    }

    // --- Configuration & save/load ---

    /// @brief Return the fully-resolved config this ESN was built from — handy for
    /// rebuilding an identical ESN or for serialization.
    [[nodiscard]] ESNConfig GetConfig() const;

    /// @brief A portable snapshot of the trained readout's weights — everything
    /// that learning produces. Save it to disk to reuse a trained model later
    /// without retraining; the fixed reservoir is fully determined by the config +
    /// seed, so it never needs saving.
    struct ReadoutState
    {
        std::vector<double> weights;
        bool is_trained = false;
    };

    /// @brief Capture the current readout weights (see @ref ReadoutState).
    [[nodiscard]] ReadoutState GetReadoutState() const;
    /// @brief Load previously-saved readout weights (from @ref GetReadoutState)
    /// back into this ESN. A not-trained state is ignored.
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
