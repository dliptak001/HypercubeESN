#pragma once

#include <memory>
#include <vector>
#include "Reservoir.h"
#include "Readout.h"

/// @brief Everything needed to build an @ref ESN: the settings for the fixed
/// reservoir plus the settings for the trainable readout. The individual fields
/// are documented on @ref ReservoirConfig and @ref ReadoutConfig.
///
/// The two fields below live here rather than on either sub-config because they
/// describe the *seam* between the halves — how much of the reservoir's state the
/// readout is shown. Together they set the readout's input shape: it consumes
/// B = readout_slices + (aux_input_dim > 0) blocks of N, so @ref ESN derives
/// `readout.dim = reservoir.dim + log2(B)`. Neither the reservoir nor the readout
/// reads them on its own.
struct ESNConfig
{
    ReservoirConfig reservoir;
    ReadoutConfig readout;

    /// How many reservoir delay-line slices the readout consumes, newest first (>= 1).
    /// 1 (the default) shows it only the current state. Larger values hand it the
    /// temporal memory the reservoir already computes for its recurrent gather and
    /// otherwise discards; must not exceed `reservoir.history_depth`. Independent of
    /// `history_depth` on purpose: widening the readout's view leaves the reservoir's
    /// own dynamics untouched.
    size_t readout_slices = 1;

    /// Width of the auxiliary input block (0 = no aux block). When > 0 the readout
    /// input carries one extra N-wide block holding the caller's raw aux vector,
    /// broadcast onto subcubes, and every Predict/TrainStep must supply that vector.
    /// The aux input feeds only the readout — it never drives the reservoir.
    size_t aux_input_dim = 0;
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
    /// **Drive ports.** Task @p inputs always stage on the input port. Optional
    /// @p external_feedback (when non-null) stages caller-owned closed-loop drive
    /// on the external-feedback port (e.g. previous-step prediction for free-run).
    /// If this ESN was built with @c full_state_feedback, each step **also**
    /// applies internal full-state feedback φ = V·x automatically inside the
    /// reservoir — that path is not passed here (use @ref SetFullStateFeedbackGain).
    ///
    /// @param inputs              NumInputs() floats — the task input for this step.
    /// @param external_feedback   nullptr to skip external feedback; otherwise
    ///                            NumExternalFeedbackChannels() floats.
    /// @throws std::invalid_argument if @p external_feedback is non-null but this
    ///         ESN has no external-feedback port (NumExternalFeedbackChannels()==0).
    void ReservoirStep(const float* inputs, const float* external_feedback = nullptr);

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
    /// values per timestep laid end to end. Each step calls @ref ReservoirStep
    /// without external feedback. **If full-state feedback is enabled, FSF still
    /// applies each step** (internal φ = V·x). For external-feedback warm-up,
    /// call @ref ReservoirStep yourself with the second argument.
    ///
    /// This is also the warm-up for online/streaming training: drive here to
    /// settle the reservoir before your first @ref TrainStep / @ref TrainStepBatch.
    void ReservoirWarmup(const float* inputs, size_t num_steps);

    /// @brief Drive the reservoir for @p num_steps and **save the readout input after
    /// each step** into an internal buffer, ready for batch @ref Train and the
    /// scoring methods (@ref R2 / @ref NRMSE / @ref Accuracy).
    ///
    /// This is the data-collection step: each recorded row (ReadoutInputWidth() floats)
    /// becomes one training row, to be paired with one target. @p inputs has the same
    /// row-major layout as @ref ReservoirWarmup. External feedback is not injected
    /// (pass it via @ref ReservoirStep if needed). **Full-state feedback, if enabled,
    /// still applies** so recorded states match the FSF-closed dynamics.
    ///
    /// @throws std::logic_error if the readout has an auxiliary block
    ///         (aux_input_dim > 0). The batch path has nowhere to take a per-step
    ///         u_raw from — drive such a model online via @ref TrainStep / @ref Predict.
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
    /// toward @p target using the reservoir's *current* readout input.
    ///
    /// This is the streaming alternative to batch @ref Train: instead of
    /// collecting all states then fitting at once, you interleave drive-a-step
    /// (@ref ReservoirStep) with learn-a-step here. The task is fixed at
    /// construction — for regression, @p target is NumOutputs() floats; for
    /// classification, a single float holding the class index.
    /// @param lr           learning rate (the step size of this update).
    /// @param weight_decay optional L2 regularization strength (0 = off).
    /// @param u_raw        the auxiliary input for this step (aux_input_dim floats),
    ///                     or nullptr when the readout has no aux block. See @ref Predict.
    void TrainStep(const float* target, float lr, float weight_decay = 0.0f,
                   const float* u_raw = nullptr);

    /// @brief Like @ref TrainStep, but takes one gradient step over a mini-batch
    /// of readout inputs you have collected yourself (each ReadoutInputWidth()
    /// floats, e.g. saved with @ref CopyReadoutInput).
    ///
    /// For regression, @p targets is @p count * NumOutputs() floats; for
    /// classification, @p targets is @p count floats (class indices).
    void TrainStepBatch(const float* readout_inputs, const float* targets, size_t count,
                        float lr, float weight_decay = 0.0f);

    /// @brief Copy the reservoir's current state (ReservoirNeuronCount() floats —
    /// all N neurons) into @p out. This is the newest delay-line slice only; it is
    /// **not** the readout's input unless the readout was configured with a single
    /// slice and no aux block (see @ref CopyReadoutInput).
    void CopyReservoirState(float* out) const;

    /// @brief Assemble and copy the readout's current input (ReadoutInputWidth()
    /// floats) into @p out — the block-structured vector the readout actually sees.
    ///
    /// The readout input is B = ReadoutBlockCount() blocks of N, laid out on a
    /// (dim + log2 B)-hypercube: one block per reservoir delay-line slice, plus an
    /// optional auxiliary block. @ref ReadoutBlockOf says where each slot lands.
    /// Use it to accumulate a mini-batch for @ref TrainStepBatch, or to inspect what
    /// the readout consumes. @p u_raw follows the @ref Predict contract.
    void CopyReadoutInput(float* out, const float* u_raw = nullptr) const;

    // ---------------------------------------------------------------
    //  Prediction & evaluation
    // ---------------------------------------------------------------

    /// @brief Run the readout on the reservoir's **current** readout input and return
    /// the prediction (NumOutputs() floats) — typically called right after a
    /// @ref ReservoirStep. Only valid when the readout has no auxiliary block; with
    /// one, use the @p u_raw overload.
    /// @throws std::invalid_argument if the readout was built with aux_input_dim > 0.
    [[nodiscard]] std::vector<float> Predict() const;

    /// @brief Like @ref Predict, but writes the NumOutputs() prediction values
    /// into caller-provided @p out instead of allocating a vector. For hot loops
    /// that reuse a buffer (e.g. an ensemble reading each member every tick).
    /// @p out must have room for NumOutputs() floats.
    ///
    /// @param u_raw The auxiliary input for this step — aux_input_dim floats, broadcast
    ///        onto the readout's aux block. Pass nullptr iff the readout has no aux
    ///        block. Supplying one without the other throws, so an aux block is never
    ///        silently zeroed and a stray u_raw is never silently ignored.
    void Predict(float* out, const float* u_raw = nullptr) const;

    /// @brief Predict from a **recorded** state instead of the live one: runs the
    /// readout on the state saved at @p timestep during @ref ReservoirRun. Returns
    /// NumOutputs() floats.
    /// @throws std::out_of_range if @p timestep >= NumCollectedStates().
    [[nodiscard]] std::vector<float> PredictFromRecorded(size_t timestep) const;

    /// @brief Predict from a readout input **you supply**, rather than the reservoir's
    /// own.
    ///
    /// @p readout_input is ReadoutInputWidth() floats — usually one you saved earlier
    /// with @ref CopyReadoutInput. Returns NumOutputs() floats. Use it to predict from
    /// a stored input, or to modify one before the readout sees it. Unlike
    /// @ref Predict, it never touches the live reservoir.
    [[nodiscard]] std::vector<float> PredictFromState(const float* readout_input) const;

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

    /// @brief Return a copy of every recorded readout input, row-major:
    /// NumCollectedStates() rows of ReadoutInputWidth() floats each.
    ///
    /// Mainly for inspection or for feeding rows to other tools; training and
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
    /// This is the length of every reservoir state vector, and of one readout-input
    /// *block* (see @ref ReadoutInputWidth for the whole input).
    [[nodiscard]] size_t ReservoirNeuronCount() const { return n_; }

    /// @brief Length of the readout's input vector: ReadoutBlockCount() * N. The
    /// readout sees a (dim + log2 B)-hypercube built from B blocks of N.
    [[nodiscard]] size_t ReadoutInputWidth() const { return readout_width_; }

    /// @brief Number of N-wide blocks in the readout input, B = readout_slices +
    /// (aux_input_dim > 0). Always a power of two.
    [[nodiscard]] size_t ReadoutBlockCount() const { return readout_blocks_; }

    /// @brief Which block a readout-input **slot** occupies. Slots `[0, readout_slices)`
    /// are reservoir state ages (0 = newest); slot `readout_slices` is the auxiliary
    /// block, when present.
    ///
    /// The mapping is a deliberate permutation, not the identity: an HCNN conv gathers
    /// only single-bit-flip neighbours and has no self term, so a filter can combine
    /// two blocks only when their indices differ in exactly **two** bits. The map pairs
    /// consecutive slots onto such block pairs, putting `{age 0, age 1}` — the velocity
    /// — within reach of one conv filter.
    /// @throws std::out_of_range if @p slot >= ReadoutBlockCount().
    [[nodiscard]] size_t ReadoutBlockOf(size_t slot) const;

    /// @brief Number of external-feedback channels D
    /// (cfg.reservoir.num_external_feedback_channels).
    ///
    /// 0 means no external-feedback port. When D > 0, a non-null second argument
    /// to @ref ReservoirStep must supply exactly D floats.
    [[nodiscard]] size_t NumExternalFeedbackChannels() const
    {
        return esn_config_.reservoir.num_external_feedback_channels;
    }

    /// @brief True if built with cfg.reservoir.full_state_feedback.
    [[nodiscard]] bool FullStateFeedbackEnabled() const
    {
        return reservoir_->FullStateFeedbackEnabled();
    }

    /// @brief Set full-state gain V (length @ref ReservoirNeuronCount). No-op path
    /// throws if FSF was not enabled at construction. Mid-run changes apply on the
    /// next @ref ReservoirStep.
    void SetFullStateFeedbackGain(const float* v, size_t n);

    /// @brief Copy current V into @p v_out (length n == N). Throws if FSF off.
    void GetFullStateFeedbackGain(float* v_out, size_t n) const;

    // --- Configuration & save/load ---

    /// @brief Return the fully-resolved config this ESN was built from — handy for
    /// rebuilding an identical ESN or for serialization.
    ///
    /// Note: when full-state feedback is enabled, the gain V is **not** in the
    /// config (use @ref GetFullStateFeedbackGain). Config + seeds rebuild weight
    /// blocks and B_fsf, not a non-zero V.
    [[nodiscard]] ESNConfig GetConfig() const;

    /// @brief A portable snapshot of the trained readout's weights — everything
    /// that readout learning produces. Save it to disk to reuse a trained model
    /// later without retraining. Reservoir topology/weights are determined by
    /// config + seeds; a non-zero FSF gain V is separate (Get/Set above).
    struct ReadoutState
    {
        std::vector<double> weights;
        bool is_trained = false;
    };

    /// @brief Capture the current readout weights (see @ref ReadoutState).
    [[nodiscard]] ReadoutState GetReadoutState() const;
    /// @brief Load previously-saved readout weights (from @ref GetReadoutState)
    /// back into this ESN. A not-trained state is ignored.
    /// @param mode Eval (default) restores parameters only; ResumeTrain also
    ///        resets CNN optimizer moments for continued online training.
    void SetReadoutState(const ReadoutState& state,
                         ReadoutLoadMode mode = ReadoutLoadMode::Eval);

    /// @brief 1-based epoch of the best restored weights after the last batch
    /// @ref Train with @c ReadoutConfig::restore_best_epoch, else 0. See
    /// @ref Readout::BestEpoch.
    [[nodiscard]] int ReadoutBestEpoch() const { return readout_.BestEpoch(); }

    /// @brief Export the trained readout as HCNW + arch sidecar (see
    /// @ref Readout::SaveHcnnModel). Path stem without extension.
    void SaveReadoutHcnnModel(const std::string& path_stem) const;

    /// @brief Load HCNW (+ optional arch sidecar) into the live readout
    /// (see @ref Readout::LoadHcnnModel).
    void LoadReadoutHcnnModel(const std::string& path_stem,
                              ReadoutLoadMode mode = ReadoutLoadMode::Eval);

    /// @brief Architecture summary of the HCNN readout (param counts, layers).
    [[nodiscard]] std::string ReadoutArchSummary() const;

private:
    std::unique_ptr<Reservoir> reservoir_;
    Readout readout_;
    ESNConfig esn_config_;

    size_t n_ = 0; // reservoir neuron count N = 2^dim (one readout-input block)
    size_t num_inputs_ = 1;

    size_t readout_slices_ = 1; // reservoir delay-line slices fed to the readout
    size_t d_aux_ = 0; // auxiliary-input width (0 = no aux block)
    size_t readout_blocks_ = 1; // B = readout_slices_ + (d_aux_ > 0); a power of two
    size_t readout_width_ = 0; // n_ * readout_blocks_
    std::vector<size_t> block_of_; // slot -> block index (see ReadoutBlockOf)

    std::vector<float> states_; // recorded readout inputs, rows of readout_width_
    size_t num_collected_ = 0;

    // Validates the slice/aux/pooling config and derives readout.dim from the
    // reservoir's dim plus log2(B). Throws on an inconsistent config.
    static ReadoutConfig MakeReadoutConfig(const ESNConfig& cfg);

    // slot -> block permutation placing consecutive slots on blocks whose indices
    // differ in two bits (the only pairs one HCNN conv filter can combine).
    static std::vector<size_t> MakeBlockMap(size_t blocks);

    // Gather the delay-line slices (by logical age) and the aux block into
    // readout_input_, in block_of_ order. The one place the readout's view is built.
    void AssembleReadoutInput(const float* u_raw) const;

    const float* ReadoutInput(size_t timestep) const;
    [[nodiscard]] std::vector<float> ReadoutStates(size_t start, size_t count) const;

    mutable std::vector<float> readout_input_; // readout_width_ scratch
};
