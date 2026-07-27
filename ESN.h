#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Readout.h"
#include "Reservoir.h"

/// @brief Construction parameters for @ref ESN: reservoir + readout + the seam
/// between them.
///
/// @c reservoir and @c readout are documented on @ref ReservoirConfig and
/// @ref ReadoutConfig. The seam field lives here because neither half owns it:
/// it decides how much of the reservoir delay line is packed into the readout
/// input. With B = readout_slices (must be a power of two), the HCNN start
/// dimension is @c reservoir.dim + log2(B); @ref ESN sets @c readout.dim
/// accordingly at construction (do not set @c readout.dim yourself).
struct ESNConfig
{
    ReservoirConfig reservoir;
    ReadoutConfig readout;

    /// How many delay-line ages the readout sees, newest first. Must be ≥ 1, a
    /// **power of two**, and ≤ @c reservoir.history_depth. B = 1 → only the
    /// current state; B > 1 → multi-block input on a (dim + log2 B)-cube.
    /// Independent of history_depth: widening the readout view does not change
    /// reservoir dynamics.
    size_t readout_slices = 1;
};

/// @brief Echo-state pipeline: fixed @ref Reservoir + trainable @ref Readout
/// (HypercubeCNN).
///
/// ## Idea
/// Only the readout is trained. The reservoir is a frozen nonlinear dynamical
/// system that lifts the input stream into a high-dimensional state; the readout
/// maps (one or more) published slices of that state to the task output.
///
/// ```
///   inputs [+ optional ext-fb]
///        │
///        ▼
///   Reservoir (fixed)
///        │
///        ▼
///   SliceAt(0 .. B-1)  ── pack B blocks of N ──▶  HCNN readout (trained) ──▶ y
/// ```
///
/// Only the readout emits @c y. The reservoir never writes the task output
/// directly; it only publishes state the readout reads. (Closed-loop
/// @c external_feedback is an *input* back into the reservoir, supplied by the
/// caller — e.g. a previous prediction — not a second path to @c y.)
///
/// ## Lifecycle (batch)
/// ```
///   ESN esn(cfg);
///   esn.ReservoirWarmup(u, n_warm);   // wash out cold start; nothing recorded
///   esn.ReservoirRun(u, n_collect);   // drive + append readout inputs
///   esn.Train(targets, n_train);      // fit readout on recorded prefix
///   esn.R2 / NRMSE / Accuracy(...);   // score recorded window
/// ```
///
/// ## Lifecycle (streaming)
/// ```
///   esn.ReservoirWarmup(...);
///   loop: ReservoirStep → Predict / TrainStep
/// ```
///
/// ## Vocabulary
/// - **Timestep** — one @ref ReservoirStep.
/// - **N** — reservoir neurons = 2^dim (@ref ReservoirNeuronCount).
/// - **Reservoir state** — newest slice only (@ref Outputs / @ref CopyReservoirState).
/// - **Readout input** — B packed blocks of N (@ref ReadoutInputWidth); what the
///   HCNN actually sees. Equals reservoir state only when B = 1.
/// - **Open loop** — drive with task input only. **Closed loop** — also stage
///   @c external_feedback on the reservoir (caller-owned; see @ref ReservoirStep).
///
/// @note **Not thread-safe.** Const predict paths use a shared scratch buffer.
///       One ESN instance per thread.
class ESN
{
public:
    /// Build reservoir from @c cfg.reservoir and readout from the derived config
    /// (@ref MakeReadoutConfig). Reservoir and HCNN weights are ready immediately.
    explicit ESN(const ESNConfig& cfg);

    // --- Reservoir driving -------------------------------------------------

    /// @brief One reservoir step: stage task input, optional external feedback,
    /// then @ref Reservoir::Step. No learning.
    ///
    /// @param inputs            NumInputs() floats (this timestep).
    /// @param external_feedback nullptr to skip; else exactly
    ///                          NumExternalFeedbackChannels() floats (caller-owned
    ///                          closed-loop drive, e.g. previous prediction).
    /// @throws std::invalid_argument if @p external_feedback is non-null when D = 0.
    void ReservoirStep(const float* inputs, const float* external_feedback = nullptr);

    /// @brief Drive @p num_steps without recording (wash out the zero initial
    /// state). @p inputs is row-major: num_steps × NumInputs(). No external
    /// feedback — use @ref ReservoirStep if you need it. Standard prelude for
    /// batch @ref ReservoirRun and streaming @ref TrainStep.
    void ReservoirWarmup(const float* inputs, size_t num_steps);

    /// @brief Drive @p num_steps and **append** each assembled readout input to
    /// the internal buffer (for @ref Train / @ref R2 / @ref NRMSE / @ref Accuracy).
    /// Same input layout as @ref ReservoirWarmup. No external feedback in this
    /// path. Pass @p clear_recorded true to discard prior rows first. Live
    /// reservoir state and trained readout weights are left as-is except for the
    /// drive itself.
    void ReservoirRun(const float* inputs, size_t num_steps, bool clear_recorded = false);

    /// @brief Zero reservoir dynamics (state + history). Recorded rows and readout
    /// weights are preserved.
    void ReservoirClear();

    // --- Training ----------------------------------------------------------

    /// @brief Batch-fit the readout on recorded timesteps [0, @p train_size).
    /// @p targets layout matches @ref R2 (indexed from timestep 0).
    /// @throws std::out_of_range if @p train_size > NumCollectedStates().
    void Train(const float* targets, size_t train_size);

    /// @brief One online gradient step on the **current** readout input (after
    /// staging drives and @ref ReservoirStep as you choose). Task is fixed at
    /// construction: regression → NumOutputs() floats; classification → one
    /// class-index float.
    void TrainStep(const float* target, float lr, float weight_decay = 0.0f);

    /// @brief One online gradient step on a mini-batch of readout inputs you
    /// supply (each ReadoutInputWidth() floats, e.g. from @ref CopyReadoutInput).
    /// Regression: count × NumOutputs() targets; classification: count class indices.
    void TrainStepBatch(const float* readout_inputs, const float* targets, size_t count,
                        float lr, float weight_decay = 0.0f);

    /// @brief Copy the newest reservoir slice (N floats) into @p out.
    /// Not the full multi-slice readout input unless B = 1.
    void CopyReservoirState(float* out) const;

    /// @brief Assemble and copy the current readout input (ReadoutInputWidth()
    /// floats) into @p out — B blocks of N in @ref ReadoutBlockOf order.
    void CopyReadoutInput(float* out) const;

    // --- Prediction & evaluation -------------------------------------------

    /// @brief Predict from the live readout input (assemble then HCNN forward).
    /// Returns NumOutputs() floats.
    [[nodiscard]] std::vector<float> Predict() const;

    /// @brief Predict into caller @p out (NumOutputs() floats). Prefer for hot loops.
    void Predict(float* out) const;

    /// @brief Predict from a recorded row at @p timestep (@ref ReservoirRun).
    /// @throws std::out_of_range if @p timestep >= NumCollectedStates().
    [[nodiscard]] std::vector<float> PredictFromRecorded(size_t timestep) const;

    /// @brief Predict from a caller-supplied readout input (ReadoutInputWidth()
    /// floats). Does not touch the live reservoir.
    [[nodiscard]] std::vector<float> PredictFromState(const float* readout_input) const;

    /// @brief PredictFromState into @p out (NumOutputs() floats).
    void PredictFromState(const float* readout_input, float* out) const;

    /// @brief R² on recorded timesteps [@p start, @p start+@p count).
    /// @p targets must cover [0, start+count) (regression: row-major
    /// (start+count)×NumOutputs(); classification: start+count floats). Indexed
    /// from targets[start * NumOutputs()] — pass the full array, not a slice.
    /// @throws std::out_of_range if the window exceeds NumCollectedStates().
    [[nodiscard]] double R2(const float* targets, size_t start, size_t count) const;

    /// @brief Mean over outputs of NRMSE = RMSE / std(target) on the same window
    /// contract as @ref R2. Lower is better; 0 = perfect. Degenerate target
    /// variance → +inf on that output.
    [[nodiscard]] double NRMSE(const float* targets, size_t start, size_t count) const;

    /// @brief Classification accuracy on recorded [@p start, @p start+@p count).
    /// @p labels cover [0, start+count) as floats holding class indices.
    [[nodiscard]] double Accuracy(const float* labels, size_t start, size_t count) const;

    // --- Recorded buffer ---------------------------------------------------

    /// @brief Copy of all recorded readout inputs: NumCollectedStates() rows ×
    /// ReadoutInputWidth(). Training/scoring use the internal buffer directly.
    [[nodiscard]] std::vector<float> CollectedStates() const;

    // --- Accessors ---------------------------------------------------------

    /// Rows currently stored by @ref ReservoirRun (index range for recorded APIs).
    [[nodiscard]] size_t NumCollectedStates() const { return num_collected_; }

    /// Input channels per timestep (length of each drive row).
    [[nodiscard]] size_t NumInputs() const { return num_inputs_; }

    /// Prediction width: regression targets, or class count.
    [[nodiscard]] size_t NumOutputs() const { return readout_.NumOutputs(); }

    /// Reservoir hypercube dimension (cfg.reservoir.dim).
    [[nodiscard]] size_t ReservoirHypercubeDimension() const { return reservoir_->Dim(); }

    /// N = 2^ReservoirHypercubeDimension() — length of one block / reservoir state.
    [[nodiscard]] size_t ReservoirNeuronCount() const { return n_; }

    /// Full readout input length: B × N.
    [[nodiscard]] size_t ReadoutInputWidth() const { return readout_width_; }

    /// B = readout_slices (power of two).
    [[nodiscard]] size_t ReadoutBlockCount() const { return readout_blocks_; }

    /// @brief Physical block index for logical slot @p slot (age 0 = newest).
    ///
    /// Not the identity when B > 2: HCNN conv has no self term and only 1-bit
    /// neighbors, so consecutive ages are mapped onto block indices that differ
    /// in **two** bits (so one filter can see “velocity” {age0, age1}).
    /// @throws std::out_of_range if @p slot >= B.
    [[nodiscard]] size_t ReadoutBlockOf(size_t slot) const;

    /// D = cfg.reservoir.num_external_feedback_channels (0 = no ext-fb port).
    [[nodiscard]] size_t NumExternalFeedbackChannels() const
    {
        return esn_config_.reservoir.num_external_feedback_channels;
    }

    // --- Config & persistence ----------------------------------------------

    /// Config this instance was built from (@c readout.dim filled with the
    /// derived HCNN start dimension).
    [[nodiscard]] ESNConfig GetConfig() const;

    /// Portable readout weight blob (reservoir is config + seed, not stored here).
    struct ReadoutState
    {
        std::vector<double> weights;
        /// True if weights are worth loading; matches @ref Readout::IsTrained
        /// (network exists after construction — not “has seen data”).
        bool is_trained = false;
    };

    [[nodiscard]] ReadoutState GetReadoutState() const;

    /// Restore weights from @ref GetReadoutState. No-op if @c !state.is_trained.
    /// @p mode Eval = parameters only; ResumeTrain also resets optimizer moments.
    void SetReadoutState(const ReadoutState& state,
                         ReadoutLoadMode mode = ReadoutLoadMode::Eval);

    /// 1-based best epoch after last batch @ref Train with restore_best_epoch, else 0.
    [[nodiscard]] int ReadoutBestEpoch() const { return readout_.BestEpoch(); }

    /// Export HCNW + arch sidecar (@ref Readout::SaveHcnnModel). Stem without extension.
    void SaveReadoutHcnnModel(const std::string& path_stem) const;

    /// Load HCNW (+ optional arch validation) into the live readout.
    void LoadReadoutHcnnModel(const std::string& path_stem,
                              ReadoutLoadMode mode = ReadoutLoadMode::Eval);

    /// Human-readable fixed-reservoir weight count + HCNN stack summary.
    [[nodiscard]] std::string ReadoutArchSummary() const;

private:
    std::unique_ptr<Reservoir> reservoir_;
    Readout readout_;
    ESNConfig esn_config_;

    size_t n_ = 0;            ///< N = 2^dim (one block)
    size_t num_inputs_ = 1;

    size_t readout_slices_ = 1;  ///< B logical ages
    size_t readout_blocks_ = 1;  ///< = readout_slices_ (power of two)
    size_t readout_width_ = 0;   ///< B * N
    std::vector<size_t> block_of_; ///< logical slot → physical block

    std::vector<float> states_; ///< Recorded readout inputs, row-major
    size_t num_collected_ = 0;

    /// Validate seam; set @c readout.dim = reservoir.dim + log2(B).
    static ReadoutConfig MakeReadoutConfig(const ESNConfig& cfg);

    /// Permutation so consecutive ages land on 2-bit-apart blocks when B > 2.
    static std::vector<size_t> MakeBlockMap(size_t blocks);

    /// Pack SliceAt(0..B-1) into @c readout_input_ in block_of_ order.
    void AssembleReadoutInput() const;

    const float* ReadoutInput(size_t timestep) const;
    [[nodiscard]] std::vector<float> ReadoutStates(size_t start, size_t count) const;

    mutable std::vector<float> readout_input_; ///< Assemble scratch (B*N)
};
