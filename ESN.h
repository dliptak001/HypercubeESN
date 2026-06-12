#pragma once

#include <memory>
#include <vector>
#include "Reservoir.h"
#include "Readout.h"

/// ESN-side configuration of the closed-loop feedback training scheme
/// (docs/FeedbackTrainingMethodology.md §6.14). Holds F's ReadoutConfig plus
/// the scheme's own knobs and the §6.13 runtime lesion flag. This block does
/// NOT enable the feedback path: `reservoir.num_feedback_channels > 0` is the
/// single enablement switch (§6.13); when that is 0 this block is inert.
struct FeedbackConfig
{
    /// F's architecture/training parameters. The ESN forces `dim` (the shared
    /// subsample geometry), `num_outputs = 1` and `task = Regression`; the
    /// batch-path fields (epochs, batch_size, lr_max/lr_min_frac/
    /// lr_decay_epochs, momentum) are ignored in streaming v1. Defaults per
    /// §6.14: 1 layer, 8 channels (small capacity is the regularizer against
    /// memorizing sparse accept samples), seed 43 (off P's 42 to avoid
    /// twin-init in degenerate same-shape configs).
    ReadoutConfig readout{.num_layers = 1, .conv_channels = 8, .seed = 43};

    float epsilon = 0.05f; ///< Probe perturbation, pre-clamp space (§6.11, §6.14).
    float margin = 0.0f; ///< Accept margin; 0 = any strict improvement accepts (§6.6).
    float lr = 2e-4f; ///< F's constant Adam learning rate (§6.14).
    size_t pretrain_steps = 10000; ///< P pre-train budget = its cosine horizon (§6.9).
    float p_lr = 5e-4f; ///< P's constant alternation lr; the pre-train cosine anneals into it (§6.9).
    bool force_zero = false; ///< Runtime lesion flag: inject 0 at the clamp seam (§6.13).
};

struct ESNConfig
{
    ReservoirConfig reservoir;
    ReadoutConfig readout;
    /// Fraction of the N reservoir vertices fed to the readout, in (0.0, 1.0].
    /// The readout sees a stride-selected sub-hypercube of ceil(N / stride)
    /// vertices, where stride = N / round(N * output_fraction).
    ///
    /// Only values whose resulting stride is a power of two are accepted; the
    /// ESN ctor throws std::invalid_argument otherwise. The mapping is lossy:
    /// values between the exact points round down to the nearest power-of-2
    /// stride (e.g. 0.4 -> stride 2 -> effectively 0.5), and values that would
    /// yield a non-power-of-2 stride (e.g. 0.3 -> stride 3) are rejected. The
    /// exactly-honored values are {1.0, 0.5, 0.25, 0.125, 0.0625, ...}.
    float output_fraction = 1.0f;

    /// Closed-loop feedback training scheme (§6.14). Inert unless
    /// `reservoir.num_feedback_channels > 0`. Declared last so existing
    /// designated initializers of the fields above stay valid.
    FeedbackConfig feedback;
};


/// @brief Echo-state network implementing the full pipeline:
///        Reservoir -> [Output Selection] -> Readout.
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

    /// @brief One timestep on the live reservoir — the closed-loop step
    /// driver (docs/FeedbackTrainingMethodology.md §2.1, §6.15). With
    /// feedback configured (@ref HasFeedback): evaluate F frozen on the
    /// current subsampled live state, clamp f = tanh(F(x)) (§6.11),
    /// InjectFeedback, inject this step's inputs (NumInputs() floats), Step.
    /// Without feedback this is exactly inject-inputs + Step (open loop).
    /// Under cfg.feedback.force_zero, 0 is injected in place of tanh(F(x)) —
    /// F is still evaluated, so the lesion arm stays compute-matched to the
    /// live arm (§6.13). No readout training occurs here.
    void StepLive(const float* inputs);

    /// @brief Drive the reservoir for @p num_steps without recording states
    /// (washes out the initial transient). @p inputs is row-major,
    /// num_steps * NumInputs() floats, row-major (NumInputs() values per
    /// timestep, one per channel). Steps via @ref StepLive, so a
    /// feedback-configured ESN warms up closed-loop onto the joint
    /// (reservoir, F) attractor (§6.15).
    void Warmup(const float* inputs, size_t num_steps);

    /// @brief Drive the reservoir for @p num_steps and append the subsampled
    /// state at each step to the collected-states buffer (for batch Train /
    /// R2 / NRMSE / Accuracy). @p inputs has the same layout as @ref Warmup.
    /// Steps via @ref StepLive — with feedback configured the collected
    /// states are closed-loop states under the current (frozen) F.
    void Run(const float* inputs, size_t num_steps);

    /// @brief Discard the collected-states buffer and free its memory. The
    /// trained readout and reservoir weights are left intact.
    void ClearStates();

    /// @brief Zero the reservoir's live state only; collected states and the
    /// trained readout are preserved.
    void ResetReservoirOnly();

    // ---------------------------------------------------------------
    //  Training
    // ---------------------------------------------------------------

    /// @brief Batch-train the readout on collected timesteps [0, train_size).
    /// Requires train_size <= NumCollected(). @p targets layout matches @ref R2.
    void Train(const float* targets, size_t train_size);

    /// @brief Prepare for online (streaming) training: warm up the reservoir on
    /// @p warmup_inputs (same layout as @ref Warmup), then build the readout's
    /// CNN. Call before any TrainLive* method.
    void InitOnline(const float* warmup_inputs, size_t warmup_count);

    /// @brief Single-step online classification training on the live reservoir
    /// state against @p target_class.
    void TrainLiveStep(float target_class, float lr, float weight_decay);

    /// @brief Copy the current subsampled live reservoir state (NumOutputVerts()
    /// floats) into @p out, for external mini-batch accumulation.
    void CopyLiveState(float* out) const;

    /// No-weight_decay overload: inherits `cfg.readout.weight_decay`.
    void TrainLiveBatch(const float* states, const int* targets, size_t count, float lr);

    /// @brief Mini-batch online classification training on pre-accumulated
    /// states (each NumOutputVerts() floats) with integer class @p targets.
    void TrainLiveBatch(const float* states, const int* targets, size_t count, float lr, float weight_decay);

    /// @brief Single-step online regression training on the live reservoir
    /// state against @p target (NumOutputs() floats).
    void TrainLiveStepRegression(const float* target, float lr, float weight_decay);

    /// @brief Mini-batch online regression training on pre-accumulated states
    /// (each NumOutputVerts() floats) with @p targets (count * NumOutputs()).
    void TrainLiveBatchRegression(const float* states, const float* targets,
                                  size_t count, float lr, float weight_decay);

    // ---------------------------------------------------------------
    //  Feedback training orchestration
    //  (docs/FeedbackTrainingMethodology.md §4, §6.9, §6.16, §6.17)
    // ---------------------------------------------------------------

    /// Per-cycle report from @ref TrainFeedbackCycle. §6.9 wants the probe
    /// acceptance rate logged from day one; this is the value stream the
    /// caller accumulates it from (the full FeedbackTelemetry ring buffer is
    /// a separate §8 row).
    struct FeedbackCycleInfo
    {
        bool pretrain = false; ///< This example fell in the P pre-train phase (no probes).
        float e0 = 0.0f; ///< Baseline probe loss (NaN during pretrain).
        float e_plus = 0.0f; ///< +ε probe loss (NaN during pretrain).
        float e_minus = 0.0f; ///< −ε probe loss (NaN during pretrain).
        float sf = 0.0f; ///< F's raw (pre-clamp) operating point Sf (NaN during pretrain).
        bool accepted = false; ///< A probe direction won and F took one training step.
        float sign = 0.0f; ///< +1/−1 (winning direction) when accepted, else 0.
        float p_lr = 0.0f; ///< P's lr this example (§6.9 cosine during pretrain, then constant).
    };

    /// @brief Consume one training example (u, y) of the stream — the §4
    /// Pass-2-then-Pass-1 cycle (every example is probed, then committed,
    /// §6.5). Regression-P overload; @p target is NumOutputs() floats.
    ///
    /// Phase is tracked internally (§6.9): the first
    /// cfg.feedback.pretrain_steps examples train P alone (no probes, no F
    /// updates, F frozen but its clamped output still drives the loop) with
    /// P's lr cosine-annealed from cfg.readout.lr_max down to
    /// cfg.feedback.p_lr. Thereafter each call runs the full cycle: snapshot,
    /// three probes at Sf and Sf ± ε (§6.6), accept iff
    /// min(E+, E−) < E0 − margin and E+ ≠ E− (every exact equality rejects),
    /// on accept one F regression step toward f* = Sf ± ε on the
    /// decision-time state (§6.4), then the real commit via @ref StepLive
    /// (injecting F's live post-update output, never f* — §6.2) and one
    /// per-step P training step at constant p_lr.
    ///
    /// Cost: 4 reservoir steps + 4 P-forwards per alternation cycle (~4×
    /// plain streaming). Requires feedback configured and @ref InitOnline
    /// already run (throws std::logic_error otherwise); throws
    /// std::invalid_argument if P's task is not Regression.
    FeedbackCycleInfo TrainFeedbackCycle(const float* input, const float* target);

    /// Classification-P overload of @ref TrainFeedbackCycle. The probe loss
    /// is the softmax cross-entropy of @p target_class from P's logits,
    /// log-sum-exp stabilized (§6.16 — accuracy is a step function at ε
    /// scale and would never fire an accept). F's own training step is
    /// regression regardless of P's task (§6.16).
    FeedbackCycleInfo TrainFeedbackCycle(const float* input, int target_class);

    /// @brief Score a held-out validation stream under the §6.17 protocol:
    /// TakeSnapshot → zero-reset → closed-loop washout (first W steps,
    /// unscored; W = the warmup count passed to @ref InitOnline) →
    /// closed-loop scoring with F and P frozen (no probes, no training,
    /// force_zero respected) → RestoreSnapshot. Training resumes gapless;
    /// consecutive scores differ only because F/P changed (zero-reset entry).
    ///
    /// Call only at a cycle boundary (between TrainFeedbackCycle calls).
    /// @p inputs is count * NumInputs() floats; @p targets spans all count
    /// steps (count * NumOutputs() floats for regression, count class labels
    /// for classification) — rows before W are skipped. Returns NRMSE
    /// (regression, averaged across outputs) or mean cross-entropy
    /// (classification). Cadence is caller policy (§6.9 default: every 1000
    /// cycles, watching for plateau).
    /// @throws std::logic_error if feedback is not configured or InitOnline
    ///         has not run; std::invalid_argument if count <= W.
    double ValidateClosedLoop(const float* inputs, const float* targets, size_t count);

    /// Examples consumed by @ref TrainFeedbackCycle (both phases); the phase
    /// switch happens at cfg.feedback.pretrain_steps.
    [[nodiscard]] size_t NumFeedbackExamples() const { return fb_examples_; }

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

    /// Run the readout on a caller-supplied state buffer (already
    /// subsampled to NumOutputVerts()).  Lets the caller modify the
    /// readout input -- e.g. brand a side channel onto the first few
    /// slots -- before prediction, without touching live reservoir state.
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

    /// @brief Extract stride-selected vertices from collected states.
    [[nodiscard]] std::vector<float> SelectedStates() const;

    // ---------------------------------------------------------------
    //  Accessors
    // ---------------------------------------------------------------
    [[nodiscard]] size_t NumCollected() const { return num_collected_; }
    [[nodiscard]] size_t NumOutputVerts() const { return num_output_verts_; }
    [[nodiscard]] size_t NumInputs() const { return num_inputs_; }

    /// Hypercube dimension of the underlying reservoir (cfg.reservoir.dim).
    [[nodiscard]] size_t Dim() const { return reservoir_->Dim(); }
    /// Reservoir neuron count N = 2^Dim().
    [[nodiscard]] size_t Size() const { return n_; }

    /// True when the closed-loop feedback path is active (the feedback
    /// readout F exists; equivalent to reservoir.num_feedback_channels > 0).
    [[nodiscard]] bool HasFeedback() const { return feedback_readout_ != nullptr; }

    /// §6.13 runtime lesion flag, settable between training and evaluation
    /// (e.g. train with the loop live, then lesion it to measure how much of
    /// the trained system's performance the closed loop carries). See
    /// @ref StepLive for the injection semantics.
    void SetForceZeroFeedback(bool on) { esn_config_.feedback.force_zero = on; }
    [[nodiscard]] bool GetForceZeroFeedback() const { return esn_config_.feedback.force_zero; }

    // --- Config & persistence ---

    [[nodiscard]] ESNConfig GetConfig() const;

    struct ReadoutState
    {
        std::vector<double> weights;
        bool is_trained = false;
    };

    [[nodiscard]] ReadoutState GetReadoutState() const;
    void SetReadoutState(const ReadoutState& state);

    /// Two-readout checkpointing (docs/FeedbackTrainingMethodology.md §8):
    /// F's counterpart of GetReadoutState/SetReadoutState — same weights-blob
    /// + is_trained shape, so caller-side serialization stays uniform. F is
    /// persist-worthy from construction (§6.15: the eagerly built random F
    /// IS the live policy from cycle 0). Adam moments are not serialized for
    /// either readout, so a resumed run restarts optimizer state — a stated,
    /// accepted v1 limitation. Both throw std::logic_error when feedback is
    /// not configured (@ref HasFeedback is false).
    [[nodiscard]] ReadoutState GetFeedbackState() const;
    void SetFeedbackState(const ReadoutState& state);

private:
    std::unique_ptr<Reservoir> reservoir_;
    Readout readout_;
    /// F — the feedback readout (docs/FeedbackTrainingMethodology.md §3).
    /// Null unless reservoir.num_feedback_channels > 0; when present, its CNN
    /// is built eagerly at ESN construction (§6.15) so it exists before the
    /// first closed-loop Step it feeds.
    std::unique_ptr<Readout> feedback_readout_;
    ESNConfig esn_config_;

    size_t n_ = 0; // reservoir neuron count N = 2^dim
    size_t num_inputs_ = 1;
    size_t output_stride_ = 1;
    size_t num_output_verts_ = 0;

    std::vector<float> states_;
    size_t num_collected_ = 0;

    /**** feedback training orchestration ****/
    size_t fb_examples_ = 0; // examples consumed by TrainFeedbackCycle (phase counter, §6.9)
    size_t warmup_count_ = 0; // W given to InitOnline — reused as the §6.17 validation washout
    std::vector<float> fb_decision_state_; // subsample(Sx): F's training input (§6.4); survives probe clobbering of scratch_subsampled_
    std::vector<float> fb_pred_; // P-forward output buffer for probes/validation

    /// The §6.11 clamp seam: inject tanh(raw) on channel 0, or 0 under the
    /// §6.13 force_zero lesion flag. Every feedback injection — StepLive
    /// commits, probes, validation — goes through here.
    void InjectFeedbackClamped(float raw);

    /// One probe (§4 steps b–e): inject feedback @p raw_feedback (pre-clamp)
    /// and @p input, Step, P-forward, return the §6.16 probe loss. The caller
    /// owns snapshot/restore around this.
    double ProbeLoss(const float* input, float raw_feedback,
                     const float* target, int target_class);

    FeedbackCycleInfo TrainFeedbackCycleImpl(const float* input,
                                             const float* target, int target_class);

    void RequireFeedbackTraining(const char* method) const;

    struct ReadoutGeometry
    {
        size_t output_stride;
        size_t num_output_verts;
        size_t dim;
    };

    static ReadoutGeometry ComputeReadoutGeometry(size_t dim, float output_fraction);
    static ReadoutConfig MakeReadoutConfig(const ESNConfig& cfg, const ReadoutGeometry& geo);
    static ReadoutConfig MakeFeedbackReadoutConfig(const ESNConfig& cfg, const ReadoutGeometry& geo);

    // Delegating-target ctor: receives the geometry computed once by the public
    // ctor, so ComputeReadoutGeometry is not run again for the member init.
    ESN(const ESNConfig& cfg, const ReadoutGeometry& geo);

    const float* ReadoutInput(size_t timestep) const;
    [[nodiscard]] std::vector<float> ReadoutStates(size_t start, size_t count) const;

    mutable std::vector<float> scratch_subsampled_;
};
