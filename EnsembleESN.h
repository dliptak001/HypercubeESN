#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "ESN.h"

/// Consensus statistic used to combine member outputs (design §6).
enum class Combine
{
    Mean,   ///< per-channel mean. Exact conservation Sum_i Delta_i = 0. Default.
    Median  ///< per-channel median. Robust to one straying member; needs M >= 3.
};

/// Configuration for an EnsembleESN — M ESN members coupled through the
/// consensus feedback driver path (design §1, §7). Members share ONE base
/// config and differ only by their derived reservoir seed (§5). The schedule
/// parameters (the kappa ramp + competence gate, the washout) are owned here:
/// EnsembleESN is a *policy* object, not a bare lockstep stepper (§4.2).
struct EnsembleConfig
{
    /// Shared base config for every member. Two fields are overridden per
    /// member by the ctor: `reservoir.seed` (derived from `ensemble_seed`, §5)
    /// and `reservoir.verbose` (forced false so M members do not each print a
    /// construction banner). All other fields are used verbatim and identically
    /// across members — same dim, spectral_radius, leak_rate, activation,
    /// readout architecture, etc. (§5). The "One D, three roles" identity is
    /// enforced: `readout.num_outputs == reservoir.num_feedback_channels == D`,
    /// and D must be > 0 (feedback is the whole point, §1/§3).
    ESNConfig base;

    /// Master seed; per-member seeds are derived from it (§5). Equal
    /// `ensemble_seed` reproduces the whole ensemble exactly.
    uint64_t ensemble_seed = 73895;

    /// M — member count. Default 3 (smallest M for which a median is
    /// meaningful, §5). M >= 1; `Combine::Median` requires M >= 3.
    size_t num_members = 3;

    /// Consensus statistic (§6). Default mean.
    Combine combine = Combine::Mean;

    // ----- Online readout training (shared, §4.2/G4) -----
    // One lr / weight_decay passed verbatim to every member's online step.
    // Held effectively constant through the kappa ramp (the ramp must stay slow
    // relative to readout adaptation); never annealed toward zero while kappa
    // is still moving. Annealing, if any, is the caller's business after kappa
    // holds (§10 Q3).
    float lr = 0.01f;
    float weight_decay = 0.0f;

    // ----- Lifecycle (§7.1) -----
    /// W — initial washout length. During steps [0, W) the readout update is
    /// suppressed (we do not fit on transient states that still remember
    /// x(0)=0); the reservoir is still driven (input + zero-deviation feedback,
    /// since kappa starts at ~0). Plays the transient-killing role of a single
    /// ESN's warmup_count.
    size_t washout = 100;

    /// Short re-washout imposed at each BeginSequence() boundary: that many
    /// steps with the readout update suppressed while the reset dynamics
    /// re-settle. The kappa schedule and competence already achieved are NOT
    /// rewound on a sequence reset (§7.1). 0 disables the re-washout.
    size_t resequence_washout = 16;

    // ----- Kappa ramp + competence gate (§4.2, class-owned) -----
    /// kappa_0 — the starting intensity (mechanism live but barely biting).
    float kappa_start = 0.0f;
    /// kappa* — the target intensity held after the ramp completes.
    float kappa_target = 0.5f;
    /// Per-step linear increment of kappa once the gate opens. <= 0 snaps kappa
    /// straight to kappa_target the moment the gate fires. The ramp should be
    /// slow relative to online readout adaptation (§4.2).
    float kappa_ramp_rate = 0.0f;
    /// Competence gate: the ramp opens once the smoothed consensus error falls
    /// below this threshold (§4.2/G3). NOTE: with the default 0.0 the gate
    /// never fires (a smoothed |error| is never < 0), so kappa is held at
    /// `kappa_start` for the whole run — that is exactly the kappa=0 / fixed-low
    /// measurement baseline. A coupled run must set a positive threshold.
    float gate_threshold = 0.0f;
    /// EMA factor for the running consensus-error estimate the gate reads
    /// (consensus_err_ <- (1-a)*consensus_err_ + a*step_error). In (0, 1].
    float gate_err_ema_alpha = 0.05f;
};

/// @brief Consensus feedback coupling of M ESN members (design doc
/// `docs/ensemble_esn_feedback.md`).
///
/// Every online step, the ensemble reads each member's output y_i at its
/// current state, forms the consensus c (mean/median), and injects each
/// member's scaled deviation phi_i = kappa * (y_i - c) on its D feedback
/// channels before stepping — single-step closed-loop causality, no delay line
/// (§3). The coupling is live for the entire run, training and inference alike;
/// only the intensity kappa varies, on a competence-gated ramp this class owns
/// (§4.2).
///
/// Feedback-only, online-only (§1): there is no batch path and no
/// feedback-less / averaging-only mode. The kappa=0 baseline is just the
/// degenerate left edge of the one ramp, not a separate configuration.
///
/// @note Not thread-safe at the instance level (each member ESN owns mutable
///       scratch). One EnsembleESN per thread.
class EnsembleESN
{
public:
    explicit EnsembleESN(const EnsembleConfig& cfg);

    // ---------------------------------------------------------------
    //  Driving
    // ---------------------------------------------------------------

    /// @brief One lockstep online step across all members (§3, §7.3).
    /// Reads every member's output, forms the consensus, (when training)
    /// updates each readout toward @p target, then injects each member's scaled
    /// deviation and steps it. The class advances kappa internally from the
    /// consensus error (§4.2) — the caller never computes kappa.
    /// @param input    NumInputs() floats — the task input u(t) for this step,
    ///                 injected on every member's input channels.
    /// @param target   NumOutputs() floats — the regression target. Pass
    ///                 nullptr for inference: no readout update is taken and
    ///                 kappa holds. During the washout (the initial W steps and
    ///                 any re-washout) the readout update is suppressed even if
    ///                 @p target is non-null.
    /// @param c_out    NumOutputs() floats — receives the consensus c(t), the
    ///                 ensemble's output for this step.
    void Step(const float* input, const float* target, float* c_out);

    /// @brief Begin a fresh, independent sequence. Resets every member's
    /// reservoir state together (trained readout weights preserved) and
    /// re-imposes the short `resequence_washout` (§7.1). The kappa schedule,
    /// the competence already achieved, and the step counter are NOT rewound.
    void BeginSequence();

    // ---------------------------------------------------------------
    //  Diagnostic surface (read-only, §7.4)
    // ---------------------------------------------------------------

    /// Member i's last output y_i (NumOutputs() floats) — the value used to
    /// form the most recent consensus, not a fresh re-evaluation.
    void MemberOutput(size_t i, float* out) const;

    /// All members' last outputs as an M x D row-major buffer
    /// (NumMembers()*NumOutputs() floats), filled in one call.
    void AllMemberOutputs(float* out_MxD) const;

    /// Current feedback intensity kappa — the operating point of the ramp schedule.
    [[nodiscard]] float Kappa() const { return kappa_; }

    /// Has the competence-gated ramp triggered yet? (§4.2)
    [[nodiscard]] bool GateOpen() const { return gate_open_; }

    /// Monotone step counter t_ (aligns traces to the §7.1 schedule).
    [[nodiscard]] size_t CurrentStep() const { return t_; }

    // ---------------------------------------------------------------
    //  Accessors
    // ---------------------------------------------------------------

    [[nodiscard]] size_t NumMembers() const { return M_; }
    [[nodiscard]] size_t NumOutputs() const { return D_; }
    [[nodiscard]] size_t NumInputs() const { return num_inputs_; }

private:
    size_t M_ = 0;            // member count
    size_t D_ = 0;            // output dim = num_feedback_channels (One D, three roles)
    size_t num_inputs_ = 0;   // task input width
    size_t t_ = 0;            // monotone step counter
    size_t washout_remaining_ = 0; // steps left with the readout update suppressed

    Combine combine_ = Combine::Mean;
    float lr_ = 0.0f;
    float wd_ = 0.0f;

    // kappa ramp + competence gate (§4.2)
    float kappa_ = 0.0f;
    float kappa_target_ = 0.0f;
    float kappa_ramp_rate_ = 0.0f;
    float gate_threshold_ = 0.0f;
    float gate_err_ema_alpha_ = 0.0f;
    float consensus_err_ = 0.0f;   // EMA of |consensus - target|
    bool gate_open_ = false;
    bool err_init_ = false;        // has consensus_err_ been seeded yet?
    size_t resequence_washout_ = 0;

    std::vector<std::unique_ptr<ESN>> esn_;

    // pre-allocated per-step scratch (no heap traffic per tick, decision #5)
    std::vector<float> y_flat_;        // M*D — last member outputs (also the §7.4 source)
    std::vector<float> phi_;           // D   — current member's coupling drive
    mutable std::vector<float> median_scratch_; // M — per-channel gather for the median

    // class-owned competence-gated ramp (§4.2): fold this step's consensus
    // error into consensus_err_, open the gate when it crosses the threshold,
    // and step kappa toward kappa_target. No-op at inference (target==nullptr)
    // and during the washout.
    void AdvanceKappa(const float* c_out, const float* target);

    // write the consensus of y_flat_ (M x D) into c_out (D), per combine_.
    void Consensus(float* c_out) const;
};
