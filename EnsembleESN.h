#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "ESN.h"

/// Consensus statistic used to combine member outputs (design §6).
enum class Combine
{
    Mean, ///< per-channel mean. Exact conservation Sum_i Delta_i = 0. Default.
    Median ///< per-channel median. Robust to one straying member; needs M >= 3.
};

/// Configuration for an EnsembleESN — M ESN members coupled through the
/// consensus feedback driver path (design §1, §7). Members share ONE base
/// config and differ only by their derived reservoir seed (§5). The coupling
/// intensity kappa is not configured here — the caller drives it at runtime via
/// EnsembleESN::SetKappa.
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

    /// Per-step online learning rate seeding EnsembleESN's lr at construction;
    /// caller-managed thereafter via SetLr. Note: `base.readout.lr_max` /
    /// `base.readout.weight_decay` govern only the batch Readout::Train path
    /// and are ignored by the ensemble's online stepping.
    float learning_rate = 0.001f;

    /// L2 weight decay seeding EnsembleESN's wd at construction (0 = off);
    /// caller-managed thereafter via SetWeightDecay.
    float weight_decay = 0.0f;

    /// Auxiliary-input width d_aux (0 = feature OFF). When > 0, each member gets a
    /// fixed random projection W_u_i (d_aux -> N) and @ref EnsembleESN::Step /
    /// @ref EnsembleESN::Predict accept an optional raw aux vector u_raw of this many
    /// floats, blended into that member's readout input as
    /// F_i = k*x_i + (1-k)*(W_u_i . u_raw). The aux vector is a *new* input type,
    /// separate from the reservoir drive, and it feeds only the readout — it never
    /// drives the reservoir dynamics. u_raw is shared across members; the projection
    /// W_u_i is per-member.
    size_t aux_input_dim = 0;

    /// Draw scale for the per-member W_u weights, applied with a 1/sqrt(d_aux) fan-in
    /// normalization (mirroring the reservoir's input_scaling). The caller should
    /// normalize u_raw to ~O(1) so k*x and (1-k)*u sit on comparable scales.
    float aux_scaling = 0.5f;

    void SetDIM(const size_t dim) { base.reservoir.dim = base.readout.dim = dim; };

    void SetSeed(const size_t seed)
    {
        ensemble_seed = seed;
        base.readout.seed = seed;
    };
};

/// @brief Consensus feedback coupling of M ESN members (design doc
/// `docs/ensemble_esn_feedback.md`).
///
/// Every online step, the ensemble reads each member's output y_i at its
/// current state, forms the consensus c (mean/median), and injects each
/// member's scaled deviation phi_i = kappa * (y_i - c) on its D feedback
/// channels before stepping — single-step closed-loop causality, no delay line
/// (§3). The coupling is live for the entire run, training and inference alike.
///
/// The coupling intensity kappa is set by the caller (@ref SetKappa) and held
/// fixed by this class between calls. Ramping or gating it over the run, if
/// wanted, is the caller's policy — not this class's job. kappa = 0 runs the
/// members uncoupled.
///
/// Online-only (§1): there is no batch path.
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
    /// deviation phi_i = kappa*(y_i - c) and steps it.
    /// @param input    NumInputs() floats — the task input u(t) for this step,
    ///                 injected on every member's input channels.
    /// @param target   NumOutputs() floats — the regression target: each member's
    ///                 readout takes one online update toward it. Pass nullptr for
    ///                 inference (no readout update).
    /// @param c_out    NumOutputs() floats — receives the consensus c(t), the
    ///                 ensemble's output for this step.
    /// @param u_raw    optional auxiliary input — cfg.aux_input_dim floats — blended
    ///                 into every member's readout input as F_i = k*x_i +
    ///                 (1-k)*(W_u_i . u_raw) for this step's read and (when training)
    ///                 fit. nullptr leaves the readout on the raw reservoir state
    ///                 (exact pre-feature behavior). @throws std::invalid_argument if
    ///                 non-null while the ensemble was built with aux_input_dim == 0.
    void Step(const float* input, const float* target, float* c_out,
              const float* u_raw = nullptr);

    /// @brief Like @ref Step, but with PER-MEMBER inputs: row i of
    /// @p inputs_MxI drives member i. Identical lockstep semantics otherwise
    /// (consensus read, optional training, deviation coupling).
    ///
    /// This is the coupling's load-bearing mode. Under the shared-input
    /// @ref Step every member rides one common drive, so member outputs stay
    /// nearly identical and phi_i = kappa*(y_i - c) has almost nothing to act
    /// on. With per-member drive — typically each member fed back its OWN
    /// output in a closed loop — the member trajectories genuinely diverge and
    /// the coupling becomes a real consensus-attraction force between M
    /// otherwise-independent trajectories. Note what it still cannot do: phi
    /// sums to zero across members (mean combine), so an error component
    /// SHARED by all members passes through the coupling untouched in either mode.
    /// @param inputs_MxI NumMembers() x NumInputs() floats, row-major —
    ///                   row i is member i's input u_i(t) for this step.
    /// @param target     as @ref Step (one shared target; nullptr = inference).
    /// @param c_out      as @ref Step — receives the consensus c(t).
    /// @param u_raw      as @ref Step — optional shared auxiliary input for this step.
    void StepPerMember(const float* inputs_MxI, const float* target, float* c_out,
                       const float* u_raw = nullptr);

    /// @brief Fresh consensus read at every member's CURRENT state — no readout
    /// update, no reservoir step. The ensemble counterpart of ESN::Predict.
    /// Closed-loop callers read this first, build the next input from it, then
    /// drive @ref Step (whose own read at the unchanged state yields the same
    /// values). Refreshes the member-output diagnostic buffer (§7.4).
    /// @param c_out NumOutputs() floats — receives the consensus.
    /// @param u_raw optional shared auxiliary input (as @ref Step). Pass the SAME
    ///        u_raw here and to the following @ref Step so the pre-step read and the
    ///        step's own read land on the same blended state.
    void Predict(float* c_out, const float* u_raw = nullptr);

    /// @brief Set the feedback coupling intensity kappa applied on subsequent
    /// steps (phi_i = kappa*(y_i - c)). The caller owns the schedule; this class
    /// holds kappa fixed between calls. Takes effect on the next @ref Step.
    void SetKappa(float kappa) { kappa_ = kappa; }

    /// @brief Set the readout-input blend coefficient k applied on subsequent steps
    /// when an auxiliary input is supplied: F_i = k*x_i + (1-k)*(W_u_i . u_raw).
    /// k = 1 (the default) is pure reservoir state — exact pre-feature behavior;
    /// k = 0 drives the readout purely off the projected aux signal. Shared across
    /// members and caller-managed like @ref SetKappa. No effect on steps where
    /// u_raw is nullptr.
    void SetMix(float k) { k_ = k; }

    /// @brief Set the shared online learning rate / L2 weight-decay applied to
    /// every member's readout on each subsequent training @ref Step. Seeded from
    /// the construction-time config; caller-managed like @ref SetKappa thereafter
    /// (e.g. to anneal lr once the coupling has settled). Takes effect on the
    /// next @ref Step. (Not part of @ref State — a reload restores the config
    /// value, not a mid-run override.)
    void SetLr(float lr) { lr_ = lr; }
    void SetWeightDecay(float weight_decay) { wd_ = weight_decay; }

    // ---------------------------------------------------------------
    //  Diagnostic surface (read-only, §7.4)
    // ---------------------------------------------------------------

    /// Member i's last output y_i (NumOutputs() floats) — the value used to
    /// form the most recent consensus, not a fresh re-evaluation.
    void MemberOutput(size_t i, float* out) const;

    /// All members' last outputs as an M x D row-major buffer
    /// (NumMembers()*NumOutputs() floats), filled in one call.
    void AllMemberOutputs(float* out_MxD) const;

    /// Current feedback coupling intensity kappa (set via @ref SetKappa).
    [[nodiscard]] float Kappa() const { return kappa_; }

    /// Current readout-input blend coefficient k (set via @ref SetMix).
    [[nodiscard]] float Mix() const { return k_; }

    /// Current shared online learning rate / L2 (set via @ref SetLr / @ref SetWeightDecay).
    [[nodiscard]] float Lr() const { return lr_; }
    [[nodiscard]] float WeightDecay() const { return wd_; }

    // ---------------------------------------------------------------
    //  Accessors
    // ---------------------------------------------------------------

    [[nodiscard]] size_t NumMembers() const { return M_; }
    [[nodiscard]] size_t NumOutputs() const { return D_; }
    [[nodiscard]] size_t NumInputs() const { return num_inputs_; }

    // ---------------------------------------------------------------
    //  Persistence
    // ---------------------------------------------------------------

    /// The persistable state of a trained ensemble: every member's readout
    /// weights plus the coupling intensity. The EnsembleConfig is NOT held here —
    /// the caller reconstructs an identically configured EnsembleESN (which
    /// re-derives each member's reservoir seed, §5) and then restores this. Like
    /// the single ESN, the reservoirs' live dynamical state is NOT captured: a
    /// restored ensemble has cold reservoirs (drive them through a warmup before
    /// trusting outputs).
    struct State
    {
        std::vector<std::vector<double>> member_weights; ///< M readout-weight blobs, member order
        float kappa = 0.0f; ///< coupling intensity at capture
        float mix = 1.0f; ///< readout-input blend coefficient k at capture (1 = pure state)
    };

    /// Capture the trained state (all member readout weights + kappa).
    [[nodiscard]] State GetState() const;

    /// Restore a previously captured state into this (identically configured)
    /// ensemble. The reservoirs stay cold (their dynamical state is not part of
    /// @ref State) — drive them through a warmup before trusting outputs.
    /// @throws std::invalid_argument if @p s.member_weights.size() != NumMembers().
    void SetState(const State& s);

private:
    size_t M_ = 0; // member count
    size_t D_ = 0; // output dim = num_feedback_channels (One D, three roles)
    size_t num_inputs_ = 0; // task input width
    size_t n_ = 0; // reservoir neuron count N = 2^dim (shared across members)
    size_t d_aux_ = 0; // auxiliary-input width (0 = aux feature off)

    Combine combine_ = Combine::Mean;
    float lr_ = 0.0f;
    float wd_ = 0.0f;

    // feedback coupling intensity (caller-managed via SetKappa)
    float kappa_ = 0.0f;

    // readout-input blend coefficient k (caller-managed via SetMix); 1 = pure state
    float k_ = 1.0f;

    std::vector<std::unique_ptr<ESN>> esn_;

    // per-member fixed aux projection W_u_i, row-major N x d_aux (empty when d_aux_ == 0)
    std::vector<std::vector<float>> Wu_;

    // pre-allocated per-step scratch (no heap traffic per tick, decision #5)
    std::vector<float> y_flat_; // M*D — last member outputs (also the §7.4 source)
    std::vector<float> phi_; // D   — current member's coupling drive
    std::vector<float> f_; // N   — blended readout input F_i for the current member
    mutable std::vector<float> median_scratch_; // M — per-channel gather for the median

    // write the consensus of y_flat_ (M x D) into c_out (D), per combine_.
    void Consensus(float* c_out) const;

    // fill out (N floats) with member i's blended readout input
    // F_i = k*x_i + (1-k)*(W_u_i . u_raw). Requires d_aux_ > 0.
    void BlendedState(size_t i, const float* u_raw, float* out) const;

    // shared core of Step / StepPerMember: member i is driven by
    // inputs + i*input_stride (stride 0 = one shared row for all members). When
    // u_raw != nullptr the readout reads/trains on the blended state (see BlendedState).
    void StepImpl(const float* inputs, size_t input_stride, const float* target,
                  float* c_out, const float* u_raw);
};
