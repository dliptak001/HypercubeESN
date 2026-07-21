#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <span>
#include <vector>

struct ReservoirConfig
{
    size_t dim = 10; // hypercube dimension; reservoir has N = 2^dim neurons (5 <= dim <= 16)
    uint64_t seed = 73895;
    float spectral_radius = 0.99f;
    float leak_rate = 1.0f; // 1.0 = full replacement, <1.0 = leaky integrator
    float input_scaling = 0.5f;
    size_t num_inputs = 1;
    size_t history_depth = 16;
    float history_floor = 1.0f;
    // deepest-history recurrent weight scale K in [0.1, 1.0]; linearly tapers older history slices (1.0 = no taper)
    bool verbose = true;

    // --- External feedback drive port (caller-owned values each step) ---
    // 0 disables the path entirely (no buffer, no weights). Any D in [1, N] is
    // admissible — D need NOT divide N (non-dividing D leaves the N mod D tail
    // vertices at reset-zero: benign zero sources that still receive drive via
    // the neighbor gather).
    size_t num_external_feedback_channels = 0;
    float external_feedback_scaling = 0.5f; // DIM-invariant: weights × scaling/√dim

    // --- Full-state linear feedback (internal drive; construction-only enable) ---
    // When false: zero FSF allocation. When true: B_fsf + staging buffer + V
    // (length N). From standalone fsf_seed (not mixed from `seed`): first N → V as
    // U(-1,1) with no scale baked in; then N·dim → B_fsf × fsf_scaling/√dim.
    // Each Step: φ = V·x, pad[v] = fsf_stage_scaling·φ·V[v] (w ≡ V forever),
    // gather via B_fsf. See docs/full_state_linear_feedback.md.
    bool full_state_feedback = false;
    uint64_t fsf_seed = 1;
    float fsf_scaling = 0.5f;        // B_fsf: U(-1,1) × scaling/√dim (construction)
    float fsf_stage_scaling = 1.0f;  // Step: pad[v] = scale · φ · V[v]

    float bias_scaling = 0.02f; // per-neuron additive bias drawn U(-1,1)*bias_scaling, added to the activation (after the tanh); OFF by default (0 disables)

    // --- Lorentzian activation envelope ---
    // A_lorentz(x) = tanh(x * gain),  gain = 1 + lorentz_gamma * phi,
    // phi = 1/(1 + x^2 * lorentz_inv_sigma2) ∈ (0,1].
    //   gamma = 0           => gain ≡ 1 => plain tanh(x)
    //   gamma > 0           => steeper central slope, tanh tails (sharpening)
    //   gamma < 0, |g| > 1  => central gain crosses 0 => non-monotone "fold"
    // Runtime so the activation shape is a sweep axis (no recompile).
    float lorentz_gamma      = 0.0f;  // 0 reduces A_lorentz to tanh
    float lorentz_inv_sigma2 = 250.0f; // 1/sigma^2
};

/// @brief The **fixed (never-trained) recurrent core of an @ref ESN** — a pool of
/// N = 2^dim neurons wired together on a Boolean-hypercube graph.
///
/// ## Why a hypercube?
/// Each neuron sits on a vertex of a @c dim-dimensional hypercube and connects
/// only to its @c dim *nearest neighbors* — the vertices one bit-flip away
/// (Hamming distance 1). That gives a sparse, regular, well-mixed topology whose
/// wiring is pure arithmetic: neighbor @c i of vertex @c v is @c v XOR (1<<i),
/// computed on the fly by @ref NearestMask, so no adjacency list is ever stored.
/// Because @c dim is a runtime config field, one class serves every size; N and
/// all weight/state buffers are sized once at construction.
///
/// ## What one step computes
/// For each neuron, @ref Step gathers its neighbors' signals through separate
/// weight blocks, squashes the sum, and blends it with the neuron's old value:
/// ```
///   input ──────────────────┐
///   recurrent history ──────┼─▶ Σ (over dim neighbors) ─▶ activation ─▶ leak ─▶ new state
///   external feedback (opt) ┤                              (+ bias)
///   full-state feedback (opt)┘
/// ```
///   - **input** — your per-step drive (see @ref InjectInput).
///   - **recurrent history** — delay line of @c history_depth past slices.
///   - **leak blend** — @c leak_rate mixes the fresh activation with the previous
///     state (1.0 = full replacement; < 1.0 makes a slower "leaky integrator").
///
/// The recurrent weights are rescaled at construction to a target **spectral
/// radius** (@c spectral_radius). @ref GetRealizedSpectralRadius reports the
/// value actually achieved. Drive-port weights sit **outside** that rescale.
///
/// ## Per-step contract
/// ```
///   InjectInput(...)                 // stage task input
///   InjectExternalFeedback(...)      // optional — caller-owned closed loop
///   Step()                           // may also stage internal FSF (φ = V·x), then
///                                    // update, age history, clear all staged drives
/// ```
/// Injected drives are *consumed and cleared* by every @ref Step. Full-state
/// feedback (when enabled) is staged **inside** @ref Step from the current
/// published state — the caller does not inject it. Newest state: @ref Outputs.
///
/// ## Optional extras (all off by default)
///   - **External feedback** (@c num_external_feedback_channels > 0): caller-owned
///     per-step drive, twin of the input path (@ref InjectExternalFeedback).
///     Outside the spectral-radius rescale.
///   - **Full-state linear feedback** (@c full_state_feedback): internal drive
///     each step: φ = V·x, pad[v] = @c fsf_stage_scaling·φ·V[v] (w ≡ V forever),
///     gather through B_fsf. V is U(-1,1) from @c fsf_seed; B_fsf from same seed
///     with @c fsf_scaling. Outside SR. Zero alloc when off.
///   - **Per-neuron bias** (@c bias_scaling > 0): fixed additive term per neuron.
///   - **Lorentzian activation** (@c lorentz_gamma != 0); gamma = 0 is plain tanh.
///
/// ## Lifetime
/// Non-copyable and non-movable; obtain instances via @ref Create.
class Reservoir
{
public:
    /// Inline neighbor mask computation — no stored adjacency.
    /// Mask for neighbor i in [0, dim):  1 << i  →  1, 2, 4, 8, ...
    static constexpr uint32_t NearestMask(size_t i) { return 1u << i; }

    /// @brief Construct a reservoir from a fully resolved config — the only way to
    /// obtain an instance (the type is non-copyable and non-movable).
    ///
    /// Initializes the weights from @c cfg.seed, rescales the recurrent block to
    /// the target spectral radius, and leaves the reservoir reset (zero state).
    /// @throws std::invalid_argument if any config field is out of range (e.g.
    ///         dim outside [5, 16], num_inputs does not divide N, history_depth
    ///         outside [1, 64]).
    static std::unique_ptr<Reservoir> Create(const ReservoirConfig& cfg)
    {
        return std::unique_ptr<Reservoir>(new Reservoir(cfg));
    }

    Reservoir(const Reservoir&) = delete;
    Reservoir& operator=(const Reservoir&) = delete;

    /// @brief Advance the reservoir by one timestep.
    ///
    /// Recomputes every vertex's state from staged input / external feedback /
    /// (if enabled) full-state feedback and the recurrent history, ages the
    /// history ring by one slice (the new state becomes slice 0), and clears the
    /// per-step staged drives. Stage input/external feedback first with
    /// @ref InjectInput / @ref InjectExternalFeedback — they are consumed here.
    /// Full-state feedback is staged inside this call when enabled.
    void Step();

    /// @brief Stage the input for one channel ahead of the next @ref Step.
    ///
    /// Channel @p channel drives the contiguous vertex block
    /// [channel * N/num_inputs, (channel+1) * N/num_inputs). Inputs are cleared by
    /// every @ref Step, so call this each timestep an input is desired.
    /// @throws std::invalid_argument if @p channel >= num_inputs.
    void InjectInput(size_t channel, float input);

    /// @brief Reset dynamical state and history to zero and re-home the slice ring
    /// (undriven live state). Weights, bias, and FSF gain V (if enabled) are left
    /// unchanged — construction-time parameters, not dynamical state. Staged input /
    /// external-feedback / FSF buffers are cleared.
    void Clear();

    /// @brief The current reservoir state — the most-recent history slice; this is
    /// the feature vector the readout consumes.
    /// @return Pointer to N floats, valid until the next @ref Step or @ref Clear.
    [[nodiscard]] const float* Outputs() const { return slice_ptrs_[0]; }

    /// @brief The state slice from @p age timesteps ago — a read-only view into the
    /// delay line. `SliceAt(0)` is the current state (identical to @ref Outputs);
    /// `SliceAt(1)` is the state one step back, and so on.
    ///
    /// Indexes by **logical age, not physical position**. @ref Step rotates the slice
    /// ring, so the history buffer's block order changes every timestep and its raw
    /// layout is meaningless to a consumer. Always read the delay line through this
    /// (or @ref TakeSnapshot); never straight out of the underlying buffer.
    ///
    /// This exposes the temporal memory the reservoir already computes and stores for
    /// its recurrent gather — a consumer that reads only @ref Outputs discards the
    /// other `history_depth - 1` slices.
    ///
    /// @return Pointer to N floats, valid until the next @ref Step or @ref Clear.
    /// @throws std::out_of_range if @p age >= history_depth.
    [[nodiscard]] const float* SliceAt(size_t age) const;

    /// @brief The post-rescale spectral radius measured at construction (the secant
    /// root-find's final estimate), which approximates the configured target
    /// @c spectral_radius.
    [[nodiscard]] float GetRealizedSpectralRadius() const { return realized_spectral_radius_; }

    /// @brief Reconstruct the @ref ReservoirConfig this reservoir was built from.
    ///
    /// Every config field is read back from a stored member (including FSF knobs
    /// even when FSF is off). @c Create(GetConfig()) rebuilds matching weights and
    /// FSF gain V from @c seed / @c fsf_seed and the FSF scales. @c spectral_radius
    /// here is the configured TARGET, not the realized value — use
    /// @ref GetRealizedSpectralRadius for the post-rescale estimate.
    [[nodiscard]] ReservoirConfig GetConfig() const;

    /// @brief Hypercube dimension; the reservoir has N = 2^Dim() neurons.
    [[nodiscard]] size_t Dim() const { return dim_; }

    /// @brief Neuron count N = 2^Dim() (the length of the @ref Outputs feature vector).
    [[nodiscard]] size_t Size() const { return n_; }

    /// @brief Stage one external-feedback channel ahead of the next @ref Step.
    ///
    /// Caller-owned closed-loop drive (twin of @ref InjectInput). Channel @p channel
    /// drives the contiguous vertex block
    /// [channel * floor(N/D), (channel+1) * floor(N/D)) through the external-feedback
    /// weight block. Cleared by every @ref Step. Requires
    /// @c num_external_feedback_channels > 0.
    /// @throws std::invalid_argument if external feedback is not configured or
    ///         @p channel is out of range.
    void InjectExternalFeedback(size_t channel, float value);

    /// @brief Stage all D external-feedback channels at once (@p count must equal D).
    /// @throws std::invalid_argument if @p count != num_external_feedback_channels,
    ///         or if @p values is null when @p count > 0.
    void InjectExternalFeedback(const float* values, size_t count);

    /// @brief True if this reservoir was built with @c full_state_feedback.
    [[nodiscard]] bool FullStateFeedbackEnabled() const { return fsf_enabled_; }

    /// @brief Copyable capture of the reservoir's persistent dynamical state: the
    /// live vertex state plus every history slice in logical age order (slice 0 =
    /// most recent).
    ///
    /// The per-step staged drives (input, external feedback, FSF staging buffer)
    /// are NOT captured — they are consumed and cleared by every @ref Step, so a
    /// snapshot taken between steps has nothing staged. Weights and FSF gain V are
    /// not included (both are construction-time params from config + seeds): a
    /// snapshot is only meaningful for the reservoir it was taken from (or one
    /// rebuilt from an identical config).
    struct Snapshot
    {
        std::vector<float> state; ///< live vertex state, N floats
        std::vector<float> history; ///< N * history_depth floats, slice-major, most recent slice first
    };

    /// @brief Capture the persistent dynamical state (state + history ring).
    ///
    /// Call between steps (staged drives are not captured). History slices are
    /// stored in logical age order regardless of the ring's current rotation, so
    /// the snapshot is canonical: snapshots of identical dynamics compare equal
    /// even when taken at different ring phases.
    [[nodiscard]] Snapshot TakeSnapshot() const;

    /// @brief Bit-exact restore of a state captured by @ref TakeSnapshot.
    ///
    /// Copies the state and history back, re-homes the slice ring to the canonical
    /// rotation, and clears staged input / external feedback / FSF buffers — so the
    /// post-restore trajectory depends on the snapshot, construction-time FSF
    /// params (if enabled), and subsequent injections. Restoring and replaying the
    /// same drives on an identically configured reservoir is bit-exact.
    /// @throws std::invalid_argument if the snapshot's buffer sizes do not match
    ///         this reservoir's N and history_depth.
    void RestoreSnapshot(const Snapshot& snap);

private:
    explicit Reservoir(const ReservoirConfig& cfg);

    /// Deleter for buffers from @ref AllocAligned. Must mirror that allocation
    /// exactly: the raw `::operator new[]`/`delete[]` pair (NOT a `new float[]`
    /// expression) carries no array cookie, and the over-aligned form takes the
    /// `align_val_t` overload — so a plain `delete[]`/`delete` here would mismatch
    /// the allocation and corrupt the heap. Keep the 64-byte alignment in sync.
    struct AlignedFree
    {
        void operator()(float* p) const noexcept { ::operator delete[](p, std::align_val_t{64}); }
    };

    /// Allocate @p count floats on a 64-byte (cache-line) boundary, uninitialized.
    /// Uses the raw aligned `::operator new[]` (no array cookie, no construction);
    /// every buffer so allocated MUST be freed through @ref AlignedFree.
    static float* AllocAligned(size_t count)
    {
        return static_cast<float*>(::operator new[](count * sizeof(float), std::align_val_t{64}));
    }

    uint64_t rng_seed_;

    size_t dim_ = 0; // hypercube dimension (ReservoirConfig::dim)
    size_t n_ = 0; // neuron count N = 2^dim_
    size_t num_input_weights_ = 0; // n_ * dim_ — size of the input-weight block

    std::unique_ptr<float[], AlignedFree> vtx_input_;
    std::unique_ptr<float[], AlignedFree> vtx_state_;
    std::unique_ptr<float[], AlignedFree> vtx_output_history_;
    std::unique_ptr<float[], AlignedFree> vtx_weight_;
    std::unique_ptr<float*[]> slice_ptrs_;
    std::unique_ptr<float[], AlignedFree> vtx_bias_;

    size_t num_inputs_ = 1;
    float spectral_radius_ = 0.99f;
    float leak_rate_ = 1.0f;
    float input_scaling_ = 1.0f;
    float realized_spectral_radius_ = 0.0f; // set by Initialize() after rescale
    bool verbose_ = true;
    size_t history_depth_ = 1;
    float history_floor_ = 1.0f; // cfg.history_floor — deepest-history taper scale K
    size_t num_weights_ = 0;

    /**** external feedback (caller-owned) ****/
    size_t num_ext_feedback_channels_ = 0;
    float ext_feedback_scaling_ = 1.0f;
    size_t num_ext_feedback_weights_ = 0; // n_ * dim_ or 0
    std::unique_ptr<float[], AlignedFree> vtx_ext_feedback_;

    /**** full-state linear feedback (internal; construction-only enable) ****/
    bool fsf_enabled_ = false;
    uint64_t fsf_seed_ = 1;
    float fsf_scaling_ = 0.5f;
    float fsf_stage_scaling_ = 1.0f;
    size_t num_fsf_weights_ = 0; // n_ * dim_ or 0
    std::unique_ptr<float[], AlignedFree> vtx_fsf_; // staging buffer
    std::vector<float> fsf_v_; // full-state vector V ∈ U(-1,1)^N when enabled (empty when off)

    /**** per neuron bias ****/
    float bias_scaling_;

    /**** Lorentzian activation envelope (see ReservoirConfig) ****/
    float lorentz_gamma_      = 1.1f;
    float lorentz_inv_sigma2_ = 250.0f;

    void Initialize();
    void UpdateState(size_t v, float old_output_v);
    [[nodiscard]] float EstimateSpectralRadius(std::span<float> x, std::span<float> y) const;

    /// Index (in floats) of the recurrent weight block — after input, ext-fb, FSF.
    [[nodiscard]] size_t RecurrentWeightBase() const
    {
        return num_input_weights_ + num_ext_feedback_weights_ + num_fsf_weights_;
    }
};
