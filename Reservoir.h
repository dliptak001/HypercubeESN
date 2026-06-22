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

    size_t num_feedback_channels = 0; // number of feedback driver channels; 0 disables the feedback path entirely. Any D in [1, N] is admissible — D need NOT divide N (a non-dividing D leaves the N mod D tail vertices at reset-zero, benign zero sources that still receive drive via the neighbor gather)
    float feedback_scaling = 0.5f; // DIM-invariant feedback drive: feedback weights carry a 1/sqrt(DIM) fan-in normalization, mirroring input_scaling (only allocated/used when num_feedback_channels > 0)

    float bias_scaling = 0.02f; // per-neuron additive bias drawn U(-1,1)*bias_scaling, summed pre-tanh; OFF by default (0 disables)

    // --- Lorentzian activation envelope ---
    // A_lorentz(x) = tanh(x * gain),  gain = 1 + lorentz_gamma * phi,
    // phi = 1/(1 + x^2 * lorentz_inv_sigma2) ∈ (0,1].
    //   gamma = 0           => gain ≡ 1 => plain tanh(x)
    //   gamma > 0           => steeper central slope, tanh tails (sharpening)
    //   gamma < 0, |g| > 1  => central gain crosses 0 => non-monotone "fold"
    // Runtime so the activation shape is a sweep axis (no recompile).
    float lorentz_gamma      = 1.1f;  // 0 reduces A_lorentz to tanh
    float lorentz_inv_sigma2 = 250.0f; // 1/sigma^2
};

/// @brief Reservoir-computing reservoir whose recurrent topology is a Boolean
/// hypercube graph: N = 2^dim neurons sit on hypercube vertices and each
/// neuron connects to its dim single-bit-flip (Hamming-distance-1) neighbors.
/// Connectivity is computed inline via XOR masks (@ref NearestMask) — no
/// adjacency is stored.
///
/// The hypercube dimension is a runtime config field (@c ReservoirConfig::dim),
/// so a single type serves all dimensions; the neuron count N = 2^dim and all
/// state/weight buffers are sized at construction. Input weights are decoupled
/// from the recurrent weights (a separate per-edge input block), and the
/// recurrent path carries a configurable @c history_depth delay line (M past
/// output slices), giving the readout direct access to short-range temporal
/// structure. Recurrent weights are rescaled at construction to a target
/// spectral radius.
///
/// An optional feedback path (@c ReservoirConfig::num_feedback_channels > 0)
/// adds a second per-vertex driver buffer, mechanically identical to the input
/// path: its own weight block, summed into each neuron via the same dim-neighbor
/// gather, injected by @ref InjectFeedback and consumed each @ref Step. It is the
/// closed-loop hook (e.g. routing a readout-derived signal back in); with zero
/// channels it is a no-op — no weights are allocated and the open-loop
/// realization is unchanged. Note the feedback block sits OUTSIDE the
/// spectral-radius estimate, so it does not bound closed-loop stability.
///
/// Every neuron also carries a fixed additive bias (@c ReservoirConfig::bias_scaling),
/// drawn once at construction from a @c Bias-labelled substream of the single
/// @c ReservoirConfig::seed as @c U(-1,1) * bias_scaling and summed into the
/// pre-activation just before the @c tanh. Like feedback, bias is OFF by default
/// (bias_scaling = 0.0); set bias_scaling > 0 to enable it. It is a fixed model
/// parameter, not dynamical state — @ref Clear leaves it untouched and @ref TakeSnapshot
/// does not capture it. Like the feedback block it sits OUTSIDE the spectral-radius
/// estimate (it is an additive constant, not part of the linear recurrent operator)
/// and carries no fan-in normalization (a single per-neuron term has no fan-in).
///
/// Lifetime: non-copyable and non-movable; obtain instances via @ref Create.
/// Per-step contract: @ref InjectInput and/or @ref InjectFeedback (both optional)
/// then @ref Step; the injected input and feedback are consumed and cleared by
/// each Step.
class Reservoir
{
public:
    /// Inline neighbor mask computation — no stored adjacency.
    /// Mask for neighbor i in [0, dim):  1 << i  →  1, 2, 4, 8, ...
    static constexpr uint32_t NearestMask(size_t i) { return 1u << i; }

    /// @brief Construct a reservoir from a fully resolved config.
    /// This is the only way to obtain an instance (the type is non-copyable and
    /// non-movable). Initializes weights, rescales them to the target spectral
    /// radius, and leaves the reservoir reset (zero state).
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
    /// Recomputes every vertex state from its injected input and the recurrent
    /// history, ages the history ring by one slice (newest state becomes slice
    /// 0), and clears the per-step input. Pair with @ref InjectInput, which
    /// must precede each Step since the input is consumed here.
    void Step();

    /// @brief Stage the input for one channel ahead of the next @ref Step.
    /// Channel @p channel drives the contiguous vertex block
    /// [channel * N/num_inputs, (channel+1) * N/num_inputs). Inputs are cleared
    /// by every Step, so call this each timestep an input is desired.
    /// @throws std::invalid_argument if @p channel >= num_inputs.
    void InjectInput(size_t channel, float input);

    /// @brief Clear all state and history to zero and re-home the slice ring.
    /// Returns the reservoir to its post-construction (undriven) state; the
    /// learned/initialized weights are left unchanged.
    void Clear();

    /// @brief The current reservoir state — the most-recent history slice.
    /// @return Pointer to N floats (the readout feature vector), valid until the
    ///         next @ref Step or @ref Clear.
    [[nodiscard]] const float* Outputs() const { return slice_ptrs_[0]; }

    /// @brief Realized post-rescale spectral radius measured at construction
    /// (the secant root-find's final estimate), which approximates the
    /// configured target spectral_radius.
    [[nodiscard]] float GetRealizedSpectralRadius() const { return realized_spectral_radius_; }

    /// @brief Reconstruct the @ref ReservoirConfig this reservoir was built from.
    /// Every field is read back from a stored member, so @c Create(GetConfig())
    /// rebuilds an identical reservoir (the weights are deterministic in the
    /// seed). @c spectral_radius is the configured TARGET, not the realized value
    /// — use @ref GetRealizedSpectralRadius for the post-rescale estimate. Gives
    /// consumers the access needed to serialize a standalone reservoir.
    [[nodiscard]] ReservoirConfig GetConfig() const;

    /// Hypercube dimension; the reservoir has N = 2^Dim() neurons.
    [[nodiscard]] size_t Dim() const { return dim_; }

    /// Neuron count N = 2^Dim() (the length of the @ref Outputs feature vector).
    [[nodiscard]] size_t Size() const { return n_; }

    /// @brief Stage the feedback signal for one channel ahead of the next @ref Step.
    /// The closed-loop analogue of @ref InjectInput: channel @p channel drives the
    /// contiguous vertex block [channel * N/num_feedback_channels,
    /// (channel+1) * N/num_feedback_channels) through the separate feedback weight
    /// block. Like input, feedback is cleared by every @ref Step, so call this each
    /// timestep a feedback drive is desired (typically with the previous step's
    /// readout-derived value y(t-1), since y(t) does not yet exist when Step needs
    /// it). Requires the reservoir to have been built with num_feedback_channels > 0.
    /// @throws std::invalid_argument if feedback is not configured
    ///         (num_feedback_channels == 0) or @p channel >= num_feedback_channels.
    void InjectFeedback(size_t channel, float feedback);

    /// @brief Vector form: stage all @p count feedback channels at once ahead of
    /// the next @ref Step. Convenience wrapper over the per-channel overload — the
    /// external-drive entry point for a D-channel feedback port.
    /// @throws std::invalid_argument if @p count != num_feedback_channels.
    void InjectFeedback(const float* feedback, size_t count);

    /// @brief Copyable capture of the reservoir's persistent dynamical state:
    /// the live vertex state plus every history slice in logical age order
    /// (slice 0 = most recent). The per-step staged drives (input/feedback)
    /// are NOT captured — they are consumed and cleared by every @ref Step, so
    /// a snapshot taken between steps has nothing staged. Weights are not
    /// included: a snapshot is only meaningful for the reservoir it was taken
    /// from (or one built from an identical config).
    struct Snapshot
    {
        std::vector<float> state; ///< live vertex state, N floats
        std::vector<float> history; ///< N * history_depth floats, slice-major, most recent slice first
    };

    /// @brief Capture the persistent dynamical state (state + history ring).
    /// Call between steps (per-step staged input/feedback are not captured).
    /// History slices are stored in logical age order regardless of the ring's
    /// current rotation, so the snapshot is canonical: snapshots of identical
    /// dynamics compare equal even when taken at different ring phases.
    [[nodiscard]] Snapshot TakeSnapshot() const;

    /// @brief Bit-exact restore of a state captured by @ref TakeSnapshot.
    /// Copies the state and history back, re-homes the slice ring to the
    /// canonical rotation, and clears any staged input/feedback — so the
    /// post-restore trajectory depends only on the snapshot and subsequent
    /// injections: restoring and replaying the same inputs reproduces the
    /// identical trajectory bit-for-bit — a branch-point primitive for any
    /// snapshot-and-replay use.
    /// @throws std::invalid_argument if the snapshot's buffer sizes do not
    ///         match this reservoir's N and history_depth.
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

    /**** feedback ****/
    size_t num_feedback_channels_ = 0;
    float feedback_scaling_ = 1.0f;
    size_t num_feedback_weights_ = 0; // n_ * dim_ — size of the feedback-weight block
    std::unique_ptr<float[], AlignedFree> vtx_feedback_;

    /**** per neuron bias ****/
    float bias_scaling_;

    /**** Lorentzian activation envelope (see ReservoirConfig) ****/
    float lorentz_gamma_      = 1.1f;
    float lorentz_inv_sigma2_ = 250.0f;

    void Initialize();
    void UpdateState(size_t v, float old_output_v);
    [[nodiscard]] float EstimateSpectralRadius(std::span<float> x, std::span<float> y) const;
};
