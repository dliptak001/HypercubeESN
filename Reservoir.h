#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <span>
#include <vector>

/// @brief Construction-time parameters for @ref Reservoir.
///
/// All fields are fixed at @ref Reservoir::Create; dynamics (state, history,
/// staged drives) are not part of this struct. @ref Reservoir::GetConfig returns
/// an equivalent snapshot; @c spectral_radius is the **target** used for the
/// recurrent rescale, not the realized estimate
/// (@ref Reservoir::GetRealizedSpectralRadius).
struct ReservoirConfig
{
    /// Hypercube dimension; neuron count N = 2^dim. Valid range **[5, 16]**.
    size_t dim = 10;

    /// Master RNG seed. Named substreams (recurrent / input / external-feedback /
    /// bias / SR probe) are derived via SplitMix64 — see Reservoir.cpp.
    uint64_t seed = 73895;

    /// Target spectral radius for the **recurrent** weight block only (> 0).
    /// Drive-port weights (input, external feedback) are outside this rescale.
    float spectral_radius = 0.99f;

    /// Leaky-integrator mix: 1 = full replacement each step; in (0, 1) blends with
    /// previous output. Valid range **(0, 1]**.
    float leak_rate = 1.0f;

    /// Input drive strength. Input weights are drawn U(-1,1) then scaled by
    /// @c input_scaling / √dim (fan-in normalization). Retune per task/DIM.
    float input_scaling = 0.5f;

    /// Number of input channels (≥ 1). Must **divide N** evenly so each channel
    /// owns a contiguous vertex block of size N/num_inputs.
    size_t num_inputs = 1;

    /// Delay-line length M: past published slices in the recurrent gather.
    /// Valid range **[1, 64]**.
    size_t history_depth = 16;

    /// If true, print one construction banner (DIM, M, seed, leak, scales, SR).
    bool verbose = true;

    // --- External feedback (caller-owned closed-loop drive) ---
    /// 0 = path disabled (no buffer, no weights). Else D in **[1, N]**; D need
    /// **not** divide N (tail vertices stay zero as sources but still receive
    /// drive via the neighbor gather).
    size_t num_external_feedback_channels = 0;

    /// External-feedback weight scale: U(-1,1) then × scaling/√dim (like input).
    /// Only used when @c num_external_feedback_channels > 0.
    float external_feedback_scaling = 0.5f;

    /// Per-neuron bias: U(-1,1) × bias_scaling, added **after** tanh. **0**
    /// disables bias. Default is a small nonzero scale (not “off”).
    float bias_scaling = 0.02f;
};

/// @brief Fixed (never-trained) recurrent core of an @ref ESN: N = 2^dim neurons
/// on a Boolean hypercube.
///
/// ## Topology
/// Neuron @c v connects only to its @c dim Hamming-distance-1 neighbors. Neighbor
/// @c i is @c v XOR (1<<i) (@ref NearestMask) — pure arithmetic, no adjacency list.
/// @c dim is a runtime field (one class for every legal size).
///
/// ## One step
/// For each neuron, @ref Step gathers neighbors through separate weight blocks,
/// applies plain @c tanh, adds fixed bias, and leak-blends with the previous value:
/// ```
///   input ──────────────────┐
///   recurrent history ──────┼─▶ Σ (dim neighbors) ─▶ tanh ─▶ +bias ─▶ leak ─▶ state
///   external feedback (opt) ┘
/// ```
/// Recurrent weights are rescaled at construction to target @c spectral_radius
/// (see @ref GetRealizedSpectralRadius). Input and external-feedback weights sit
/// **outside** that rescale.
///
/// ## Per-step contract
/// ```
///   InjectInput(...)                  // stage task input (required when driving)
///   InjectExternalFeedback(...)       // optional — only if D > 0
///   Step()                            // update, age delay line, clear stages
///   // read Outputs() / SliceAt(age)
/// ```
/// Staged drives are **consumed and zeroed** by every @ref Step.
///
/// ## Optional ports
/// - **External feedback** — when D > 0: caller-owned twin of the input path.
/// - **Bias** — construction-time per-neuron offset when @c bias_scaling != 0.
///
/// ## Lifetime
/// Non-copyable and non-movable; obtain instances only via @ref Create.
class Reservoir
{
public:
    /// Neighbor bit mask for axis @p i in [0, dim): @c 1u << i.
    static constexpr uint32_t NearestMask(size_t i) { return 1u << i; }

    /// @brief Build a reservoir from @p cfg (only construction path).
    ///
    /// Draws weights from @c cfg.seed, rescales the recurrent block toward
    /// @c cfg.spectral_radius, and leaves dynamical state at rest (zero).
    /// @throws std::invalid_argument if any field is out of range (e.g. dim not in
    ///         [5, 16], num_inputs does not divide N, history_depth not in [1, 64]).
    static std::unique_ptr<Reservoir> Create(const ReservoirConfig& cfg)
    {
        return std::unique_ptr<Reservoir>(new Reservoir(cfg));
    }

    Reservoir(const Reservoir&) = delete;
    Reservoir& operator=(const Reservoir&) = delete;

    /// @brief Advance one timestep: update all vertices, age the delay-line ring
    /// (new state becomes logical age 0), clear staged input / external feedback.
    ///
    /// Stage drives first with @ref InjectInput / @ref InjectExternalFeedback.
    void Step();

    /// @brief Stage one input channel for the next @ref Step.
    ///
    /// Channel @p channel writes @p input onto vertices
    /// [channel · N/num_inputs, (channel+1) · N/num_inputs). Cleared by @ref Step.
    /// @throws std::invalid_argument if @p channel >= num_inputs.
    void InjectInput(size_t channel, float input);

    /// @brief Zero dynamical state and history; re-home the slice ring.
    ///
    /// Weights and bias are unchanged (construction-time). Staged drive buffers
    /// are cleared.
    void Clear();

    /// @brief Newest published state (delay-line age 0) — N floats for the readout.
    /// @return Valid until the next @ref Step or @ref Clear.
    [[nodiscard]] const float* Outputs() const { return slice_ptrs_[0]; }

    /// @brief Published state from @p age steps ago (logical age).
    ///
    /// @c SliceAt(0) ≡ @ref Outputs. Indexes **logical age**, not physical ring
    /// position — always use this (or @ref TakeSnapshot), never the raw history
    /// buffer layout. Ages 1 … history_depth-1 are the rest of the delay line the
    /// recurrent gather already uses.
    /// @return N floats; valid until the next @ref Step or @ref Clear.
    /// @throws std::out_of_range if @p age >= history_depth.
    [[nodiscard]] const float* SliceAt(size_t age) const;

    /// @brief Post-rescale spectral-radius estimate from construction (secant root).
    /// Approximates the configured target @c spectral_radius.
    [[nodiscard]] float GetRealizedSpectralRadius() const { return realized_spectral_radius_; }

    /// @brief Config this instance was built from (all fields).
    ///
    /// @c Create(GetConfig()) rebuilds matching weights from @c seed.
    /// @c spectral_radius is the **target**, not the realized estimate.
    [[nodiscard]] ReservoirConfig GetConfig() const;

    /// @brief Hypercube dimension; N = 2^Dim().
    [[nodiscard]] size_t Dim() const { return dim_; }

    /// @brief Neuron count N = 2^Dim() (length of @ref Outputs).
    [[nodiscard]] size_t Size() const { return n_; }

    /// @brief Stage one external-feedback channel for the next @ref Step.
    ///
    /// Twin of @ref InjectInput on the external-feedback weight block. Channel
    /// @p channel drives [channel · floor(N/D), (channel+1) · floor(N/D)). Cleared
    /// by @ref Step. Requires D = num_external_feedback_channels > 0.
    /// @throws std::invalid_argument if external feedback is not configured or
    ///         @p channel is out of range.
    void InjectExternalFeedback(size_t channel, float value);

    /// @brief Stage all D external-feedback channels at once.
    /// @throws std::invalid_argument if @p count != D, or @p values is null when
    ///         count > 0.
    void InjectExternalFeedback(const float* values, size_t count);

    /// @brief Persistent dynamical state: live vertex values + full history ring
    /// in logical age order (slice 0 = most recent).
    ///
    /// Staged drives are **not** included (empty between steps). Weights and bias
    /// are not included — a snapshot is only meaningful for this reservoir or one
    /// rebuilt from an identical config.
    struct Snapshot
    {
        std::vector<float> state;   ///< N floats (live vtx_state_)
        std::vector<float> history; ///< N * history_depth, slice-major, newest first
    };

    /// @brief Capture state + history (logical age order; ring-phase independent).
    /// Call between steps.
    [[nodiscard]] Snapshot TakeSnapshot() const;

    /// @brief Restore a @ref TakeSnapshot capture: state, history, canonical ring
    /// home, cleared staged drives. Same drives afterward → bit-exact replay on an
    /// identically configured reservoir.
    /// @throws std::invalid_argument if buffer sizes do not match N and history_depth.
    void RestoreSnapshot(const Snapshot& snap);

private:
    explicit Reservoir(const ReservoirConfig& cfg);

    /// Deleter for @ref AllocAligned: must use the matching aligned
    /// @c ::operator delete[] (no array cookie; 64-byte alignment).
    struct AlignedFree
    {
        void operator()(float* p) const noexcept
        {
            ::operator delete[](p, std::align_val_t{64});
        }
    };

    /// Uninitialized @p count floats, 64-byte aligned. Free only via @ref AlignedFree.
    static float* AllocAligned(size_t count)
    {
        return static_cast<float*>(
            ::operator new[](count * sizeof(float), std::align_val_t{64}));
    }

    uint64_t rng_seed_ = 0;

    size_t dim_ = 0;              ///< Hypercube dimension
    size_t n_ = 0;                ///< N = 2^dim_
    size_t num_input_weights_ = 0; ///< n_ * dim_ (input weight block size)

    std::unique_ptr<float[], AlignedFree> vtx_input_;           ///< Staged input field
    std::unique_ptr<float[], AlignedFree> vtx_state_;           ///< Write target of UpdateState
    std::unique_ptr<float[], AlignedFree> vtx_output_history_;  ///< Ring storage (M blocks of N)
    std::unique_ptr<float[], AlignedFree> vtx_weight_;          ///< [in | ext? | rec]
    std::unique_ptr<float*[]> slice_ptrs_;                      ///< Logical age → history block
    std::unique_ptr<float[], AlignedFree> vtx_bias_;            ///< Fixed per-neuron bias

    size_t num_inputs_ = 1;
    float spectral_radius_ = 0.99f;           ///< Configured target
    float leak_rate_ = 1.0f;
    float input_scaling_ = 0.5f;
    float realized_spectral_radius_ = 0.0f; ///< Set in Initialize after secant
    bool verbose_ = true;
    size_t history_depth_ = 1;
    size_t num_weights_ = 0; ///< Total floats in vtx_weight_

    size_t num_ext_feedback_channels_ = 0;
    float ext_feedback_scaling_ = 0.5f;
    size_t num_ext_feedback_weights_ = 0; ///< n_ * dim_ or 0
    std::unique_ptr<float[], AlignedFree> vtx_ext_feedback_; ///< Staged ext-fb field

    float bias_scaling_ = 0.0f;

    void Initialize();
    void UpdateState(size_t v, float old_output_v);
    [[nodiscard]] float EstimateSpectralRadius(std::span<float> x, std::span<float> y) const;

    /// Byte index (in floats) of the recurrent block: after input, then ext-fb if any.
    [[nodiscard]] size_t RecurrentWeightBase() const
    {
        return num_input_weights_ + num_ext_feedback_weights_;
    }
};
