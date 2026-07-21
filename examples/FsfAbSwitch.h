#pragma once

/// @file FsfAbSwitch.h
/// @brief Single A/B control for full-state linear feedback across examples.
///
/// Flip @c kEnable (and optionally @c kApplyDemoGain) here or override per-example
/// after @c ApplyTo. See docs/full_state_linear_feedback.md.
///
/// Usage (typical):
/// @code
///   ESNConfig cfg;
///   // ... other knobs ...
///   fsf_ab::ApplyTo(cfg);
///   ESN esn(cfg);
///   fsf_ab::Log(std::cout);
///   fsf_ab::MaybeSetDemoGain(esn);  // no-op unless kEnable && kApplyDemoGain
/// @endcode
///
/// Semantics:
/// - @c kEnable false — FSF port not allocated (default open-loop).
/// - @c kEnable true, V left at 0 — port on, dynamics bit-identical to off (F1).
/// - @c kEnable true + @c kApplyDemoGain / your own SetFullStateFeedbackGain —
///   closed FSF map for A/B against the off baseline.

#include "ESN.h"
#include "Reservoir.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace fsf_ab
{
    // =========================================================================
    //  A/B SWITCHES — edit these (or override fields after ApplyTo)
    // =========================================================================

    /// Master enable: construction-only FSF port (false ⇒ zero allocation).
    inline constexpr bool kEnable = false;

    /// Seeds only B_fsf (standalone; not mixed from reservoir.seed).
    inline constexpr std::uint64_t kSeed = 1;

    /// DIM-invariant FSF drive scale (weights × scaling/√dim).
    inline constexpr float kScaling = 0.5f;

    /// When true and @c kEnable, apply a small isotropic demo gain after construct.
    /// Leave false to A/B "port on, V=0" vs off, or set V yourself.
    inline constexpr bool kApplyDemoGain = false;

    /// Scale for demo V: each component = kDemoGainScale / √N (‖V‖₂ ≈ |scale|).
    inline constexpr float kDemoGainScale = 0.05f;

    // =========================================================================

    inline void ApplyTo(ReservoirConfig& r) noexcept
    {
        r.full_state_feedback = kEnable;
        r.fsf_seed = kSeed;
        r.fsf_scaling = kScaling;
    }

    inline void ApplyTo(ESNConfig& c) noexcept { ApplyTo(c.reservoir); }

    /// One-line status for logs (use cout or cerr as the example already does).
    template <class OStream>
    inline void Log(OStream& os)
    {
        os << "  FSF A/B: " << (kEnable ? "ON " : "OFF")
           << "  fsf_seed=" << kSeed
           << "  fsf_scaling=" << kScaling;
        if (kEnable)
            os << "  demo_gain=" << (kApplyDemoGain ? "yes" : "no (V=0 until SetFullStateFeedbackGain)");
        os << '\n';
    }

    /// Optional isotropic demo V. No-op if FSF is off or @c kApplyDemoGain is false.
    inline void MaybeSetDemoGain(Reservoir& r)
    {
        if (!kEnable || !kApplyDemoGain || !r.FullStateFeedbackEnabled())
            return;
        const std::size_t n = r.Size();
        std::vector<float> V(n, 0.0f);
        const float v = kDemoGainScale / std::sqrt(static_cast<float>(n));
        for (float& x : V)
            x = v;
        r.SetFullStateFeedbackGain(V.data(), V.size());
    }

    inline void MaybeSetDemoGain(ESN& esn)
    {
        if (!kEnable || !kApplyDemoGain || !esn.FullStateFeedbackEnabled())
            return;
        const std::size_t n = esn.ReservoirNeuronCount();
        std::vector<float> V(n, 0.0f);
        const float v = kDemoGainScale / std::sqrt(static_cast<float>(n));
        for (float& x : V)
            x = v;
        esn.SetFullStateFeedbackGain(V.data(), V.size());
    }
} // namespace fsf_ab
