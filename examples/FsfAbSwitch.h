#pragma once

/// @file FsfAbSwitch.h
/// @brief Shared A/B controls for full-state linear feedback (FSF) in examples.
///
/// Two independent ideas:
///   1. @c kEnable — allocate the FSF port and apply φ = V·x every step.
///   2. @c kSetGain — after construct, install a simple default gain V
///      (otherwise V stays 0 and FSF-on matches FSF-off until you set V yourself).
///
/// Production use of FSF is the same API: enable at construction, then
/// @c SetFullStateFeedbackGain with your V (hand-chosen, loaded, or optimized).
/// The isotropic @c kSetGain path is only a convenience starter for A/B, not a
/// special “demo mode.”
///
/// @code
///   ESNConfig cfg;
///   // ... other knobs ...
///   fsf_ab::ApplyTo(cfg);
///   ESN esn(cfg);
///   fsf_ab::MaybeSetGain(esn);   // no-op unless kEnable && kSetGain
///   // or: esn.SetFullStateFeedbackGain(my_V.data(), my_V.size());
///   fsf_ab::Log(std::cout);
/// @endcode
///
/// See docs/full_state_linear_feedback.md.

#include "ESN.h"
#include "Reservoir.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace fsf_ab
{
    // =========================================================================
    //  A/B SWITCHES
    // =========================================================================

    /// Allocate FSF port and apply φ = V·x each step (false ⇒ zero FSF allocation).
    inline constexpr bool kEnable = false;

    /// Seeds only B_fsf (standalone; not mixed from reservoir.seed).
    inline constexpr std::uint64_t kSeed = 1;

    /// How hard the FSF *injection weights* push φ into the reservoir
    /// (same role as input_scaling: weights × scaling/√dim). Independent of V.
    inline constexpr float kScaling = 0.5f;

    /// If true (and @c kEnable), install a default isotropic gain V after construct.
    /// If false, V stays 0 — FSF-on with V=0 matches FSF-off until you call
    /// @c SetFullStateFeedbackGain yourself.
    inline constexpr bool kSetGain = false;

    /// Size of that default V when @c kSetGain: each component = kGainScale/√N
    /// so ‖V‖₂ ≈ |kGainScale|. Ignored if @c kSetGain is false.
    inline constexpr float kGainScale = 0.05f;

    // =========================================================================

    inline void ApplyTo(ReservoirConfig& r) noexcept
    {
        r.full_state_feedback = kEnable;
        r.fsf_seed = kSeed;
        r.fsf_scaling = kScaling;
    }

    inline void ApplyTo(ESNConfig& c) noexcept { ApplyTo(c.reservoir); }

    template <class OStream>
    inline void Log(OStream& os)
    {
        os << "  FSF A/B: " << (kEnable ? "ON " : "OFF")
           << "  fsf_seed=" << kSeed
           << "  fsf_scaling=" << kScaling;
        if (kEnable)
        {
            os << "  gain="
               << (kSetGain ? "default isotropic (kGainScale)"
                            : "V=0 until SetFullStateFeedbackGain");
        }
        os << '\n';
    }

    /// Install default isotropic V when @c kEnable && @c kSetGain; otherwise no-op.
    inline void MaybeSetGain(Reservoir& r)
    {
        if (!kEnable || !kSetGain || !r.FullStateFeedbackEnabled())
            return;
        const std::size_t n = r.Size();
        std::vector<float> V(n, 0.0f);
        const float v = kGainScale / std::sqrt(static_cast<float>(n));
        for (float& x : V)
            x = v;
        r.SetFullStateFeedbackGain(V.data(), V.size());
    }

    inline void MaybeSetGain(ESN& esn)
    {
        if (!kEnable || !kSetGain || !esn.FullStateFeedbackEnabled())
            return;
        const std::size_t n = esn.ReservoirNeuronCount();
        std::vector<float> V(n, 0.0f);
        const float v = kGainScale / std::sqrt(static_cast<float>(n));
        for (float& x : V)
            x = v;
        esn.SetFullStateFeedbackGain(V.data(), V.size());
    }
} // namespace fsf_ab
