#pragma once

/// @file FsfAbSwitch.h
/// @brief Shared A/B controls for full-state linear feedback (FSF) in examples.
///
/// When enabled, V and B_fsf are drawn at construction from @c fsf_seed
/// (@c fsf_v_scaling / @c fsf_scaling). No Set/Get of V — seed + scales only.
///
/// @code
///   ESNConfig cfg;
///   // ... other knobs ...
///   fsf_ab::ApplyTo(cfg);
///   ESN esn(cfg);
///   fsf_ab::Log(std::cout);
/// @endcode
///
/// See docs/full_state_linear_feedback.md.

#include "ESN.h"
#include "Reservoir.h"

#include <cstdint>
#include <iostream>

namespace fsf_ab
{
    // =========================================================================
    //  A/B SWITCHES
    // =========================================================================

    /// Allocate FSF and apply φ = V·x / pad = φ⊙V each step (false ⇒ zero alloc).
    inline constexpr bool kEnable = false;

    /// Seeds V then B_fsf (standalone; not mixed from reservoir.seed).
    inline constexpr std::uint64_t kSeed = 1;

    /// B_fsf: U(-1,1) × kScaling/√dim (inject/gather weights).
    inline constexpr float kScaling = 0.5f;

    /// V: U(-1,1) × kVScaling (state→φ and pad paint; w ≡ V).
    inline constexpr float kVScaling = 1.0f;

    // =========================================================================

    inline void ApplyTo(ReservoirConfig& r) noexcept
    {
        r.full_state_feedback = kEnable;
        r.fsf_seed = kSeed;
        r.fsf_scaling = kScaling;
        r.fsf_v_scaling = kVScaling;
    }

    inline void ApplyTo(ESNConfig& c) noexcept { ApplyTo(c.reservoir); }

    template <class OStream>
    inline void Log(OStream& os)
    {
        os << "  FSF A/B: " << (kEnable ? "ON " : "OFF")
           << "  fsf_seed=" << kSeed
           << "  fsf_scaling=" << kScaling
           << "  fsf_v_scaling=" << kVScaling
           << '\n';
    }
} // namespace fsf_ab
