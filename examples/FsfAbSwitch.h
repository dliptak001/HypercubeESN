#pragma once

/// @file FsfAbSwitch.h
/// @brief Shared A/B controls for full-state linear feedback (FSF) in examples.
///
/// When enabled, V is drawn U(-1,1) from @c fsf_seed (no scale baked in); B_fsf
/// from the same seed with @c fsf_scaling. Score/stage scales apply in Step only
/// so φ and pad painting stay independently tunable. No Set/Get of V.
///
/// @code
///   ESNConfig cfg;
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

    /// Allocate FSF and apply the internal path each step (false ⇒ zero alloc).
    inline constexpr bool kEnable = false;

    /// Seeds V (U(-1,1)) then B_fsf (standalone; not mixed from reservoir.seed).
    inline constexpr std::uint64_t kSeed = 1;

    /// B_fsf construction: U(-1,1) × kScaling/√dim.
    inline constexpr float kScaling = 0.5f;

    /// Step: φ = kScoreScaling · (V · x).
    inline constexpr float kScoreScaling = 1.0f;

    /// Step: pad[v] = kStageScaling · φ · V[v].
    inline constexpr float kStageScaling = 1.0f;

    // =========================================================================

    inline void ApplyTo(ReservoirConfig& r) noexcept
    {
        r.full_state_feedback = kEnable;
        r.fsf_seed = kSeed;
        r.fsf_scaling = kScaling;
        r.fsf_score_scaling = kScoreScaling;
        r.fsf_stage_scaling = kStageScaling;
    }

    inline void ApplyTo(ESNConfig& c) noexcept { ApplyTo(c.reservoir); }

    template <class OStream>
    inline void Log(OStream& os)
    {
        os << "  FSF A/B: " << (kEnable ? "ON " : "OFF")
           << "  fsf_seed=" << kSeed
           << "  fsf_scaling=" << kScaling
           << "  fsf_score_scaling=" << kScoreScaling
           << "  fsf_stage_scaling=" << kStageScaling
           << '\n';
    }
} // namespace fsf_ab
