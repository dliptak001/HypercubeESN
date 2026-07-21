#pragma once

/// @file FsfAbSwitch.h
/// @brief Shared A/B controls for full-state linear feedback (FSF) in examples.
///
/// When enabled: V ~ U(-1,1) from fsf_seed; B_fsf from same seed with fsf_scaling.
/// pad[v] = fsf_stage_scaling · (V·x) · V[v] in Step. No Set/Get of V.
///
/// See docs/full_state_linear_feedback.md.

#include "ESN.h"
#include "Reservoir.h"

#include <cstdint>
#include <iostream>

namespace fsf_ab
{
    inline constexpr bool kEnable = false;
    inline constexpr std::uint64_t kSeed = 1;
    /// B_fsf construction: U(-1,1) × kScaling/√dim.
    inline constexpr float kScaling = 0.5f;
    /// Step: pad[v] = kStageScaling · (V·x) · V[v].
    inline constexpr float kStageScaling = 1.0f;

    inline void ApplyTo(ReservoirConfig& r) noexcept
    {
        r.full_state_feedback = kEnable;
        r.fsf_seed = kSeed;
        r.fsf_scaling = kScaling;
        r.fsf_stage_scaling = kStageScaling;
    }

    inline void ApplyTo(ESNConfig& c) noexcept { ApplyTo(c.reservoir); }

    template <class OStream>
    inline void Log(OStream& os)
    {
        os << "  FSF A/B: " << (kEnable ? "ON " : "OFF")
           << "  fsf_seed=" << kSeed
           << "  fsf_scaling=" << kScaling
           << "  fsf_stage_scaling=" << kStageScaling
           << '\n';
    }
} // namespace fsf_ab
