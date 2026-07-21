#pragma once

/// @file FsfAbSwitch.h
/// @brief Shared A/B controls for full-state linear feedback (FSF) in examples.
///
/// When enabled: V ~ U(-1,1) from fsf_seed; B_fsf from same seed with fsf_scaling.
/// Step: φ = fsf_stage_scaling·(V·x), fill vtx_fsf_ with φ (ext-fb D=1). No Set/Get.
///
/// See docs/full_state_linear_feedback.md.

#include "ESN.h"
#include "Reservoir.h"

#include <cstdint>
#include <iostream>

namespace fsf_ab
{
    inline constexpr bool kEnable = true;
    inline constexpr std::uint64_t kSeed = 13459873;
    /// B_fsf construction: U(-1,1) × kScaling/√dim.
    inline constexpr float kScaling = 0.5f;
    /// Step: φ = kStageScaling · (V·x), then fill vtx_fsf_ with φ.
    inline constexpr float kStageScaling = 0.1f;

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
