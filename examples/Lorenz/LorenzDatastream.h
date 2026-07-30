#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Cursor.h"
#include "LorenzAttractor.h"

/// One normalized [-1, 1] Lorenz sample (float storage for the reservoir).
struct NormalizedState
{
    float x = 0, y = 0, z = 0;
};

/// Current cursor index + sample pointer (nullptr if index is off the stream).
struct LorenzDatastreamResult
{
    int32_t index = 0;
    const NormalizedState* sample = nullptr;
};

/// Orbit length + train window + IC for @ref LorenzDatastream construction.
/// Defaults of 0 are tripwires — fill before use.
struct LorenzDatastreamConfig
{
    int32_t span = 0;        ///< last train index; window [0, span] (Cursor)
    size_t stream_length = 0;///< RK4 steps stored after discard; storage is stream_length + 1 samples
    /// Integrate this many RK4 steps from the IC without storing (attractor burn-in).
    /// Free-run uses this to land at the same orbit phase as a full train-length wash edge
    /// while only allocating wash + free-run runway.
    size_t discard_steps = 0;
    LorenzAttractor::State initial_lorenz_state = {0.5, 0.5, 0.5};
    float lorenz_dt = 0.02f;
};

/// One Lorenz-63 orbit, normalized to float [-1, 1], walked by a forward @ref Cursor.
///
/// Construction: optional discard integrate → store integrate (double) → normalize
/// (shared scale over stored samples only) → float stream.
/// This class **is** a Cursor: Reset / OOB / Index / Span come from the base.
/// States / Step map the cursor into the stream buffer.
///
///     index 0 ====================== span ........ stream end
///            train section                free-run runway
///
/// Owns one orbit; rebuild per epoch / free-run.
class LorenzDatastream : public Cursor
{
public:
    explicit LorenzDatastream(const LorenzDatastreamConfig& cfg, bool print_header = false);

    /// Sample at the current index (no advance).
    [[nodiscard]] LorenzDatastreamResult States() const;

    /// Advance cursor one step, then return sample at the new index.
    LorenzDatastreamResult Step();

    [[nodiscard]] const std::vector<NormalizedState>& GetDataStream() const { return data_stream_; }

    void PrintOrbit() const;

private:
    LorenzAttractor::State seed_state_;
    std::vector<NormalizedState> data_stream_;

    [[nodiscard]] const NormalizedState* SampleAt(int32_t index) const;

    /// Integrate @p discard_steps without storage, then store @p stream_length + 1 samples.
    static std::vector<LorenzAttractor::State> Build(size_t stream_length,
                                                     size_t discard_steps,
                                                     const LorenzAttractor::State& seed,
                                                     float dt);

    void Normalize(const std::vector<LorenzAttractor::State>& raw);
};
