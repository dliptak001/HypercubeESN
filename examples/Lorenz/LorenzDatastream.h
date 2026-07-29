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
    size_t stream_length = 0;///< RK4 steps; storage is stream_length + 1 samples
    LorenzAttractor::State initial_lorenz_state = {0.5, 0.5, 0.5};
    float lorenz_dt = 0.02f;
};

/// One Lorenz-63 orbit, normalized to float [-1, 1], walked by a forward @ref Cursor.
///
/// Construction: integrate (double) → normalize (shared scale) → float stream.
/// This class **is** a Cursor: Reset / OOB / Index / Span come from the base.
/// States / Step map the cursor into the stream buffer.
///
///     index 0 ====================== span ........ stream end
///            train / washout              free-run runway
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

    static std::vector<LorenzAttractor::State> Build(size_t stream_length,
                                                     const LorenzAttractor::State& seed,
                                                     float dt);

    void Normalize(const std::vector<LorenzAttractor::State>& raw);
};
