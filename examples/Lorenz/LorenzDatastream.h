#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "Cursor.h"
#include "LorenzAttractor.h"


/// @brief One normalized [-1, 1] Lorenz sample (x, y, z) in float storage.
///
/// The reservoir consumes floats, so the stream is narrowed once in
/// @ref LorenzDatastream::Normalize rather than per step.
struct NormalizedState
{
    float x = 0, y = 0, z = 0;
};

/// @brief What @ref LorenzDatastream::States and @ref LorenzDatastream::Step hand
/// back for the current cursor position.
struct LorenzDatastreamResult
{
    int32_t index = 0; ///< current forward cursor index
    const NormalizedState* sample = nullptr; ///< stream[index] when in-bounds; nullptr if
                                             ///< index is outside the integrated stream.
};

/// @brief Construction parameters for @ref LorenzDatastream: how long an orbit to
/// integrate and where the training window sits on it.
///
/// The geometry fields default to 0 as tripwires — a default-constructed config
/// must be filled in (span > 0, stream_length > 0, span within the stream).
struct LorenzDatastreamConfig
{
    int32_t cursor_span = 0;         ///< last train index (must be > 0); window is [0, span]
    size_t stream_length = 0;        ///< number of RK4 steps (stream holds stream_length + 1 samples)
    LorenzAttractor::State initial_lorenz_state = {0.5, 0.5, 0.5};
    float lorenz_dt = 0.02f;         ///< RK4 integration step (canonical Lorenz-63 dt)
};

/// @brief Integrates one Lorenz-63 orbit, normalizes it to float [-1, 1], and
/// serves samples through a single forward @ref Cursor.
///
/// Construction integrates `stream_length + 1` samples (via @ref LorenzAttractor),
/// then @ref Normalize maps them to [-1, 1] with a per-channel midpoint offset and
/// one shared scale (relative amplitudes preserved).
///
/// Layout (train [0, span] inclusive):
///
///     index 0 ====================== span          stream end
///            training / washout          eval / free-run runway
///
///   @ref States  -> {index, &stream[index]} when in-bounds; sample is nullptr
///                   if the index is outside the stream.
///   @ref Step    -> advances the cursor, then returns the same pair.
///
/// One instance owns one orbit; the harness rebuilds it per epoch / free-run.
class LorenzDatastream : public Cursor
{
public:
    LorenzDatastream(const LorenzDatastreamConfig& cfg, bool print_header = false);

    [[nodiscard]] LorenzDatastreamResult States();
    LorenzDatastreamResult Step();

    [[nodiscard]] const std::vector<NormalizedState>& GetDataStream() const { return data_stream_; }

    void PrintOrbit();

private:
    LorenzDatastreamConfig cfg_;

    std::vector<NormalizedState> data_stream_;

    [[nodiscard]] std::vector<LorenzAttractor::State> Build(size_t stream_length,
                                                            const LorenzAttractor::State& initial_lorenz_state,
                                                            float lorenz_dt) const;

    void Normalize(const std::vector<LorenzAttractor::State>& raw);

    [[nodiscard]] const NormalizedState* SampleAt(int32_t index) const;
};
