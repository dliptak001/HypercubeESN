#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "JanusCursor.h"
#include "LorenzAttractor.h"


/// @brief One normalized [-1, 1] Lorenz sample (x, y, z) in float storage.
///
/// The reservoir consumes floats, so the stream is narrowed once in
/// @ref LorenzDatastream::Normalize rather than per step — and the hot cursor
/// window then occupies half the cache it would as double storage.
struct NormalizedState
{
    float x = 0, y = 0, z = 0;
};

/// @brief What @ref LorenzDatastream::States and @ref LorenzDatastream::Step hand
/// back for the current cursor position.
struct LorenzDatastreamResult
{
    float distance = 0;         ///< @ref JanusCursor::Distance (normalized cursor separation, -1 .. +1)
    NormalizedState past;       ///< the PAST sample (by value; the past cursor always addresses valid history)
    const NormalizedState* future = nullptr; ///< the FUTURE sample, or nullptr once the future cursor has run
                                             ///< out of in-window data (OOB). Callers MUST null-check before deref.
};

/// @brief Construction parameters for @ref LorenzDatastream: how long an orbit to
/// integrate and where the Janus cursor window sits on it.
///
/// The three geometry fields default to 0 as tripwires, not usable values — a
/// default-constructed config is REQUIRED to be filled in, and passing one as-is
/// throws from the LorenzDatastream / JanusCursor constructors (span > 0,
/// stream_length > 0, window within [0, stream_length]). Only the orbit fields
/// (initial state, dt) carry sensible defaults.
struct LorenzDatastreamConfig
{
    int32_t cursor_span = 0;        ///< width of the Janus training window (must be > 0)
    int32_t cursor_center_index = 0;///< stream index the window is centered on (must exceed span/2)
    size_t stream_length = 0;       ///< number of RK4 steps to integrate (stream holds stream_length + 1 samples)
    LorenzAttractor::State initial_lorenz_state = {0.5, 0.5, 0.5}; ///< orbit initial condition
    float lorenz_dt = 0.02;         ///< RK4 integration step (canonical Lorenz-63 dt)
};

/// @brief The Lorenz example's data source: integrates one Lorenz-63 orbit,
/// normalizes it to float [-1, 1], and serves it through a pair of Janus cursors.
///
/// Construction does all the heavy lifting once: @ref Build integrates
/// `stream_length + 1` samples in full double precision (via @ref LorenzAttractor),
/// then @ref Normalize maps them to [-1, 1] using a PER-CHANNEL offset but a SINGLE
/// SHARED scale, so the attractor's true relative amplitudes are preserved (x really
/// is narrower than y; z is shifted down off its ~+24 DC level). The result lives in
/// @ref data_stream_ as compact float storage the reservoir can consume directly.
///
/// Read access is via the inherited JanusCursor — two cursors over the same stream:
///
///     index 0            lb          center           ub              N
///        |---------------|===============================|-------------|
///          past runway            training window          eval runway
///        (anchor history)     (past & future cursors     (future cursor's
///                              sweep this span)            generative tail)
///
///   @ref States  -> {Distance, stream[past], &stream[future]} at the current spot;
///                   future is nullptr when OOB (same pointer contract as Step — never
///                   forms an out-of-window address). Throws if past has underrun.
///   @ref Step    -> advances both cursors, then returns the same triple; the future
///                   pointer is nullptr once the future cursor runs off the window
///                   (OOB), and Step throws if the past cursor underruns its history.
///
/// One instance owns one orbit; the harness rebuilds it per epoch / free-run.
class LorenzDatastream : public JanusCursor
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
};
