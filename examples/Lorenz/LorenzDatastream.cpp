#include "LorenzDatastream.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

LorenzDatastream::LorenzDatastream(const LorenzDatastreamConfig& cfg)
    : JanusCursor(cfg.cursor_span, cfg.cursor_center_index)
{
    if (cfg.stream_length == 0)
        throw std::out_of_range("LorenzDatastream::LorenzDatastream - expecting stream_length > 0");

    if (cfg.lorenz_dt <= 0.0f)
        throw std::out_of_range("LorenzDatastream::LorenzDatastream - expecting lorenz_dt > 0");

    const int32_t window_lb = cfg.cursor_center_index - cfg.cursor_span / 2;
    const int32_t window_ub = cfg.cursor_center_index + cfg.cursor_span / 2;
    if (window_lb < 0)
        throw std::out_of_range("LorenzDatastream::LorenzDatastream - cursor window underruns the stream");
    if (static_cast<size_t>(window_ub) > cfg.stream_length)
        throw std::out_of_range("LorenzDatastream::LorenzDatastream - cursor window overruns the stream");

    Normalize(Build(cfg.stream_length, cfg.initial_lorenz_state, cfg.lorenz_dt));
}

LorenzDatastreamResult LorenzDatastream::States()
{
    auto [past, future] = Indices();
    return {Distance(), data_stream_[past], &data_stream_[future]};
}

NormalizedState LorenzDatastream::NextFutureState() const
{
    auto [past, future] = NextIndices();
    return data_stream_[future];
}

LorenzDatastreamResult LorenzDatastream::Step([[maybe_unused]] const bool useGeneratedFuture)
{
    auto [past, future] = JanusCursor::Step();
    if (past < 0)
        throw std::out_of_range("LorenzDatastream::Step - free-run outran the anchor history");
    return {Distance(), data_stream_[past], OOB() ? nullptr : &data_stream_[future]};
}

std::vector<LorenzAttractor::State> LorenzDatastream::Build(const size_t stream_length,
                                                            const LorenzAttractor::State& initial_lorenz_state,
                                                            const float lorenz_dt) const
{
    // Integrate in full double precision; Normalize() narrows to float storage once.
    LorenzAttractor attractor(initial_lorenz_state);
    std::vector<LorenzAttractor::State> raw;
    raw.reserve(stream_length + 1); // make room for start + one per step
    raw.push_back(initial_lorenz_state);
    for (std::size_t i = 0; i < stream_length; ++i)
        raw.push_back(attractor.step(lorenz_dt)); // drop a breadcrumb where it landed
    return raw;
}

void LorenzDatastream::Normalize(const std::vector<LorenzAttractor::State>& raw)
{
    // Per-channel affine map raw S -> [-1, 1] (JanusCursor.md §4b / Appendix A).
    // x, y straddle zero already, so they carry no offset (scale = largest |excursion|);
    // z sits up at ~+24, so it gets a midpoint offset that drops its DC level onto zero
    // plus a half-range scale. The scale/offset values below are measured from this stream,
    // so any seed / dt yields its own envelope.

    // 1. Scan the raw stream for each channel's extremes.
    double x_min = raw.front().x, x_max = x_min;
    double y_min = raw.front().y, y_max = y_min;
    double z_min = raw.front().z, z_max = z_min;
    for (const LorenzAttractor::State& s : raw)
    {
        x_min = std::min(x_min, s.x);
        x_max = std::max(x_max, s.x);
        y_min = std::min(y_min, s.y);
        y_max = std::max(y_max, s.y);
        z_min = std::min(z_min, s.z);
        z_max = std::max(z_max, s.z);
    }

    // 2. Derive the per-channel scale / offset.
    x_scale_ = std::max(std::abs(x_min), std::abs(x_max)); // symmetric: offset 0
    y_scale_ = std::max(std::abs(y_min), std::abs(y_max)); // symmetric: offset 0
    z_offset_ = (z_max + z_min) / 2.0; // ~+24, the "make it bimodal" shift
    z_scale_ = (z_max - z_min) / 2.0; // half-range

    // A constant channel would give a zero scale; fall back to 1.0 so the map below
    // is a no-op rather than a division by zero (degenerate streams only).
    if (x_scale_ == 0.0) x_scale_ = 1.0;
    if (y_scale_ == 0.0) y_scale_ = 1.0;
    if (z_scale_ == 0.0) z_scale_ = 1.0;

    // 3. Write the normalized [-1, 1] stream once, narrowed to float storage. The
    //    scales/offset stay on the object (as doubles) so consumers can invert the
    //    map: v = scale * v_hat + offset.
    data_stream_.reserve(raw.size());
    for (const LorenzAttractor::State& s : raw)
    {
        data_stream_.push_back({static_cast<float>(s.x / x_scale_),
                                static_cast<float>(s.y / y_scale_),
                                static_cast<float>((s.z - z_offset_) / z_scale_)});
    }
}
