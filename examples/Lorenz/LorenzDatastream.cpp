#include "LorenzDatastream.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

LorenzDatastream::LorenzDatastream(const int32_t cursor_span,
                                   const int32_t cursor_focus_index,
                                   const size_t stream_length,
                                   const LorenzAttractor::State& initial_lorenz_state,
                                   const float lorenz_dt)
    : JanusCursor(cursor_span, cursor_focus_index), stream_length_(stream_length)
{
    if (stream_length_ == 0)
        throw std::out_of_range("LorenzDatastream::LorenzDatastream - expecting stream_length > 0");

    if (lorenz_dt <= 0.0f)
        throw std::out_of_range("LorenzDatastream::LorenzDatastream - expecting lorenz_dt > 0");

    const int32_t window_lb = cursor_focus_index - cursor_span / 2;
    const int32_t window_ub = cursor_focus_index + cursor_span / 2;
    if (window_lb < 0)
        throw std::out_of_range("LorenzDatastream::LorenzDatastream - cursor window underruns the stream");
    if (static_cast<size_t>(window_ub) >= stream_length_)
        throw std::out_of_range("LorenzDatastream::LorenzDatastream - cursor window overruns the stream");

    Build(initial_lorenz_state, lorenz_dt);

    Normalize();
}

void LorenzDatastream::Build(const LorenzAttractor::State& initial_lorenz_state, const float lorenz_dt)
{
    LorenzAttractor attractor(initial_lorenz_state);
    data_stream_.reserve(stream_length_ + 1); // make room for start + one per step
    data_stream_.push_back(initial_lorenz_state);
    for (std::size_t i = 0; i < stream_length_; ++i)
        data_stream_.push_back(attractor.step(lorenz_dt)); // drop a breadcrumb where it landed
}

void LorenzDatastream::Normalize()
{
    // Per-channel affine map raw S -> [-1, 1] (JanusCursor.md §4b / Appendix A).
    // x, y straddle zero already, so they carry no offset (scale = largest |excursion|);
    // z sits up at ~+24, so it gets a midpoint offset that drops its DC level onto zero
    // plus a half-range scale. The eight numbers below are NOT hardcoded — they are
    // measured from this stream, so any seed / dt yields its own envelope.

    // 1. Scan the stream for each channel's extremes.
    double x_min = data_stream_.front().x, x_max = x_min;
    double y_min = data_stream_.front().y, y_max = y_min;
    double z_min = data_stream_.front().z, z_max = z_min;
    for (const LorenzAttractor::State& s : data_stream_)
    {
        x_min = std::min(x_min, s.x); x_max = std::max(x_max, s.x);
        y_min = std::min(y_min, s.y); y_max = std::max(y_max, s.y);
        z_min = std::min(z_min, s.z); z_max = std::max(z_max, s.z);
    }

    // 2. Derive the per-channel scale / offset.
    x_scale_  = std::max(std::abs(x_min), std::abs(x_max)); // symmetric: offset 0
    y_scale_  = std::max(std::abs(y_min), std::abs(y_max)); // symmetric: offset 0
    z_offset_ = (z_max + z_min) / 2.0;                      // ~+24, the "make it bimodal" shift
    z_scale_  = (z_max - z_min) / 2.0;                      // half-range

    // A constant channel would give a zero scale; fall back to 1.0 so the map below
    // is a no-op rather than a division by zero (degenerate streams only).
    if (x_scale_ == 0.0) x_scale_ = 1.0;
    if (y_scale_ == 0.0) y_scale_ = 1.0;
    if (z_scale_ == 0.0) z_scale_ = 1.0;

    // 3. Rewrite the stream in place to its normalized [-1, 1] values. The scales/offset
    //    stay on the object so consumers can invert the map: v = scale * v_hat + offset.
    for (LorenzAttractor::State& s : data_stream_)
    {
        s.x = s.x / x_scale_;
        s.y = s.y / y_scale_;
        s.z = (s.z - z_offset_) / z_scale_;
    }
}

void LorenzDatastream::Evaluation()
{
    // Dev harness for the Janus shuttle: drive the bounded reflecting scan over a few
    // periods and print the two cursor indices side by side. The backward (S[N_c - i])
    // and forward (S[N_c + i]) indices mirror about the focus and stay inside [lb, ub].
    constexpr int32_t span   = 8;
    constexpr int32_t focus  = 12;   // must exceed span (enforced by the JanusCursor base)
    constexpr size_t  length = 25;   // must exceed focus + span/2 (the cursor window's top)
    constexpr float   dt     = 0.02f;

    LorenzDatastream stream(span, focus, length, LorenzAttractor::State{}, dt);

    const int32_t H      = span / 2;
    const int32_t lb     = focus - H;
    const int32_t ub     = focus + H;
    const int32_t period = 4 * H;    // one full triangle-wave period: ctr->ub->lb->ctr
    constexpr int periods = 3;

    std::printf("Janus shuttle  span=%d  focus=%d  window=[%d, %d]\n", span, focus, lb, ub);
    std::printf("  step  backward  forward\n");
    for (int k = 0; k < periods * period; ++k)
    {
        const auto [backward, forward] = stream.StepBounded();
        std::printf("  %4d  %8d  %7d\n", k, backward, forward);
    }
}
