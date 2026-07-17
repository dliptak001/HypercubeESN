#include "LorenzDatastream.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

LorenzDatastream::LorenzDatastream(const LorenzDatastreamConfig& cfg, bool print_header)
    : JanusCursor(cfg.cursor_span, cfg.cursor_center_index)
{
    if (cfg.stream_length == 0)
        throw std::out_of_range("LorenzDatastream::LorenzDatastream - expecting stream_length > 0");

    if (cfg.lorenz_dt <= 0.0f)
        throw std::out_of_range("LorenzDatastream::LorenzDatastream - expecting lorenz_dt > 0");

    cfg_ = cfg;
    const int32_t window_lb = cfg.cursor_center_index - cfg.cursor_span / 2;
    const int32_t window_ub = cfg.cursor_center_index + cfg.cursor_span / 2;
    if (window_lb < 0)
        throw std::out_of_range("LorenzDatastream::LorenzDatastream - cursor window underruns the stream");
    if (static_cast<size_t>(window_ub) > cfg.stream_length)
        throw std::out_of_range("LorenzDatastream::LorenzDatastream - cursor window overruns the stream");

    Normalize(Build(cfg.stream_length, cfg.initial_lorenz_state, cfg.lorenz_dt));

    // Construction banner: this run's stream/window geometry (JanusCursor.md §1).
    // Fixed-width cells keep the columns aligned for any config values.
    const int32_t H = cfg.cursor_span / 2;
    const size_t N = cfg.stream_length;
    const size_t E = N - static_cast<size_t>(window_ub); // generative/eval runway length
    const auto dots = [](const long long v) {
        char buf[16];
        std::snprintf(buf, sizeof buf, "%lld", v);
        std::string cell(buf);
        return std::string(14 - cell.size(), '.') + cell;
    };

    if (print_header)
    {
        std::printf("[LorenzDatastream] %zu+1 samples  dt=%.3f  center=%d  H=%d  window=[%d, %d]\n",
                    N, cfg.lorenz_dt, cfg.cursor_center_index, H, window_lb, window_ub);
        std::printf("  array index n:%14d%s%s%s%s\n", 0, dots(window_lb).c_str(),
                    dots(cfg.cursor_center_index).c_str(), dots(window_ub).c_str(),
                    dots(static_cast<long long>(N)).c_str());
        std::printf("%16s%14s%14s%14s%14s%14s\n", "", "|", "|", "|", "|", "|");
        std::printf("%16s%14s%14s%14s%14s%14s\n", "", "seed", "train edge", "anchor pt", "train edge", "stream end");
        std::printf("%16s%14s%14s%14s%14s%14s\n", "", "T=0", "(lb)", "(center)", "(ub)", "(ub+E)");
        std::printf("  region [0, %d) = past free-run runway (anchor history for the past cursor)\n", window_lb);
        std::printf("  region [%d, %d] = training window (span %d)\n", window_lb, window_ub, cfg.cursor_span);
        std::printf("  region (%d, %zu] = prediction / evaluation runway (E = %zu) - the future cursor\n",
                    window_ub, N, E);
        std::printf("%16s goes GENERATIVE here: future input channels come from the ensemble's own output\n", "");
    }
}

LorenzDatastreamResult LorenzDatastream::States()
{
    auto [past, future] = Indices();
    return {Distance(), data_stream_[past], &data_stream_[future]};
}

LorenzDatastreamResult LorenzDatastream::Step()
{
    auto [past, future] = JanusCursor::Step();
    if (past < 0)
        throw std::out_of_range("LorenzDatastream::Step - free-run outran the anchor history");
    return {Distance(), data_stream_[past], OOB() ? nullptr : &data_stream_[future]};
}

void LorenzDatastream::PrintOrbit()
{
    cfg_.initial_lorenz_state.print();
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
    // Affine map raw S -> [-1, 1] with a PER-CHANNEL offset but a SINGLE SHARED scale
    // (JanusCursor.md §4b / Appendix A). Each channel is first centered on its own
    // midpoint (x, y straddle zero already; z sits up at ~+24 and gets shifted down),
    // then all three are divided by ONE scale = the widest channel's half-range. Sharing
    // the scale preserves the attractor's true relative amplitudes: the reservoir sees x
    // as genuinely narrower than y rather than every axis stretched to fill [-1, 1]. The
    // widest channel just reaches +-1; the others use proportionally less of the range.
    // The extremes are measured from this stream, so any seed / dt yields its own envelope.

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

    // 2. Per-channel midpoint offsets drop each channel's DC level onto zero...
    const double x_offset = (x_max + x_min) / 2.0;
    const double y_offset = (y_max + y_min) / 2.0;
    const double z_offset = (z_max + z_min) / 2.0; // ~+24, the "make it bimodal" shift

    // ...and one shared scale = the widest half-range, so the same unit is applied to
    // all three channels. A degenerate (constant) stream gives a zero scale; fall back
    // to 1.0 so the map below is a no-op rather than a division by zero.
    double scale = std::max({(x_max - x_min) / 2.0,
                             (y_max - y_min) / 2.0,
                             (z_max - z_min) / 2.0});
    if (scale == 0.0) scale = 1.0;

    // 3. Write the normalized [-1, 1] stream once, narrowed to float storage.
    data_stream_.reserve(raw.size());
    for (const LorenzAttractor::State& s : raw)
    {
        data_stream_.push_back({static_cast<float>((s.x - x_offset) / scale),
                                static_cast<float>((s.y - y_offset) / scale),
                                static_cast<float>((s.z - z_offset) / scale)});
    }
}
