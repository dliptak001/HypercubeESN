#include "LorenzDatastream.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

LorenzDatastream::LorenzDatastream(const LorenzDatastreamConfig& cfg, bool print_header)
    : Cursor(cfg.cursor_span, cfg.cursor_start_index)
{
    if (cfg.stream_length == 0)
        throw std::out_of_range("LorenzDatastream::LorenzDatastream - expecting stream_length > 0");

    if (cfg.lorenz_dt <= 0.0f)
        throw std::out_of_range("LorenzDatastream::LorenzDatastream - expecting lorenz_dt > 0");

    cfg_ = cfg;
    const int32_t window_lb = cfg.cursor_start_index;
    const int32_t window_ub = cfg.cursor_start_index + cfg.cursor_span;
    if (window_lb < 0)
        throw std::out_of_range("LorenzDatastream::LorenzDatastream - cursor window underruns the stream");
    if (static_cast<size_t>(window_ub) > cfg.stream_length)
        throw std::out_of_range("LorenzDatastream::LorenzDatastream - cursor window overruns the stream");

    Normalize(Build(cfg.stream_length, cfg.initial_lorenz_state, cfg.lorenz_dt));

    const size_t N = cfg.stream_length;
    const size_t E = N > static_cast<size_t>(window_ub)
                         ? N - static_cast<size_t>(window_ub)
                         : 0;
    const auto dots = [](const long long v) {
        char buf[16];
        std::snprintf(buf, sizeof buf, "%lld", v);
        std::string cell(buf);
        const size_t pad = cell.size() < 14 ? 14 - cell.size() : 0;
        return std::string(pad, '.') + cell;
    };

    if (print_header)
    {
        std::printf("[LorenzDatastream] %zu+1 samples  dt=%.3f  start=%d  span=%d  window=[%d, %d]\n",
                    N, cfg.lorenz_dt, cfg.cursor_start_index, cfg.cursor_span, window_lb, window_ub);
        std::printf("  array index n:%14d%s%s%s\n", 0, dots(window_lb).c_str(),
                    dots(window_ub).c_str(), dots(static_cast<long long>(N)).c_str());
        std::printf("%16s%14s%14s%14s%14s\n", "", "|", "|", "|", "|");
        std::printf("%16s%14s%14s%14s%14s\n", "", "seed", "train start", "train end", "stream end");
        std::printf("%16s%14s%14s%14s%14s\n", "", "T=0", "(lb)", "(ub)", "(ub+E)");
        std::printf("  region [%d, %d] = training / washout window (span %d)\n",
                    window_lb, window_ub, cfg.cursor_span);
        std::printf("  region (%d, %zu] = free-run / evaluation runway (E = %zu)\n",
                    window_ub, N, E);
        std::printf("%16s generative: input drive is the model's own prediction\n", "");
    }
}

const NormalizedState* LorenzDatastream::SampleAt(const int32_t index) const
{
    if (index < 0 || static_cast<size_t>(index) >= data_stream_.size())
        return nullptr;
    return &data_stream_[static_cast<size_t>(index)];
}

LorenzDatastreamResult LorenzDatastream::States()
{
    const int32_t i = Index();
    return {i, SampleAt(i)};
}

LorenzDatastreamResult LorenzDatastream::Step()
{
    const int32_t i = Cursor::Step();
    return {i, SampleAt(i)};
}

void LorenzDatastream::PrintOrbit()
{
    cfg_.initial_lorenz_state.print();
}

std::vector<LorenzAttractor::State> LorenzDatastream::Build(const size_t stream_length,
                                                            const LorenzAttractor::State& initial_lorenz_state,
                                                            const float lorenz_dt) const
{
    LorenzAttractor attractor(initial_lorenz_state);
    std::vector<LorenzAttractor::State> raw;
    raw.reserve(stream_length + 1);
    raw.push_back(initial_lorenz_state);
    for (std::size_t i = 0; i < stream_length; ++i)
        raw.push_back(attractor.step(lorenz_dt));
    return raw;
}

void LorenzDatastream::Normalize(const std::vector<LorenzAttractor::State>& raw)
{
    // Per-channel midpoint offset; one shared scale (relative amplitudes preserved).
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

    const double x_offset = (x_max + x_min) / 2.0;
    const double y_offset = (y_max + y_min) / 2.0;
    const double z_offset = (z_max + z_min) / 2.0;

    double scale = std::max({(x_max - x_min) / 2.0,
                             (y_max - y_min) / 2.0,
                             (z_max - z_min) / 2.0});
    if (scale == 0.0) scale = 1.0;

    data_stream_.reserve(raw.size());
    for (const LorenzAttractor::State& s : raw)
    {
        data_stream_.push_back({static_cast<float>((s.x - x_offset) / scale),
                                static_cast<float>((s.y - y_offset) / scale),
                                static_cast<float>((s.z - z_offset) / scale)});
    }
}
