#include "LorenzDatastream.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

LorenzDatastream::LorenzDatastream(const LorenzDatastreamConfig& cfg, bool print_header)
    : Cursor(cfg.span)
    , seed_state_(cfg.initial_lorenz_state)
{
    if (cfg.stream_length == 0)
        throw std::out_of_range("LorenzDatastream: stream_length must be > 0");
    if (cfg.lorenz_dt <= 0.0f)
        throw std::out_of_range("LorenzDatastream: lorenz_dt must be > 0");
    // Last train index must leave room on the integrated stream (indices 0..stream_length).
    if (static_cast<size_t>(cfg.span) > cfg.stream_length)
        throw std::out_of_range("LorenzDatastream: span exceeds stream_length");

    Normalize(Build(cfg.stream_length, cfg.initial_lorenz_state, cfg.lorenz_dt));

    if (print_header)
    {
        const int32_t span = Span();
        const size_t N = cfg.stream_length;
        const size_t E = N - static_cast<size_t>(span);
        const auto dots = [](long long v) {
            char buf[16];
            std::snprintf(buf, sizeof buf, "%lld", v);
            std::string cell(buf);
            const size_t pad = cell.size() < 14 ? 14 - cell.size() : 0;
            return std::string(pad, '.') + cell;
        };

        std::printf("[LorenzDatastream] %zu+1 samples  dt=%.3f  span=%d  window=[0, %d]\n",
                    N, cfg.lorenz_dt, span, span);
        std::printf("  array index n:%14d%s%s\n", 0, dots(span).c_str(),
                    dots(static_cast<long long>(N)).c_str());
        std::printf("%16s%14s%14s%14s\n", "", "|", "|", "|");
        std::printf("%16s%14s%14s%14s\n", "", "seed", "train end", "stream end");
        std::printf("%16s%14s%14s%14s\n", "", "T=0", "(span)", "(span+E)");
        std::printf("  region [0, %d] = train section\n", span);
        std::printf("  region (%d, %zu] = free-run runway (E = %zu)\n", span, N, E);
    }
}

const NormalizedState* LorenzDatastream::SampleAt(const int32_t index) const
{
    if (index < 0 || static_cast<size_t>(index) >= data_stream_.size())
        return nullptr;
    return &data_stream_[static_cast<size_t>(index)];
}

LorenzDatastreamResult LorenzDatastream::States() const
{
    const int32_t i = Index();
    return {i, SampleAt(i)};
}

LorenzDatastreamResult LorenzDatastream::Step()
{
    const int32_t i = Cursor::Step();
    return {i, SampleAt(i)};
}

void LorenzDatastream::PrintOrbit() const
{
    seed_state_.print();
}

std::vector<LorenzAttractor::State> LorenzDatastream::Build(const size_t stream_length,
                                                            const LorenzAttractor::State& seed,
                                                            const float dt)
{
    LorenzAttractor attractor(seed);
    std::vector<LorenzAttractor::State> raw(stream_length + 1);
    raw[0] = seed;
    for (size_t i = 0; i < stream_length; ++i)
        raw[i + 1] = attractor.step(dt);
    return raw;
}

void LorenzDatastream::Normalize(const std::vector<LorenzAttractor::State>& raw)
{
    // Per-channel midpoint; one shared scale (relative amplitudes preserved).
    double x_min = raw.front().x, x_max = x_min;
    double y_min = raw.front().y, y_max = y_min;
    double z_min = raw.front().z, z_max = z_min;
    for (const auto& s : raw)
    {
        x_min = std::min(x_min, s.x);
        x_max = std::max(x_max, s.x);
        y_min = std::min(y_min, s.y);
        y_max = std::max(y_max, s.y);
        z_min = std::min(z_min, s.z);
        z_max = std::max(z_max, s.z);
    }

    const double x_off = (x_max + x_min) * 0.5;
    const double y_off = (y_max + y_min) * 0.5;
    const double z_off = (z_max + z_min) * 0.5;
    double scale = std::max({(x_max - x_min) * 0.5,
                             (y_max - y_min) * 0.5,
                             (z_max - z_min) * 0.5});
    if (scale == 0.0)
        scale = 1.0;
    const double inv = 1.0 / scale;

    data_stream_.resize(raw.size());
    for (size_t i = 0; i < raw.size(); ++i)
    {
        const auto& s = raw[i];
        data_stream_[i] = {
            static_cast<float>((s.x - x_off) * inv),
            static_cast<float>((s.y - y_off) * inv),
            static_cast<float>((s.z - z_off) * inv)};
    }
}
