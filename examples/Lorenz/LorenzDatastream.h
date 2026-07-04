#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "JanusCursor.h"
#include "LorenzAttractor.h"


// A normalized [-1, 1] stream sample, stored as float: the reservoir consumes
// floats, so the stream is narrowed once at Normalize() rather than per step
// (and the hot window occupies half the cache of double storage).
struct NormalizedState
{
    float x, y, z;
};

using LorenzDatastreamResult = std::tuple<float, NormalizedState, const NormalizedState*>;

struct LorenzDatastreamConfig
{
    int32_t cursor_span = 0;
    int32_t cursor_center_index = 0;
    size_t stream_length = 0;
    LorenzAttractor::State initial_lorenz_state = {0.5, 0.5, 0.5};
    float lorenz_dt = 0.02;
};

class LorenzDatastream : public JanusCursor
{
public:
    LorenzDatastream(const LorenzDatastreamConfig& cfg);

    [[nodiscard]] LorenzDatastreamResult States();
    [[nodiscard]] NormalizedState NextFutureState() const;
    LorenzDatastreamResult Step(bool useGeneratedFuture);

    [[nodiscard]] double GetXScale() const { return x_scale_; }
    [[nodiscard]] double GetYScale() const { return y_scale_; }
    [[nodiscard]] double GetZScale() const { return z_scale_; }
    [[nodiscard]] double GetZOffset() const { return z_offset_; }

    [[nodiscard]] const std::vector<NormalizedState>& GetDataStream() const { return data_stream_; }

private:
    std::vector<NormalizedState> data_stream_;

    double x_scale_ = 1.0;
    double y_scale_ = 1.0;
    double z_scale_ = 1.0;
    double z_offset_ = 0.0;

    [[nodiscard]] std::vector<LorenzAttractor::State> Build(size_t stream_length,
                                                            const LorenzAttractor::State& initial_lorenz_state,
                                                            float lorenz_dt) const;

    void Normalize(const std::vector<LorenzAttractor::State>& raw);
};
