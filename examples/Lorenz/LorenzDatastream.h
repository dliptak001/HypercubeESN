#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "JanusCursor.h"
#include "LorenzAttractor.h"


using LorenzDatastreamResult = std::tuple<float, LorenzAttractor::State, const LorenzAttractor::State*>;

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

    [[nodiscard]] LorenzDatastreamResult PeekStates();
    [[nodiscard]] LorenzDatastreamResult PeekNextStates();
    LorenzDatastreamResult Step(bool useGeneratedFuture);

    [[nodiscard]] double GetXScale() const { return x_scale_; }
    [[nodiscard]] double GetYScale() const { return y_scale_; }
    [[nodiscard]] double GetZScale() const { return z_scale_; }
    [[nodiscard]] double GetZOffset() const { return z_offset_; }

    [[nodiscard]] const std::vector<LorenzAttractor::State>& GetDataStream() const { return data_stream_; }

    static void Evaluation(); // dev harness

private:
    std::vector<LorenzAttractor::State> data_stream_;

    double x_scale_ = 1.0;
    double y_scale_ = 1.0;
    double z_scale_ = 1.0;
    double z_offset_ = 0.0;

    void Build(size_t stream_length, const LorenzAttractor::State& initial_lorenz_state, float lorenz_dt);

    void Normalize();
};
