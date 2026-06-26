#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "JanusCursor.h"
#include "LorenzAttractor.h"


class LorenzDatastream : public JanusCursor
{
public:
    LorenzDatastream(int32_t cursor_span,
                     int32_t cursor_focus_index,
                     size_t stream_length,
                     const LorenzAttractor::State& initial_lorenz_state,
                     float lorenz_dt);

    double GetXScale() const { return x_scale_; };
    double GetYScale() const { return y_scale_; };
    double GetZScale() const { return z_scale_; };
    double GetZOffset() const { return z_offset_; };

    const std::vector<LorenzAttractor::State>& GetDataStream() const { return data_stream_; };

    static void Evaluation(); // dev harness: dump the Janus shuttle's cursor indices over several periods (invoked from main.cpp)

private:
    size_t stream_length_;

    std::vector<LorenzAttractor::State> data_stream_;

    double x_scale_ = 1.0;
    double y_scale_ = 1.0;
    double z_scale_ = 1.0;
    double z_offset_ = 0.0;

    void Build(const LorenzAttractor::State& initial_lorenz_state, float lorenz_dt);

    void Normalize();
};
