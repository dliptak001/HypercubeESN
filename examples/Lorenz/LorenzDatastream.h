#pragma once

#include <cstdint>
#include <memory>
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

    int32_t GetStreamLength() const { return stream_length_; };
    const float* GetX() const { return x_.get(); };
    const float* GetY() const { return y_.get(); };
    const float* GetZ() const { return z_.get(); };

private:
    size_t stream_length_;

    std::unique_ptr<float[], AlignedFree> x_;
    std::unique_ptr<float[], AlignedFree> y_;
    std::unique_ptr<float[], AlignedFree> z_;

    std::vector<LorenzAttractor::State> points_;

    double x_scale_;
    double y_scale_;
    double z_scale_;
    double z_offset_;

    void Build(const LorenzAttractor::State& initial_lorenz_state, float lorenz_dt);

    void Normalize();
};


#endif //HYPERCUBEESN_DATASTREAM_H
