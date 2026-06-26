#pragma once

#include <cstdint>
#include <memory>
#include "JanusCursor.h"
#include "LorenzAttractor.h"


class LorenzDatastream : public JanusCursor
{
public:
    LorenzDatastream(const int32_t cursor_span,
                     const int32_t cursor_focus_index,
                     const int32_t stream_length,
                     const LorenzAttractor::State& initial_lorenz_state,
                     const float lorenz_dt);

    double GetXScale() const { return x_scale_; };
    double GetYScale() const { return y_scale_; };
    double GetZScale() const { return z_scale_; };
    double GetZOffset() const { return z_offset_; };

private:
    int32_t stream_length_;

    std::unique_ptr<float[], AlignedFree> t_;
    std::unique_ptr<float[], AlignedFree> x_;
    std::unique_ptr<float[], AlignedFree> y_;
    std::unique_ptr<float[], AlignedFree> z_;

    double x_scale_;
    double y_scale_;
    double z_scale_;
    double z_offset_;

    void Build(const LorenzAttractor::State& initial_lorenz_state, const float lorenz_dt);

    void Normalize();
};


#endif //HYPERCUBEESN_DATASTREAM_H
