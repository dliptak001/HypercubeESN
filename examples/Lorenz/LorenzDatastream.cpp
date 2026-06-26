#include "LorenzDatastream.h"

#include <stdexcept>

LorenzDatastream::LorenzDatastream(const int32_t cursor_span,
                                   const int32_t cursor_focus_index,
                                   const int32_t stream_length,
                                   const LorenzAttractor::State& initial_lorenz_state,
                                   const float lorenz_dt)
    : JanusCursor(cursor_span, cursor_focus_index), stream_length_(stream_length)
{
    // TODO Claude - lets implement the full set of range validation checks...
    if (cursor_span > stream_length_)
        throw std::out_of_range("LorenzDatastream::LorenzDatastream - expecting span < length");

    t_.reset(AllocAligned(stream_length_));
    x_.reset(AllocAligned(stream_length_));
    y_.reset(AllocAligned(stream_length_));
    z_.reset(AllocAligned(stream_length_));

    Build();

    Normalize();
}

void LorenzDatastream::Build(const LorenzAttractor::State& initial_lorenz_state, const float lorenz_dt)
{
    LorenzAttractor::State state = initial_lorenz_state;
    LorenzAttractor attractor(state);

    for (int32_t i = 0; i < stream_length_; ++i)
    {
        t_[i] = i*lorenz_dt;
        x_[i] = state.x;
        y_[i] = state.y;
        z_[i] = state.z;
        state = attractor.step(lorenz_dt);
    }
}
