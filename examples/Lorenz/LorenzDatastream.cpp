#include "LorenzDatastream.h"

#include <stdexcept>

LorenzDatastream::LorenzDatastream(const int32_t cursor_span,
                                   const int32_t cursor_focus_index,
                                   const size_t stream_length,
                                   const LorenzAttractor::State& initial_lorenz_state,
                                   const float lorenz_dt)
    : JanusCursor(cursor_span, cursor_focus_index), stream_length_(stream_length)
{
    // TODO Claude - lets implement the full set of range validation checks...
    if (cursor_span > stream_length_)
        throw std::out_of_range("LorenzDatastream::LorenzDatastream - expecting span < length");

    //x_.reset(AllocAligned(stream_length_));
    //y_.reset(AllocAligned(stream_length_));
    //z_.reset(AllocAligned(stream_length_));

    Build(initial_lorenz_state, lorenz_dt);

    Normalize();
}

void LorenzDatastream::Build(const LorenzAttractor::State& initial_lorenz_state, const float lorenz_dt)
{
    LorenzAttractor attractor(initial_lorenz_state);
    points_.reserve(stream_length_ + 1); // make room for start + one per step
    points_.push_back(initial_lorenz_state);
    for (std::size_t i = 0; i < stream_length_; ++i)
        points_.push_back(attractor.step(lorenz_dt)); // drop a breadcrumb where it landed
}
