#pragma once

#include "LorenzAttractor.h"
#include <cstdint>

class LorenzAttractorPong
{
public:
#if 0
    static void Eval();
#endif

    LorenzAttractorPong(const LorenzAttractor::State& center_state, int32_t span, double dt);

    void SetCenter(const LorenzAttractor::State& center_state)
    {
        center_state_ = center_state;
        r_idx_ = 0;
        r_direction_ = 1;
    }

    const LorenzAttractor::State& attractor_a_state() const { return attractor_a_.state; }
    const LorenzAttractor::State& attractor_b_state() const { return attractor_b_.state; }

private:
    LorenzAttractor attractor_a_;
    LorenzAttractor attractor_b_;

    LorenzAttractor::State center_state_; // center is associated with a State
    int32_t lb_; // lower bound
    int32_t ub_; // upper bound
    int32_t r_idx_ = 0;
    int32_t r_direction_ = 1;
    double dt_ = 0.0f;

    void BoundedStep();
    int32_t UnBoundedStep();
    void advance_(); // re-anchor at the seam (r_idx_ == 0), else step both attractors
    void print();
};
