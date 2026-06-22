#pragma once

#include "LorenzAttractor.h"
#include <cstdint>
#include <tuple>

class LorenzPong
{
public:
    LorenzPong(const LorenzAttractor::State& center_state, int32_t span, float dt);

    void SetCenter(const LorenzAttractor::State& center_state)
    {
        center_state_ = center_state;
        r_idx_ = 0;
        r_direction_ = 1;
    }

    void BoundedStep();
    int32_t UnBoundedStep();
    static void Eval();
    LorenzAttractor attractor_a_;
    LorenzAttractor attractor_b_;

private:

    LorenzAttractor::State center_state_; // center is associated with a State
    int32_t lb_; // lower bound
    int32_t ub_; // upper bound
    int32_t r_idx_ = 0;
    int32_t r_direction_ = 1;
    float dt_ = 0.0f;

    using positions_ = std::tuple<float, float>;
    void print();
};
