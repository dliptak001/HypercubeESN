#pragma once

#include "LorenzAttractor.h"
#include <cstdint>
#include <tuple>

class LorenzPong
{
public:
    LorenzPong(int32_t span);
    
    
private:
    LorenzAttractor attractor_;

    int32_t lb_;    // lower bound
    int32_t ub_;    // upper bound
    int32_t r_idx_ = 0;
    int32_t r_direction_ = 1;
    float center_ = 0.0f;
    float dt_ = 0.0f;

    using positions_ = std::tuple<float, float>;
    positions_ BoundedStep();
    positions_ UnBoundedStep();
};