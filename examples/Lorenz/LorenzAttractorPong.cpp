/// @file LorenzAttractorPong.cpp
/// @brief This file exists so `LorenzAttractorPong` is a first-class build target
/// alongside `Lorenz` and `LorenzOnline`. Flesh out main() with the actual run.

#include <iostream>
#include "LorenzAttractorPong.h"

#if 0
void LorenzAttractorPong::Eval()
{
    LorenzAttractor::State center_state = {0.1, 0.1, 0.1};
    int32_t span = 10;
    float dt = 0.01;
    LorenzAttractorPong lp(center_state, span, dt);

    for (int i = 0; i < 21; i++)
    {
        lp.BoundedStep();
    }
}
#endif

LorenzAttractorPong::LorenzAttractorPong(const LorenzAttractor::State& center_state, const int32_t span, const double dt)
    : center_state_(center_state), lb_(-span / 2), ub_(span / 2), dt_(dt)
{
}

void LorenzAttractorPong::advance_()
{
    if (r_idx_ == 0)
    {
        // once every cycle, re-anchor attractor_a_ and then re-align attractor_b with attractor_a_
        attractor_a_.reset(center_state_);
        attractor_b_.reset(center_state_);
    }
    else
    {
        const double dt_adj = r_direction_ == 1 ? dt_ : -dt_;
        attractor_a_.step(dt_adj);
        attractor_b_.step(-dt_adj);
    }
}

/// @brief Advance one step of the back-and-forth scan under reflecting boundary
/// conditions: r_idx_ ramps as a triangle wave, reversing direction at lb_/ub_.
void LorenzAttractorPong::BoundedStep()
{
    advance_();

    print();

    if (r_idx_ >= ub_)
        r_direction_ = -1;
    else if (r_idx_ <= lb_)
        r_direction_ = 1;

    r_idx_ += r_direction_;
}

int32_t LorenzAttractorPong::UnBoundedStep()
{
    int32_t retval = 0;
    if (r_idx_ > ub_)
        retval = 1;
    else if (r_idx_ < lb_)
        retval = -1;

    advance_();

    std::cout << "[" << retval << "]: ";
    print();

    r_idx_ += r_direction_;
    return retval;
}

void LorenzAttractorPong::print()
{
    std::cout << r_idx_ * dt_ << ",";
    attractor_a_.state.print();
    std::cout << " || ";
    attractor_b_.state.print();
    std::cout << std::endl;
}


int main()
{
    std::cout << "=== HypercubeESN: LorenzAttractorPong ===\n";
    return 0;
}
