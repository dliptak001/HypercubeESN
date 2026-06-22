/// @file LorenzPong.cpp
/// @brief This file exists so `LorenzPong` is a first-class build target alongside
/// `Lorenz` and `LorenzOnline`. Flesh out main() with the actual run.

#include <iostream>
#include "LorenzPong.h"

void LorenzPong::Eval()
{
    LorenzAttractor::State center_state = {0.1, 0.1, 0.1};
    int32_t span = 10;
    float dt = 0.01;
    LorenzPong lp(center_state, span, dt);

    for (int i = 0; i < 21; i++)
    {
        lp.UnBoundedStep();
    }
}

LorenzPong::LorenzPong(const LorenzAttractor::State& center_state, const int32_t span, const float dt)
    : center_state_(center_state), lb_(-span / 2), ub_(span / 2), dt_(dt)
{
}

void LorenzPong::BoundedStep()
{
    if (r_idx_ == 0)
    {
        // once every cycle, re-anchor attractor_a_ and then re-align attractor_b with attractor_a_
        attractor_a_.reset(center_state_);
        attractor_b_.reset(center_state_);
    }
    else
    {
        const float dt_adj_ = r_direction_ == 1 ? dt_ : -dt_;
        attractor_a_.step(dt_adj_);
        attractor_b_.step(-dt_adj_);
    }

    //print();

    // Flip at the turning points *before* advancing, so the step that lands on
    // the boundary index is still taken in the outgoing direction. Checking the
    // bound after the increment flips one index too early, which eats the peak
    // step and makes the cycle return to center two indices early.
    if (r_idx_ >= ub_)
        r_direction_ = -1;
    else if (r_idx_ <= lb_)
        r_direction_ = 1;

    r_idx_ += r_direction_;
}

int32_t LorenzPong::UnBoundedStep()
{
    int32_t retval = 0;
    if (r_idx_ > ub_)
        retval = 1;
    else if (r_idx_ < lb_)
        retval = -1;

    if (r_idx_ == 0)
    {
        // once every cycle, re-anchor attractor_a_ and then re-align attractor_b with attractor_a_
        attractor_a_.reset(center_state_);
        attractor_b_.reset(center_state_);
    }
    else
    {
        const float dt_adj_ = r_direction_ == 1 ? dt_ : -dt_;
        attractor_a_.step(dt_adj_);
        attractor_b_.step(-dt_adj_);
    }

    std::cout << "[" << retval << "]: ";
    print();

    r_idx_ += r_direction_;
    return retval;
}

void LorenzPong::print()
{
    std::cout << r_idx_ * dt_ << ",";
    attractor_a_.state.print();
    std::cout << " || ";
    attractor_b_.state.print();
    std::cout << std::endl;
}


int main()
{
    std::cout << "=== HypercubeESN: LorenzPong (scaffold) ===\n";
    LorenzPong::Eval();
    return 0;
}
