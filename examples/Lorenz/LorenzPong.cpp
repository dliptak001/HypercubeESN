/// @file LorenzPong.cpp
/// @brief Scaffold for the LorenzPong example — build target wired up; the
/// experiment itself is not yet implemented.
///
/// This file exists so `LorenzPong` is a first-class build target alongside
/// `Lorenz` and `LorenzOnline`. Flesh out main() with the actual run.

#include <iostream>
#include "LorenzPong.h"

LorenzPong::LorenzPong(const int32_t span) : lb_(-span / 2), ub_(span / 2)
{
}

LorenzPong::positions_ LorenzPong::BoundedStep()
{
    r_idx_ += r_direction_;

    if (r_idx_ >= ub_)
        r_direction_ = -1;
    else  if (r_idx_ <= lb_)
        r_direction_ = 1;

    return {center_ - r_idx_*dt_, center_ + r_idx_*dt_};
}

LorenzPong::positions_ LorenzPong::UnBoundedStep()
{
    // todo - cache reservoir state
    r_idx_ += r_direction_;
    return {center_ - r_idx_*dt_, center_ + r_idx_*dt_};
}

int main()
{
    std::cout << "=== HypercubeESN: LorenzPong (scaffold) ===\n";
    std::cout << "Not yet implemented.\n";
    return 0;
}
