#pragma once

#include <cstdio>

/// Fixed-step RK4 integrator for Lorenz-63 (ground-truth orbit for the example).
///
/// Canonical chaotic parameters (σ = 10, ρ = 28, β = 8/3):
///
///     dx/dt = σ(y − x)
///     dy/dt = x(ρ − z) − y
///     dz/dt = xy − βz
///
/// @ref step advances @ref state by one classical RK4 step of size @p dt.
/// Parameters and state are public knobs. @ref LorenzDatastream integrates an
/// instance and normalizes the orbit for the ESN; the reservoir never sees this
/// type directly.
class LorenzAttractor
{
public:
    struct State
    {
        double x = 1.0, y = 1.0, z = 1.0;
        void print() const { std::printf("%g,%g,%g:   ", x, y, z); }
    };

    double sigma = 10.0;
    double rho = 28.0;
    double beta = 8.0 / 3.0;

    State state;

    LorenzAttractor() = default;
    explicit LorenzAttractor(const State& initial) : state(initial) {}

    /// One RK4 step; mutates and returns @ref state.
    State step(const double dt) { return state = rk4_step(state, dt); }

private:
    [[nodiscard]] State derivatives(const State& s) const
    {
        return {
            sigma * (s.y - s.x),
            s.x * (rho - s.z) - s.y,
            s.x * s.y - beta * s.z
        };
    }

    [[nodiscard]] State rk4_step(const State& s, const double dt) const
    {
        const double h2 = dt * 0.5;
        const auto k1 = derivatives(s);
        const auto k2 = derivatives({s.x + k1.x * h2, s.y + k1.y * h2, s.z + k1.z * h2});
        const auto k3 = derivatives({s.x + k2.x * h2, s.y + k2.y * h2, s.z + k2.z * h2});
        const auto k4 = derivatives({s.x + k3.x * dt, s.y + k3.y * dt, s.z + k3.z * dt});
        const double h6 = dt / 6.0;
        return {
            s.x + (k1.x + 2 * k2.x + 2 * k3.x + k4.x) * h6,
            s.y + (k1.y + 2 * k2.y + 2 * k3.y + k4.y) * h6,
            s.z + (k1.z + 2 * k2.z + 2 * k3.z + k4.z) * h6
        };
    }
};
