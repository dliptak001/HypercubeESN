#pragma once

#include <iostream>

/// @brief Fixed-step RK4 integrator for the Lorenz-63 system — the ground-truth
/// orbit generator behind the Lorenz example.
///
/// The state (x, y, z) evolves under three coupled ODEs with the canonical
/// chaotic parameters (sigma = 10, rho = 28, beta = 8/3):
///
///     dx/dt = sigma * (y - x)
///     dy/dt = x * (rho - z) - y
///     dz/dt = x * y - beta * z
///
/// Each advance is one classical 4th-order Runge-Kutta step of size @p dt
/// (@ref step mutates and returns the new state; @ref peek is the same math but
/// leaves @ref state untouched — handy for looking one step ahead). The
/// parameters and @ref state are public knobs: set them directly, or use
/// @ref reset to return to a chosen initial condition.
///
///     state ──derivatives──> k1 ─┐
///        │                        ├─ weighted average ──> next state
///        └──> k2 ──> k3 ──> k4 ──┘   (k1 + 2k2 + 2k3 + k4)/6 * dt
///
/// The reservoir never sees this class directly; LorenzDatastream integrates an
/// instance to produce the normalized (x, y, z) stream the ESN trains on.
class LorenzAttractor
{
public:
    struct State
    {
        double x = 1.0, y = 1.0, z = 1.0;
        void print() const { std::cout << x << "," << y << "," << z << ":   "; }
    };

    double sigma = 10.0;
    double rho = 28.0;
    double beta = 8.0 / 3.0;

    State state;

    LorenzAttractor() = default;

    explicit LorenzAttractor(const State& initial) : state(initial)
    {
    }

    void reset() { state = State{}; }
    void reset(const State& initial) { state = initial; }

    State step(const double dt) { return state = peek(state, dt); }

    [[nodiscard]] State peek(const State& s, const double dt) const { return rk4_step(s, dt); }

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
        auto k1 = derivatives(s);
        auto k2 = derivatives({s.x + k1.x * dt / 2, s.y + k1.y * dt / 2, s.z + k1.z * dt / 2});
        auto k3 = derivatives({s.x + k2.x * dt / 2, s.y + k2.y * dt / 2, s.z + k2.z * dt / 2});
        auto k4 = derivatives({s.x + k3.x * dt, s.y + k3.y * dt, s.z + k3.z * dt});

        return {
            s.x + (k1.x + 2 * k2.x + 2 * k3.x + k4.x) * dt / 6,
            s.y + (k1.y + 2 * k2.y + 2 * k3.y + k4.y) * dt / 6,
            s.z + (k1.z + 2 * k2.z + 2 * k3.z + k4.z) * dt / 6
        };
    }
};
