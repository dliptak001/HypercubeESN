#pragma once

#include <cstddef>
#include <vector>
#include <iostream>

class LorenzAttractor
{
public:
    struct State
    {
        double x = 1.0, y = 1.0, z = 1.0;
        void print() const { std::cout << x << "," << y << "," << z; }
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

    void step(const double dt) { state = peek(state, dt); }

    [[nodiscard]] State peek(const State& s, const double dt) const { return rk4_step(s, dt); }

    [[nodiscard]] std::vector<State> trajectory(const State& initial_state, const std::size_t steps, const double dt) const
    {
        std::vector<State> points;
        points.reserve(steps + 1); // make room for start + one per step

        State cur = initial_state;
        points.push_back(cur); // drop the first breadcrumb (the start)

        for (std::size_t i = 0; i < steps; ++i)
        {
            cur = peek(cur, dt); // move the copy one step
            points.push_back(cur); // drop a breadcrumb where it landed
        }
        return points;
    }

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
