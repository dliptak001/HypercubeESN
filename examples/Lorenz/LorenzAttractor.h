#pragma once
#include <cmath>
#include <vector>
#include <cstddef>

class LorenzAttractor {
public:
    struct State {
        double x = 1.0, y = 1.0, z = 1.0;
    };

    double sigma = 10.0;
    double rho   = 28.0;
    double beta  = 8.0 / 3.0;

    State state;

    LorenzAttractor() = default;
    explicit LorenzAttractor(State initial) : state(initial) {}

    void reset()              { state = State{}; }
    void reset(State initial) { state = initial; }

    // Mutating integrators
    void step(double dt) { state = advance(state, dt, false); }
    void rk4 (double dt) { state = advance(state, dt, true); }

    // Const, non-mutating advance (reusable)
    State advance(State s, double dt, bool use_rk4 = true) const {
        return use_rk4 ? rk4_step(s, dt) : euler_step(s, dt);
    }

    // Generate trajectory without modifying internal state
    std::vector<State> trajectory(size_t steps, double dt, bool use_rk4 = true) const {
        std::vector<State> points;
        points.reserve(steps + 1);

        State cur = state;
        points.push_back(cur);

        for (size_t i = 0; i < steps; ++i) {
            cur = advance(cur, dt, use_rk4);
            points.push_back(cur);
        }
        return points;
    }

private:
    State derivatives(const State& s) const {
        return {
            sigma * (s.y - s.x),
            s.x * (rho - s.z) - s.y,
            s.x * s.y - beta * s.z
        };
    }

    State euler_step(const State& s, double dt) const {
        State d = derivatives(s);
        return {s.x + d.x * dt, s.y + d.y * dt, s.z + d.z * dt};
    }

    State rk4_step(const State& s, double dt) const {
        auto k1 = derivatives(s);
        auto k2 = derivatives({s.x + k1.x*dt/2, s.y + k1.y*dt/2, s.z + k1.z*dt/2});
        auto k3 = derivatives({s.x + k2.x*dt/2, s.y + k2.y*dt/2, s.z + k2.z*dt/2});
        auto k4 = derivatives({s.x + k3.x*dt,   s.y + k3.y*dt,   s.z + k3.z*dt});

        return {
            s.x + (k1.x + 2*k2.x + 2*k3.x + k4.x) * dt / 6,
            s.y + (k1.y + 2*k2.y + 2*k3.y + k4.y) * dt / 6,
            s.z + (k1.z + 2*k2.z + 2*k3.z + k4.z) * dt / 6
        };
    }
};