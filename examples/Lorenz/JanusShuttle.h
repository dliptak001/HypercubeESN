#pragma once
#include "JanusShuttleCursor.h"
#include <cstdint>

/// @brief A function a @ref JanusShuttle can traverse: a value type with a nested
/// `State`, a public `state` member holding the current state, and the two
/// stepping operations the policy needs — `reset(State)` and `step(double dt)`.
template <class T>
concept Steppable = requires(T t, const typename T::State& s, double dt) {
    typename T::State;
    t.reset(s);
    t.step(dt);
    t.state; // public current-state member
};

/// @brief A pair of @ref Steppable copies shuttled back-and-forth along a
/// reflecting 1D scan: from a shared center, one copy runs forward in time and
/// the other backward, re-anchoring together at the seam.
///
/// All index/direction/reflection lives in @ref JanusShuttleCursor; this class
/// applies only the traversal policy: re-anchor both copies at the center on the
/// seam (index 0), otherwise step the forward copy by direction*dt and the
/// backward copy by -direction*dt. The traversed function — its dynamics, its
/// `State` — is fully decoupled: any @ref Steppable plugs in.
template <Steppable Fn>
class JanusShuttle
{
public:
    using State = typename Fn::State;

    JanusShuttle(const State& center, int32_t span, double dt)
        : center_(center), cursor_(span), dt_(dt)
    {
    }

    /// @brief Re-anchor the scan at a new center and reset the scanner to the seam.
    void SetCenter(const State& center)
    {
        center_ = center;
        cursor_.reset();
    }

    /// @brief One reflecting step: apply the policy at the scanner's current
    /// position, then let the scanner reflect-and-advance (triangle wave).
    void BoundedStep()
    {
        advance_();
        cursor_.step();
    }

    /// @brief One non-reflecting step. Reports out-of-bounds crossings: +1 past
    /// the upper bound, -1 past the lower, else 0. Direction never flips, so the
    /// scan ramps one way.
    int32_t UnBoundedStep()
    {
        const int32_t oob = cursor_.out_of_bounds();
        advance_();
        cursor_.advance_oneway();
        return oob;
    }

    /// Current state of the forward copy (runs forward in time from the center).
    [[nodiscard]] const State& forward_state() const { return forward_.state; }
    /// Current state of the backward copy (runs backward in time from the center).
    [[nodiscard]] const State& backward_state() const { return backward_.state; }
    /// Current scan index.
    [[nodiscard]] int32_t index() const { return cursor_.index(); }

private:
    Fn forward_, backward_;
    State center_;
    JanusShuttleCursor cursor_;
    double dt_ = 0.0;

    /// @brief Apply the policy at the scanner's CURRENT position: re-anchor both
    /// copies at the center on the seam, else step forward by +direction*dt and
    /// backward by -direction*dt.
    void advance_()
    {
        if (cursor_.at_seam())
        {
            forward_.reset(center_);
            backward_.reset(center_);
        }
        else
        {
            const double dt_adj = cursor_.direction() == 1 ? dt_ : -dt_;
            forward_.step(dt_adj);
            backward_.step(-dt_adj);
        }
    }
};
