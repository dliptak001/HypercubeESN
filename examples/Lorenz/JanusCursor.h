#pragma once
#include <cstdint>

// The Janus Cyclic Orbit Temporal Datastream Traversal Cursor class
class JanusCursor
{
    class CyclicOrbitCursor
    {
    public:
        explicit CyclicOrbitCursor(const int32_t focus, const int32_t span, const int32_t polarity)
            : focus_(focus), polarity_(polarity), lb_(focus - span / 2), ub_(focus + span / 2)
        {
        }

        void Reset()
        {
            idx_ = focus_;
            dir_ = polarity_;
        }

        int32_t StepBounded()
        {
            if (idx_ >= ub_)
                dir_ = -polarity_;
            else if (idx_ <= lb_)
                dir_ = polarity_;
            idx_ += dir_;
            return idx_;
        }

        int32_t StepUnbounded()
        {
            idx_ += dir_;
            return idx_;
        }

        /// @brief Out-of-bounds report: +1 past ub, -1 past lb, else 0.
        [[nodiscard]] int32_t OOB() const { return idx_ > ub_ ? 1 : (idx_ < lb_ ? -1 : 0); }

        /// @brief True at the focus
        [[nodiscard]] bool AtFocus() const { return idx_ == focus_; }

        [[nodiscard]] int32_t index() const { return idx_; }
        [[nodiscard]] int32_t direction() const { return dir_; }

    private:
        int32_t focus_;
        int32_t polarity_;
        int32_t lb_, ub_;
        int32_t idx_ = 0;
        int32_t dir_ = 1;
    };

public:
    JanusCursor(const int32_t focus, const int32_t span)
        : past_cursor_(focus, span, -1), future_cursor_(focus, span, 1)
    {
    }

    void Reset()
    {
        past_cursor_.Reset();
        future_cursor_.Reset();
    }

    void StepBounded()  // TODO - return a tuple
    {
        const int32_t past_idx = past_cursor_.StepBounded();
        const int32_t future_idx = future_cursor_.StepBounded();
    }

    int32_t StepUnbounded()  // TODO - return a tuple
    {
        const int32_t past_idx = past_cursor_.StepUnbounded();
        const int32_t future_idx = future_cursor_.StepUnbounded();
    }

    [[nodiscard]] int32_t OOB() const { return past_cursor_.OOB(); }

    [[nodiscard]] bool AtFocus() const { return past_cursor_.AtFocus(); }

private:
    CyclicOrbitCursor past_cursor_, future_cursor_;
};
