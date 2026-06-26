#pragma once
#include <tuple>
#include <cstdint>
#include <stdexcept>

using JanusCursorResult = std::pair<int32_t, int32_t>;

// The Janus Cyclic Orbit ESN Datastream Traversal Cursor class
class JanusCursor
{
    class CyclicOrbitCursor
    {
    public:
        explicit CyclicOrbitCursor(const int32_t span, const int32_t focus_index, const int32_t polarity)
            : focus_index_(focus_index), polarity_(polarity), lb_(focus_index - span / 2), ub_(focus_index + span / 2)
        {
            if (span <= 0)
                throw std::out_of_range("CyclicOrbitCursor::CyclicOrbitCursor - expecting span > 0");
            if (focus_index <= span)
                throw std::out_of_range("CyclicOrbitCursor::CyclicOrbitCursor - expecting focus_index > span");
            if (polarity != -1 && polarity != 1)
                throw std::out_of_range("CyclicOrbitCursor::CyclicOrbitCursor - expecting polarity = +-1");

            Reset(); // start at the center (idx_ = focus), facing this cursor's polarity direction
        }

        void Reset()
        {
            idx_ = focus_index_;
            dir_ = polarity_;
        }

        int32_t StepBounded()
        {
            if (idx_ >= ub_)
                dir_ = -1; // reflect off the upper bound: turn around and head down
            else if (idx_ <= lb_)
                dir_ = 1;  // reflect off the lower bound: turn around and head up
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
        [[nodiscard]] bool AtFocus() const { return idx_ == focus_index_; }

        [[nodiscard]] int32_t index() const { return idx_; }
        [[nodiscard]] int32_t direction() const { return dir_; }

    private:
        int32_t focus_index_;
        int32_t polarity_;
        int32_t lb_, ub_;
        int32_t idx_ = 0;
        int32_t dir_ = 1;
    };

public:
    JanusCursor(const int32_t span, const int32_t focus_index)
        : past_cursor_(span, focus_index, -1), future_cursor_(span, focus_index, 1)
    {
    }

    void Reset()
    {
        past_cursor_.Reset();
        future_cursor_.Reset();
    }

    JanusCursorResult StepBounded()
    {
        const int32_t past_idx = past_cursor_.StepBounded();
        const int32_t future_idx = future_cursor_.StepBounded();
        return {past_idx, future_idx};
    }

    JanusCursorResult StepUnbounded()
    {
        const int32_t past_idx = past_cursor_.StepUnbounded();
        const int32_t future_idx = future_cursor_.StepUnbounded();
        return {past_idx, future_idx};
    }

    [[nodiscard]] JanusCursorResult Indices() const
    {
        return {past_cursor_.index(), future_cursor_.index()};
    }

    [[nodiscard]] int32_t OOB() const { return future_cursor_.OOB(); }

    [[nodiscard]] bool AtFocus() const { return future_cursor_.AtFocus(); }

private:
    CyclicOrbitCursor past_cursor_, future_cursor_;
};
