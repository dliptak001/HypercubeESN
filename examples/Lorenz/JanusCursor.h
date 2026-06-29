#pragma once
#include <utility>
#include <cstdint>
#include <stdexcept>

using JanusCursorResult = std::pair<int32_t, int32_t>;

class JanusCursor
{
    class PastCursor
    {
    public:
        explicit PastCursor(const int32_t span, const int32_t center_index)
            : lb_(center_index - span / 2), ub_(center_index + span / 2)
        {
            if (span <= 0)
                throw std::out_of_range("PastCursor::PastCursor - expecting span > 0");
            if (center_index <= span / 2)
                throw std::out_of_range("PastCursor::PastCursor - expecting center_index > span/2");

            Reset();
        }

        void Reset()
        {
            idx_ = ub_;
        }

        int32_t Step()
        {
            return idx_ -= 1;
        }

        [[nodiscard]] bool OOB() const { return idx_ < lb_; }
        [[nodiscard]] int32_t index() const { return idx_; }
        [[nodiscard]] bool AtStartPosition() const { return idx_ == ub_; }

    private:
        int32_t idx_ = 0;
        int32_t lb_, ub_;
    };

    class FutureCursor
    {
    public:
        explicit FutureCursor(const int32_t span, const int32_t center_index)
            : lb_(center_index - span / 2), ub_(center_index + span / 2)
        {
            if (span <= 0)
                throw std::out_of_range("FutureCursor::FutureCursor - expecting span > 0");
            if (center_index <= span / 2)
                throw std::out_of_range("FutureCursor::FutureCursor - expecting center_index > span/2");

            Reset();
        }

        void Reset()
        {
            idx_ = lb_;
        }

        int32_t Step()
        {
            return idx_ += 1;
        }

        [[nodiscard]] bool OOB() const { return idx_ > ub_; }
        [[nodiscard]] int32_t index() const { return idx_; }
        [[nodiscard]] bool AtStartPosition() const { return idx_ == lb_; }

    private:
        int32_t idx_ = 0;
        int32_t lb_, ub_;
    };

public:
    JanusCursor(const int32_t span, const int32_t center_index)
        : span_(span), past_cursor_(span, center_index), future_cursor_(span, center_index)
    {
    }

    void Reset()
    {
        past_cursor_.Reset();
        future_cursor_.Reset();
    }

    JanusCursorResult Step()
    {
        return {past_cursor_.Step(), future_cursor_.Step()};
    }

    [[nodiscard]] JanusCursorResult Indices() const
    {
        return {past_cursor_.index(), future_cursor_.index()};
    }

    [[nodiscard]] bool OOB() const
    {
        return future_cursor_.OOB();
    }

    [[nodiscard]] float Distance() const
    {
        return 1.0f * (future_cursor_.index() - past_cursor_.index()) / span_;
    }

    [[nodiscard]] bool AtStartPosition() const { return past_cursor_.AtStartPosition(); }

private:
    int32_t span_;
    PastCursor past_cursor_;
    FutureCursor future_cursor_;
};
