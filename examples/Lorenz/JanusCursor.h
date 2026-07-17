#pragma once
#include <utility>
#include <cstdint>
#include <stdexcept>

using JanusCursorResult = std::pair<int32_t, int32_t>;

/// @brief A pair of counter-moving cursors over one stream that half-anchor the
/// Janus free-run: a past cursor reading real history and a future cursor tracking
/// the prediction horizon.
///
/// Both cursors share a window [lb, ub] centered at @p center_index with the given
/// @p span (lb = center - span/2, ub = center + span/2). @ref Reset seats them at
/// OPPOSITE edges, and each @ref Step walks them in OPPOSITE directions so they sweep
/// past each other:
///
///     stream:  0 ....... lb =========== center =========== ub ....... N
///                         ^future                          ^past       (after Reset)
///              future ──────────────────>          <────────────── past  (each Step)
///
///   - PastCursor:   starts at ub, Step decrements. It may legitimately run BELOW lb
///                   into the anchor runway [0, lb) — that is real history, not an error.
///   - FutureCursor: starts at lb, Step increments. Passing ub means it has run out of
///                   in-window data (the generative tail begins) — this is what @ref OOB
///                   reports. @ref OOB deliberately tracks ONLY the future cursor.
///
/// @ref Distance returns the signed cursor separation normalized by span: -1 at Reset
/// (fully apart), 0 as they cross at center, +1 at the mirror extreme. The class holds
/// no stream data — it hands back plain int32_t indices the owner uses to address its
/// own buffer (see @ref LorenzDatastream).
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
        [[nodiscard]] int32_t next_index() const { return idx_ - 1; }
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
        [[nodiscard]] int32_t next_index() const { return idx_ + 1; }
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

    [[nodiscard]] JanusCursorResult NextIndices() const
    {
        return {past_cursor_.next_index(), future_cursor_.next_index()};
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
