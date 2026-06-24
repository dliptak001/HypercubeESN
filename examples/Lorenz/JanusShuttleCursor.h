#pragma once
#include <cstdint>

/// @brief A 1D integer cursor that shuttles across the inclusive range [lb, ub].
///
/// Pure navigation mechanism: owns position and direction only — it knows
/// nothing about what is being traversed (no dt, no state, no function). Two
/// move primitives let a caller pick the behavior:
///   - @ref step:           shuttling move (triangle wave; reverses at the ends,
///                          never leaves [lb, ub]).
///   - @ref advance_oneway: non-reflecting move (direction never flips; the index
///                          may leave [lb, ub] — pair with @ref out_of_bounds).
/// Plus queries @ref at_seam (index == 0) and @ref out_of_bounds for callers that
/// react to the position. Starts at index 0, direction +1.
class JanusShuttleCursor
{
public:
    /// @brief Scan a symmetric span: lb = -span/2, ub = +span/2.
    explicit JanusShuttleCursor(const int32_t span) : lb_(-span / 2), ub_(span / 2) {}

    /// @brief Scan an explicit inclusive range [lb, ub].
    JanusShuttleCursor(const int32_t lb, const int32_t ub) : lb_(lb), ub_(ub) {}

    /// @brief Reset to index 0, direction +1.
    void reset()
    {
        idx_ = 0;
        dir_ = 1;
    }

    /// @brief Shuttling advance: reverse direction at either end, then step one
    /// unit. Never leaves [lb, ub] (triangle wave). Returns the new index.
    int32_t step()
    {
        if (idx_ >= ub_)
            dir_ = -1;
        else if (idx_ <= lb_)
            dir_ = 1;
        idx_ += dir_;
        return idx_;
    }

    /// @brief Non-reflecting advance: step one unit without flipping direction.
    /// The index may leave [lb, ub]; pair with @ref out_of_bounds. Returns the
    /// new index.
    int32_t advance_oneway()
    {
        idx_ += dir_;
        return idx_;
    }

    /// @brief Out-of-bounds report: +1 past ub, -1 past lb, else 0.
    [[nodiscard]] int32_t out_of_bounds() const { return idx_ > ub_ ? 1 : (idx_ < lb_ ? -1 : 0); }

    /// @brief True at the seam (index == 0) — e.g. a caller's re-anchor trigger.
    [[nodiscard]] bool at_seam() const { return idx_ == 0; }

    [[nodiscard]] int32_t index() const { return idx_; }
    [[nodiscard]] int32_t direction() const { return dir_; }

private:
    int32_t lb_, ub_;
    int32_t idx_ = 0;
    int32_t dir_ = 1;
};
