#pragma once
#include <cstdint>
#include <stdexcept>

/// @brief Single forward index over a stream window.
///
/// Holds no sample data — only an integer index and the training-window geometry
/// that defines it. The owner maps the index into its own buffer (see
/// @ref LorenzDatastream).
///
/// Window [lb, ub] with lb = @p start_index, ub = start_index + span:
///
///     stream:  0 ....... lb ======================= ub ....... N
///                          ^ Reset                    | train end
///              ──────────────────────────────────────>  each Step
///                                                       OOB when index > ub
///
///   - @ref Reset seats at lb (start of the training / washout window).
///   - @ref Step increments by one.
///   - @ref OOB is true once the index has left the training window (generative /
///     eval runway begins). The class does not know stream length N; runway bounds
///     are the owner's responsibility.
class Cursor
{
public:
    /// @param span training-window width (must be > 0). ub = start_index + span.
    /// @param start_index first index of the training window (must be >= 0).
    Cursor(const int32_t span, const int32_t start_index)
        : span_(span), lb_(start_index), ub_(start_index + span)
    {
        if (span <= 0)
            throw std::out_of_range("Cursor::Cursor - expecting span > 0");
        if (start_index < 0)
            throw std::out_of_range("Cursor::Cursor - expecting start_index >= 0");
        Reset();
    }

    void Reset() { idx_ = lb_; }

    /// Advance one step; return the new index.
    int32_t Step() { return idx_ += 1; }

    [[nodiscard]] int32_t Index() const { return idx_; }
    [[nodiscard]] int32_t NextIndex() const { return idx_ + 1; }

    /// True once the index is past the training window upper edge.
    [[nodiscard]] bool OOB() const { return idx_ > ub_; }

    [[nodiscard]] bool AtStartPosition() const { return idx_ == lb_; }

    [[nodiscard]] int32_t Span() const { return span_; }
    [[nodiscard]] int32_t LowerBound() const { return lb_; }
    [[nodiscard]] int32_t UpperBound() const { return ub_; }

private:
    int32_t span_;
    int32_t lb_;
    int32_t ub_;
    int32_t idx_ = 0;
};
