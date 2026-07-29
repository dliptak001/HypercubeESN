#pragma once
#include <cstdint>
#include <stdexcept>

/// @brief Single forward index over a stream window.
///
/// Holds no sample data — only an integer index and the training-window geometry
/// that defines it. The owner maps the index into its own buffer (see
/// @ref LorenzDatastream).
///
/// Training window [0, span] inclusive (lb always 0):
///
///     stream:  0 ======================= span ....... N
///              ^ Reset                    | train end
///              ──────────────────────────>  each Step
///                                           OOB when index > span
///
///   - @ref Reset seats at 0.
///   - @ref Seek seats at an arbitrary index (owner validates vs stream length).
///   - @ref Step increments by one.
///   - @ref OOB is true once the index has left the training window (generative /
///     eval runway begins). The class does not know stream length N; runway bounds
///     are the owner's responsibility.
class Cursor
{
public:
    /// @param span last in-window train index (must be > 0). Window is [0, span].
    explicit Cursor(const int32_t span)
        : span_(span)
    {
        if (span <= 0)
            throw std::out_of_range("Cursor::Cursor - expecting span > 0");
        Reset();
    }

    void Reset() { idx_ = 0; }

    /// Seat at @p index. Does not validate against stream length (owner's job).
    /// @throws std::out_of_range if @p index < 0.
    void Seek(const int32_t index)
    {
        if (index < 0)
            throw std::out_of_range("Cursor::Seek - index must be >= 0");
        idx_ = index;
    }

    /// Advance one step; return the new index.
    int32_t Step() { return idx_ += 1; }

    [[nodiscard]] int32_t Index() const { return idx_; }
    [[nodiscard]] int32_t NextIndex() const { return idx_ + 1; }

    /// True once the index is past the training window.
    [[nodiscard]] bool OOB() const { return idx_ > span_; }

    [[nodiscard]] bool AtStartPosition() const { return idx_ == 0; }

    [[nodiscard]] int32_t Span() const { return span_; }

private:
    int32_t span_;
    int32_t idx_ = 0;
};
