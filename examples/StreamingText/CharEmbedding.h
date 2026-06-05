#pragma once

#include "Corpus.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace streaming_text {

inline constexpr std::size_t kCharEmbedDim = 64;

/// Random per-character feature table: `corpus.vocab.size()` contiguous blocks
/// of kCharEmbedDim floats, each element drawn uniformly from [-1, 1].
/// A character maps to its block via the Corpus's class-index mapping, so the
/// table automatically tracks whatever vocab size the Corpus is built with.
///
/// Sizing kCharEmbedDim
/// --------------------
/// kCharEmbedDim * kInputHistory must divide N = 2^kDIM (Reservoir routes each
/// input channel to a contiguous block of N/num_inputs vertices), so the value
/// must be a power of two. Larger embeddings give a richer fixed random code but
/// recruit fewer reservoir vertices per channel, eroding the nonlinear expansion;
/// and because the embedding is a *fixed* random code over a ~96-symbol vocab,
/// information saturates well before the routing limit. At DIM=9 (N=512):
///
///   kCharEmbedDim | verts/channel | verdict
///   --------------+---------------+-----------------------------------
///   32            | 16            | fine
///   64 (current)  |  8            | recommended
///   128           |  4            | marginal — expansion getting thin
///   256           |  2            | too fine, don't
///
/// Rule of thumb: at DIM=9, 64 is the sweet spot and 128 is the hard ceiling.
/// At DIM=12 (N=4096) — the bigger lever for the capacity floor — 64/128 are
/// comfortable (32–64 verts/channel) and 256 becomes physically available, though
/// still of limited value for a 96-symbol vocab unless kInputHistory is also
/// raised (a wider per-char code helps the readout disentangle window position).
class CharEmbedding
{
public:
    /// Allocate a (vocab.size() * kCharEmbedDim) buffer and fill every element
    /// with a uniform draw in [-1, 1] using a std::mt19937_64 seeded with `seed`.
    /// The Corpus reference is retained for use by Lookup() and must outlive
    /// this object.
    CharEmbedding(const Corpus& corpus, std::uint64_t seed);

    /// Pointer to the kCharEmbedDim-float block for `c`, or nullptr if `c` is
    /// not in the Corpus vocab.
    [[nodiscard]] const float* Lookup(char c) const;

    [[nodiscard]] std::size_t VocabSize() const { return vocab_size_; }

private:
    const Corpus*      corpus_;
    std::size_t        vocab_size_;
    std::vector<float> data_;
};

/// K-deep input shift register over CharEmbedding lookups.  At each step the
/// caller Push()es the newest character and feeds Inputs() (a flat
/// `history * kCharEmbedDim`-float buffer) to the reservoir as one timestep.
/// Slot 0 holds the oldest embedding; slot (history-1) holds the newest.
/// Push() shifts left by one slot, dropping the oldest.  Clear() zeros the
/// buffer (call at the start of each Warmup / Train phase, paired with
/// esn.ResetReservoirOnly()).
class RollingCharWindow
{
public:
    RollingCharWindow(const CharEmbedding& embed, std::size_t history);

    void Clear();
    void Push(char c);

    [[nodiscard]] const float* Inputs() const     { return buffer_.data(); }
    [[nodiscard]] std::size_t  InputSize() const  { return buffer_.size(); }
    [[nodiscard]] std::size_t  History() const    { return history_; }

private:
    const CharEmbedding* embed_;
    std::size_t          history_;
    std::vector<float>   buffer_;  // history_ * kCharEmbedDim, slot 0 oldest
};

}  // namespace streaming_text
