#include "CharEmbedding.h"

#include <cassert>
#include <cstring>
#include <random>

namespace streaming_text {

CharEmbedding::CharEmbedding(const Corpus& corpus, std::uint64_t seed)
    : corpus_(&corpus),
      vocab_size_(corpus.vocab.size()),
      data_(vocab_size_ * kCharEmbedDim)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : data_) v = dist(rng);
}

const float* CharEmbedding::Lookup(char c) const
{
    const int cls = CharToClass(*corpus_, c);
    if (cls < 0) return nullptr;
    return data_.data() + static_cast<std::size_t>(cls) * kCharEmbedDim;
}

RollingCharWindow::RollingCharWindow(const CharEmbedding& embed, std::size_t history)
    : embed_(&embed),
      history_(history),
      buffer_(history * kCharEmbedDim, 0.0f)
{
    assert(history_ >= 1 && "RollingCharWindow history must be >= 1");
}

void RollingCharWindow::Clear()
{
    std::memset(buffer_.data(), 0, buffer_.size() * sizeof(float));
}

void RollingCharWindow::Push(char c)
{
    // Shift left by one slot: slot[i] <- slot[i+1] for i in [0, history-2].
    // Drops the oldest embedding off the front; opens the newest slot.
    if (history_ > 1) {
        std::memmove(buffer_.data(),
                     buffer_.data() + kCharEmbedDim,
                     (history_ - 1) * kCharEmbedDim * sizeof(float));
    }
    float* newest = buffer_.data() + (history_ - 1) * kCharEmbedDim;
    const float* emb = embed_->Lookup(c);
    if (emb != nullptr) {
        std::memcpy(newest, emb, kCharEmbedDim * sizeof(float));
    } else {
        // Char not in vocab — push a zero embedding rather than crash.
        std::memset(newest, 0, kCharEmbedDim * sizeof(float));
    }
}

}  // namespace streaming_text
