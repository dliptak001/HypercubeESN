#include "CharEmbedding.h"

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

CharInput::CharInput(const CharEmbedding& embed)
    : embed_(&embed),
      buffer_(kCharEmbedDim, 0.0f)
{
}

void CharInput::Set(char c)
{
    const float* emb = embed_->Lookup(c);
    if (emb != nullptr) {
        std::memcpy(buffer_.data(), emb, kCharEmbedDim * sizeof(float));
    } else {
        std::memset(buffer_.data(), 0, kCharEmbedDim * sizeof(float));
    }
}

}  // namespace streaming_text
