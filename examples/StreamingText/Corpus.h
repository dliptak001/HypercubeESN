#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace streaming_text {

inline constexpr std::size_t kAsciiTableSize = 128;  ///< char_to_class lookup size; spans 7-bit ASCII (indexed by byte value, not class)
inline constexpr std::size_t kVocabSize  = 96;   ///< newline + printable ASCII 0x20-0x7E

namespace detail {
/// All-(-1) lookup: an unbuilt Corpus maps every byte to "not in vocab" (-1)
/// rather than silently to class 0.
constexpr std::array<int, kAsciiTableSize> EmptyCharTable()
{
    std::array<int, kAsciiTableSize> t{};
    for (int& x : t) x = -1;
    return t;
}
}  // namespace detail

/// Text corpus + fixed vocabulary.  `vocab` is always the fixed 96-token
/// set; a char's class index is its position in `vocab`.
struct Corpus
{
    std::string text;
    std::string vocab;                  ///< position == class index
    std::array<int, kAsciiTableSize> char_to_class = detail::EmptyCharTable();  ///< ASCII lookup, -1 if not in vocab
};

/// Load a plain-text corpus from disk with the fixed vocab.  Returns false
/// on missing file, empty corpus, or any byte outside the fixed vocab.
/// Populates `out.text`, `out.vocab`, `out.char_to_class`.
bool LoadCorpus(const std::string& path, Corpus& out);

/// Character -> class index ([0, vocab.size())), or -1 if not in vocab.
inline int CharToClass(const Corpus& c, char ch)
{
    const auto u = static_cast<unsigned char>(ch);
    return (u < kAsciiTableSize) ? c.char_to_class[u] : -1;
}

/// Class index -> character.  Returns '?' on out-of-range.
inline char ClassToChar(const Corpus& c, int cls)
{
    if (cls < 0 || static_cast<std::size_t>(cls) >= c.vocab.size()) return '?';
    return c.vocab[static_cast<std::size_t>(cls)];
}

}  // namespace streaming_text
