#include "Corpus.h"

#include <cstdio>
#include <fstream>

namespace streaming_text {

namespace {

void BuildCharTable(Corpus& c)
{
    c.char_to_class.fill(-1);
    for (std::size_t i = 0; i < c.vocab.size(); ++i) {
        const auto u = static_cast<unsigned char>(c.vocab[i]);
        if (u < kAsciiTableSize) c.char_to_class[u] = static_cast<int>(i);
    }
}

std::string MakeFixedVocab()
{
    std::string v;
    v.reserve(kVocabSize);
    v += '\n';                        // 0x0A — newline
    // char is signed on MinGW; 0x7E (126) is the last value, so ++c never reaches
    // 0x80, where a signed-char increment would be UB. Extend past 0x7E -> use int.
    for (char c = 0x20; c <= 0x7E; ++c)  // printable ASCII
        v += c;
    return v;
}

/// The fixed 96-token vocabulary string (sorted by byte value).
const std::string& FixedVocab()
{
    static const std::string v = MakeFixedVocab();
    return v;
}

}  // namespace

bool LoadCorpus(const std::string& path, Corpus& out)
{
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;

    is.seekg(0, std::ios::end);
    const std::streamoff len = is.tellg();
    if (len <= 0) return false;                       // empty or unseekable
    out.text.resize(static_cast<std::size_t>(len));
    is.seekg(0, std::ios::beg);
    is.read(out.text.data(), len);                    // string::data() is char* (C++17+)
    if (!is) return false;                            // short read

    out.vocab = FixedVocab();
    BuildCharTable(out);

    for (std::size_t i = 0; i < out.text.size(); ++i) {
        if (CharToClass(out, out.text[i]) < 0) {
            std::fprintf(stderr, "error: corpus byte 0x%02X at offset %zu "
                         "is outside the fixed 96-token vocab\n",
                         static_cast<unsigned char>(out.text[i]), i);
            return false;
        }
    }
    return true;
}

}  // namespace streaming_text
