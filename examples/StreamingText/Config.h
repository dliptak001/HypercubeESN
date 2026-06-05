#pragma once

// Hardcoded run-time configuration for StreamingText.  Edit these values,
// rebuild, and launch the exe with no arguments.  There is
// a single mode: stream the corpus as a ring buffer and memorize it.  No
// held-out validation split, no model serialization, no autoregressive
// inference — the metric is computed inline (prequential) on the training
// stream and progress is shown via teacher-forced predicted-vs-actual
// readouts.  See StreamingText.md.
//
// NOTE: corpus_path must point to a plain-text file whose bytes all fall
// within the fixed 96-token vocabulary (newline + printable ASCII
// 0x20-0x7E).  Tiny Shakespeare (~1 MB) is the default working corpus.

#include <cstddef>
#include <cstdint>
#include <string>

#include "CharEmbedding.h"  // kCharEmbedDim
#include "Corpus.h"         // kVocabSize, Corpus
#include "ESN.h"            // ESNConfig (+ ReservoirConfig / ReadoutConfig), ReadoutActivation

namespace streaming_text::config
{
    inline constexpr std::size_t kDIM = 11;

    /// Number of most-recent characters concatenated into the reservoir's input
    /// vector per timestep.  Each timestep the reservoir sees the K most recent
    /// CharEmbedding lookups packed side by side (`K * kCharEmbedDim` channels).
    /// K=1 is "current char only"; K>1 gives the reservoir a fixed shift-register
    /// lookback so short-range n-gram patterns are directly readable.
    inline constexpr std::size_t kInputHistory = 1;

    /// Sanity guard on the channel-to-vertex distribution: Reservoir routes
    /// channel c to a contiguous vertex block, which requires num_inputs to
    /// divide N = 2^kDIM cleanly.
    static_assert((1ULL << kDIM) % (kCharEmbedDim * kInputHistory) == 0,
                  "kCharEmbedDim * kInputHistory must divide N = 2^kDIM for even "
                  "channel-to-vertex routing in the Reservoir");

    /// XOR mask applied to the reservoir seed to derive the CharEmbedding seed.
    /// Keeps the embedding table deterministic per reservoir seed but uncorrelated
    /// from the reservoir RNG stream.
    inline constexpr std::uint64_t kCharEmbedSeedXor = 0x9E3779B97F4A7C15ULL;

    // -----------------------------------------------------------------------------
    // Streaming memorization run.
    // -----------------------------------------------------------------------------
    struct Cfg
    {
        // Plain-text corpus to memorize. NOT bundled with the repo — supply your
        // own and point this at it. Default is Tiny Shakespeare (~1.1 MB), from
        // https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
        // Any ASCII/UTF-8 text works; the vocabulary is built from its distinct bytes.
        std::string corpus_path = "tinyshakespeare.txt";

        // --- Stream budget ---------------------------------------------------
        // The corpus is treated as a ring buffer of length L = text.size().
        // The stream advances pos = (pos + 1) % L; each full traversal is a
        // "lap" (the analogue of an epoch).  total_steps is the single budget;
        // there is no per-pass reset — reservoir state flows continuously.
        std::size_t warmup_chars = 1024; ///< transient reservoir warmup (no training)
        std::size_t warmup_train_chars = 32768; ///< chars driven through InitOnline (builds the CNN)
        std::size_t total_steps = 45000000; ///< total streamed chars (laps = total_steps / L)

        // ESN config: struct defaults + StreamingText overrides.  Edit fields
        // here to tune reservoir dynamics or CNN architecture.
        ESNConfig esn = []
        {
            ESNConfig c;
            c.reservoir.dim = kDIM;
            c.reservoir.seed = 7397376;
            c.reservoir.history_depth = 1;
            c.reservoir.num_inputs = static_cast<int>(kCharEmbedDim * kInputHistory);
            c.reservoir.spectral_radius = 1.3f;
            c.reservoir.leak_rate = 1.0f;
            c.reservoir.input_scaling = 2.5f;

            c.readout.seed = 54544;
            c.readout.task = ReadoutTask::Classification;
            c.readout.num_outputs = static_cast<int>(kVocabSize);
            c.readout.num_layers = 1;
            c.readout.conv_channels = 16;
            c.readout.weight_decay = 1e-5f;
            c.readout.lr_max = 0.001f;
            c.readout.lr_min_frac = 0.02f; // INERT in this example: the streaming path
                                           // (InitOnline + TrainLiveBatch) takes an explicit
                                           // per-batch lr; the readout's own cosine (the only
                                           // consumer of readout.lr_min_frac) is never run.
                                           // The active LR floor is Cfg::lr_min_frac below.
            c.readout.momentum = 0.9f; // momentum term for the readout optimizer (ADAM)
            c.readout.activation = ReadoutActivation::TANH;

            return c;
        }();

        // --- Streaming knobs not in ESNConfig --------------------------------
        int mini_batch_size = 64; ///< grad-accum chunk for TrainLiveBatch
        float lr_min_frac = 0.2f; ///< cosine schedule floor as fraction of lr_max (1.0 = flat)

        // --- Prequential metric reporting ------------------------------------
        std::size_t report_window = 10000; ///< rolling-BPC / top-1 window length (chars)
        std::size_t report_every = 100000; ///< print one live rolling line every N chars (0 = end only)

        // --- Teacher-forced sample display -----------------------------------
        // Every sample_every chars, emit sample_len chars of predicted-vs-actual
        // text at the stream's current position (no reset, no autoregression).
        // sample_every should not divide L, so sample positions precess across
        // laps automatically (any round constant works for a real corpus).
        std::size_t sample_every = 250000; ///< chars between teacher-forced readouts (0 = off)
        std::size_t sample_len = 120; ///< chars shown per readout

        bool verbose = true;
    };

    inline const Cfg kCfg;
} // namespace streaming_text::config
