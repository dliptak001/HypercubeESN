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
    /// Stock survey used 11; edit freely — the run banner reports live ESN dim/N.
    inline constexpr std::size_t kDIM = 12;

    /// Sanity guard on the channel-to-vertex distribution: Reservoir routes
    /// channel c to a contiguous vertex block, which requires num_inputs
    /// (= kCharEmbedDim) to divide N = 2^kDIM cleanly.
    static_assert((1ULL << kDIM) % kCharEmbedDim == 0,
                  "kCharEmbedDim must divide N = 2^kDIM for even "
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
        // Path is resolved relative to the process working directory (or absolute).
        // Alphabet is the fixed 96-token set (newline + printable ASCII 0x20-0x7E);
        // any other byte is a hard load error. Plain ASCII text only — not free UTF-8.
        std::string corpus_path = "C:\\Hypercube\\tinyshakespeare.txt";

        // --- Stream budget ---------------------------------------------------
        // The corpus is a ring buffer of length L = text.size().
        // The stream advances pos = (pos + 1) % L; each full traversal is a
        // "lap".  total_steps is the single budget; no per-pass reset.
        //
        // warmup_chars: reservoir-only drive (no readout forward/train) to wash
        // out zero-init and fill the delay line before the train loop.
        std::size_t warmup_chars = 1024; ///reservoir-only washout
        std::size_t total_steps = 45000000; ///< total streamed train chars (laps ≈ total_steps / L)

        // ESN config: struct defaults + StreamingText overrides.  Edit fields
        // here to tune reservoir dynamics or CNN architecture.
        //
        // Streaming path does NOT use readout.lr_min_frac or readout.momentum:
        // LR comes from Cfg::lr_min_frac + CosineLR on TrainStepBatch; optimizer
        // is Adam (momentum field is SGD-only). Leave those at library defaults.
        ESNConfig esn = []
        {
            ESNConfig c;
            c.reservoir.dim = kDIM;
            c.reservoir.seed = 665127;
            c.reservoir.history_depth = 2;
            c.reservoir.num_inputs = static_cast<int>(kCharEmbedDim);
            c.reservoir.spectral_radius = 0.8f;
            c.reservoir.leak_rate = 0.8f;
            c.reservoir.input_scaling = 2.5f;
            c.reservoir.full_state_feedback = false;
            c.reservoir.fsf_seed = 44157563;
            c.reservoir.fsf_scaling = 0.1f;
            c.reservoir.verbose = false; // avoid duplicate ctor banner; [stext] + ArchSummary cover it

            //c.readout.seed = 54544;
            c.readout.seed = 3423555;
            c.readout_slices = 2;
            c.readout.task = ReadoutTask::Classification;
            c.readout.num_outputs = static_cast<int>(kVocabSize);
            c.readout.num_layers = 1;
            c.readout.conv_channels = 2;
            c.readout.weight_decay = 1e-5f;
            c.readout.lr_max = 0.0003f;
            c.readout.pool_type = ReadoutPoolType::Max;
            c.readout.activation = ReadoutActivation::NONE; //NONE beats LEAKY beats TANH

            return c;
        }();

        // --- Streaming knobs not in ESNConfig --------------------------------
        int mini_batch_size = 64; ///< grad-accum chunk for TrainStepBatch
        float lr_min_frac = 0.2f; ///< cosine floor as fraction of readout.lr_max (active path)

        // --- Prequential metric reporting ------------------------------------
        std::size_t report_window = 10000; ///< rolling-BPC / top-1 window length (chars)
        std::size_t report_every = 100000; ///< print one live rolling line every N chars (0 = end only)

        // --- Teacher-forced sample display -----------------------------------
        // Once per completed lap (every L train steps), emit sample_len chars of
        // predicted-vs-actual text starting at the lap boundary. 0 = off.
        std::size_t sample_len = 120; ///< chars shown at end of each lap (0 = off)

        bool verbose = true; ///< gates periodic roll_bpc lines only (not the startup banner)
    };

    inline const Cfg kCfg;
} // namespace streaming_text::config
