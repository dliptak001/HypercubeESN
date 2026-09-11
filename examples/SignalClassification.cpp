/// @file SignalClassification.cpp
/// @brief Multi-class process-mode classification from reservoir state.
/// See SignalClassification.md for walkthrough and experiments.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include "ESN.h"

// =============================================================================
// ESN configuration — primary knobs for this demo (edit here)
// =============================================================================

static constexpr size_t kNumClasses = 4;

static ESNConfig MakeESNConfig()
{
    ESNConfig cfg;

    // Reservoir (fixed dynamics)
    cfg.reservoir.dim             = 10;
    cfg.reservoir.seed            = 7934791766227647176;
    cfg.reservoir.history_depth   = 8;
    cfg.reservoir.spectral_radius = 0.999f;
    cfg.reservoir.input_scaling   = 0.1f;

    // Seam: delay-line ages packed into the readout input (power of two, ≤ M)
    cfg.readout_slices = 1;

    // Readout (trainable HCNN)
    cfg.readout.task          = ReadoutTask::Classification;
    cfg.readout.num_outputs   = static_cast<int>(kNumClasses);
    cfg.readout.epochs        = 50;
    cfg.readout.activation    = ReadoutActivation::TANH;  // TANH / RELU / LEAKY_RELU / NONE
    cfg.readout.conv_channels = 8;

    return cfg;
}

// =============================================================================
// Demo / task parameters (not part of ESNConfig)
// =============================================================================

static constexpr size_t kWarmup             = 300;
static constexpr size_t kBlockSize          = 40;
static constexpr size_t kNumBlocks          = 600;  // total mode blocks in the stream
static constexpr size_t kTrainBlocks        = 420;  // 70%
static constexpr size_t kTestBlocks         = kNumBlocks - kTrainBlocks;
static constexpr size_t kCollect            = kBlockSize * kNumBlocks;
static constexpr size_t kTrainSize          = kTrainBlocks * kBlockSize;
static constexpr size_t kTestSize           = kTestBlocks * kBlockSize;
static constexpr size_t kLockK              = 3;    // consecutive correct steps => locked
static constexpr size_t kStreamPrintBlocks  = 24;   // how many test blocks to print live
static constexpr uint64_t kStreamSeed       = 123456;

// Close frequencies so the readout must key on shape, not tone.
static constexpr float kClassFreq[kNumClasses] = {0.11f, 0.13f, 0.12f, 0.10f};
static constexpr float kNoiseLevel = 0.18f;

// Industrial process-mode labels (underlying shapes: sine / square / triangle / chirp).
static const char* kClassNames[kNumClasses] = {
    "Cruise  ", "Chatter ", "Ramp    ", "Spin-up "
};
static const char* kClassShapes[kNumClasses] = {
    "sine", "square", "triangle", "chirp"
};

// =============================================================================
// Helpers
// =============================================================================

static constexpr float PI = 3.14159265358979323846f;

/// Sample one of 4 waveforms by index: 0=sine, 1=square, 2=triangle, 3=chirp.
static float GenerateWaveform(size_t waveform, float phase)
{
    switch (waveform)
    {
    case 0: return std::sin(phase);
    case 1: return std::sin(phase) >= 0.0f ? 0.9f : -0.9f;
    case 2:
    {
        float p = std::fmod(phase, 2.0f * PI);
        if (p < 0) p += 2.0f * PI;
        return (p < PI) ? (-1.0f + 2.0f * p / PI) : (3.0f - 2.0f * p / PI);
    }
    case 3: return std::sin(phase + 0.3f * phase * phase);
    default: return 0.0f;
    }
}

/// Softmax probability of the argmax class (confidence) and its index.
static void ArgmaxSoftmax(const std::vector<float>& logits,
                          size_t& predicted, float& confidence)
{
    predicted = 0;
    float best = logits[0];
    for (size_t c = 1; c < logits.size(); ++c)
    {
        if (logits[c] > best)
        {
            best = logits[c];
            predicted = c;
        }
    }

    // Numerically stable softmax over the winning class only needs the partition.
    float m = best;
    float sum = 0.0f;
    for (float z : logits)
        sum += std::exp(z - m);
    confidence = std::exp(best - m) / sum;
}

struct BlockSummary
{
    size_t true_class = 0;
    size_t majority_pred = 0;
    double block_acc = 0.0;   // fraction correct within the block
    double mean_conf = 0.0;   // mean softmax conf of argmax
    int time_to_lock = -1;    // steps until K consecutive correct; -1 = never
    bool is_switch = false;   // true class differs from previous block
};

/// Majority vote over a block's step predictions.
static size_t MajorityClass(const size_t* preds, size_t n)
{
    size_t counts[kNumClasses] = {};
    for (size_t i = 0; i < n; ++i)
        counts[preds[i]]++;
    size_t best = 0;
    for (size_t c = 1; c < kNumClasses; ++c)
        if (counts[c] > counts[best]) best = c;
    return best;
}

/// First index where K consecutive steps are correct; -1 if never.
static int TimeToLock(const size_t* preds, const size_t* labels, size_t n, size_t k)
{
    if (k == 0 || k > n) return -1;
    size_t streak = 0;
    for (size_t i = 0; i < n; ++i)
    {
        if (preds[i] == labels[i])
        {
            ++streak;
            if (streak >= k)
                return static_cast<int>(i + 1 - k); // start of the locking streak
        }
        else
        {
            streak = 0;
        }
    }
    return -1;
}

static const char* StatusLabel(const BlockSummary& b, size_t lock_k)
{
    (void)lock_k;
    if (b.time_to_lock < 0)
        return "NO LOCK";
    if (b.is_switch && b.time_to_lock > 0)
        return "SWITCHING";
    if (b.block_acc >= 0.95 && b.mean_conf >= 0.70)
        return "LOCKED";
    if (b.block_acc >= 0.80)
        return "SETTLING";
    return "CONFUSED";
}

// =============================================================================

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    const ESNConfig cfg = MakeESNConfig();
    ESN esn(cfg);
    const size_t dim = cfg.reservoir.dim;
    const size_t N   = esn.ReservoirNeuronCount();

    std::cout << "=== HypercubeESN: Signal Classification ===\n\n";
    std::cout << "Scenario: an industrial process runs in one of four vibration modes.\n";
    std::cout << "A single vibration channel feeds the reservoir; the readout must name\n";
    std::cout << "the active mode from reservoir state alone -- not from raw samples.\n\n";
    std::cout << "Process modes (shape under the hood):\n";
    std::cout << "  1. Cruise   -- smooth harmonic load          (sine,    f=0.11)\n";
    std::cout << "  2. Chatter  -- hard-edged clutch / relay      (square,  f=0.13)\n";
    std::cout << "  3. Ramp     -- linear load ramp               (triangle,f=0.12)\n";
    std::cout << "  4. Spin-up  -- accelerating runaway           (chirp,   f=0.10)\n\n";
    std::cout << "Frequencies are deliberately close; noise=" << kNoiseLevel
              << " forces classification by shape.\n";
    std::cout << "No immediate class repeat -- residual errors and lock-on delay stay visible.\n\n";

    // Demo schedule (not in ReadoutArchSummary) + architecture once.
    std::cout << "Stream:  warmup=" << kWarmup
              << "  blocks=" << kNumBlocks << "x" << kBlockSize
              << "  train=" << kTrainBlocks << " (" << kTrainSize << " steps)"
              << "  test=" << kTestBlocks << " (" << kTestSize << " steps)\n";
    std::cout << "Train:   " << cfg.readout.epochs << " epochs"
              << "  batch=" << cfg.readout.batch_size
              << "  lr_max=" << cfg.readout.lr_max
              << "  classes=" << kNumClasses << "\n\n";
    std::cout << esn.ReadoutArchSummary() << "\n";

    std::vector<float> signal(kWarmup + kCollect);
    std::vector<size_t> labels(kCollect);
    std::vector<size_t> block_class(kNumBlocks);

    std::mt19937_64 rng(kStreamSeed);
    std::uniform_real_distribution<float> noise(-kNoiseLevel, kNoiseLevel);
    std::uniform_int_distribution<size_t> class_pick(0, kNumClasses - 1);

    // Random mode schedule: uniform class, no immediate repeat (more realistic switches).
    size_t prev = kNumClasses; // invalid sentinel
    for (size_t b = 0; b < kNumBlocks; ++b)
    {
        size_t c = class_pick(rng);
        if (c == prev)
            c = (c + 1 + class_pick(rng) % (kNumClasses - 1)) % kNumClasses;
        block_class[b] = c;
        prev = c;
    }

    for (size_t t = 0; t < kWarmup; ++t)
        signal[t] = GenerateWaveform(0, kClassFreq[0] * static_cast<float>(t))
                    + noise(rng);

    for (size_t t = 0; t < kCollect; ++t)
    {
        size_t b = t / kBlockSize;
        size_t waveform = block_class[b];
        size_t t_in_block = t % kBlockSize;
        labels[t] = waveform;
        float phase = kClassFreq[waveform] * static_cast<float>(t_in_block);
        signal[kWarmup + t] = GenerateWaveform(waveform, phase) + noise(rng);
    }

    const size_t* test_labels = labels.data() + kTrainSize;

    std::cout << "--- Phase 1: Learn the four process modes ---\n\n";
    esn.ReservoirWarmup(signal.data(), kWarmup);
    esn.ReservoirRun(signal.data() + kWarmup, kTrainSize + kTestSize);

    // Integer class indices for classification (not float-as-label).
    std::vector<int> class_labels(kCollect);
    for (size_t t = 0; t < kCollect; ++t)
        class_labels[t] = static_cast<int>(labels[t]);

    std::cout << "Training on " << kTrainSize << " steps (" << kTrainBlocks
              << " blocks)..." << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    esn.Train(class_labels.data(), kTrainSize);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(2) << secs << "s)\n\n";

    // Per-step predictions + confidence on the held-out stream.
    std::vector<size_t> predictions(kTestSize);
    std::vector<float> confidences(kTestSize);
    for (size_t t = 0; t < kTestSize; ++t)
    {
        const std::vector<float> logits = esn.PredictFromRecorded(kTrainSize + t);
        ArgmaxSoftmax(logits, predictions[t], confidences[t]);
    }

    // Per-block summaries on the test stream.
    std::vector<BlockSummary> blocks(kTestBlocks);
    size_t overall_ok = 0;
    size_t confusion[kNumClasses][kNumClasses] = {};
    for (size_t t = 0; t < kTestSize; ++t)
    {
        confusion[test_labels[t]][predictions[t]]++;
        if (predictions[t] == test_labels[t])
            ++overall_ok;
    }
    double overall_acc = 100.0 * static_cast<double>(overall_ok) / static_cast<double>(kTestSize);

    for (size_t b = 0; b < kTestBlocks; ++b)
    {
        const size_t off = b * kBlockSize;
        BlockSummary& s = blocks[b];
        s.true_class = test_labels[off];
        s.is_switch = (b == 0)
            ? (block_class[kTrainBlocks] != block_class[kTrainBlocks - 1])
            : (test_labels[off] != test_labels[off - 1]);
        s.majority_pred = MajorityClass(predictions.data() + off, kBlockSize);
        s.time_to_lock = TimeToLock(predictions.data() + off, test_labels + off,
                                    kBlockSize, kLockK);

        size_t ok = 0;
        double conf_sum = 0.0;
        for (size_t i = 0; i < kBlockSize; ++i)
        {
            if (predictions[off + i] == test_labels[off + i]) ++ok;
            conf_sum += confidences[off + i];
        }
        s.block_acc = static_cast<double>(ok) / static_cast<double>(kBlockSize);
        s.mean_conf = conf_sum / static_cast<double>(kBlockSize);
    }

    // ---- Phase 2: live block monitor (abbreviated) ----
    std::cout << "--- Phase 2: Monitor the process stream ("
              << kTestBlocks << " held-out blocks) ---\n\n";
    std::cout << "Each block is scored step-by-step from frozen reservoir state.\n";
    std::cout << "Conf = mean softmax probability of the argmax class.\n";
    std::cout << "TTL  = steps until " << kLockK
              << " consecutive correct predictions (- = never).\n";
    std::cout << "Showing first " << kStreamPrintBlocks
              << " blocks; full metrics follow.\n\n";

    std::cout << "  Blk | True      Pred      | Acc%  Conf  TTL | Status\n";
    std::cout << "  ----+---------------------+-----------------+----------\n";

    size_t print_n = std::min(kStreamPrintBlocks, kTestBlocks);
    for (size_t b = 0; b < print_n; ++b)
    {
        const BlockSummary& s = blocks[b];
        size_t ok = 0;
        const size_t off = b * kBlockSize;
        for (size_t i = 0; i < kBlockSize; ++i)
            if (predictions[off + i] == test_labels[off + i]) ++ok;
        double acc_pct = 100.0 * static_cast<double>(ok) / static_cast<double>(kBlockSize);

        char ttl_buf[8];
        if (s.time_to_lock < 0)
            std::snprintf(ttl_buf, sizeof(ttl_buf), "  -");
        else
            std::snprintf(ttl_buf, sizeof(ttl_buf), "%3d", s.time_to_lock);

        std::cout << "  " << std::setw(3) << (b + 1)
                  << " | " << kClassNames[s.true_class]
                  << "  " << kClassNames[s.majority_pred]
                  << " | " << std::fixed << std::setprecision(0) << std::setw(4) << acc_pct
                  << "  " << std::setprecision(2) << std::setw(4) << s.mean_conf
                  << "  " << ttl_buf
                  << " | " << StatusLabel(s, kLockK) << "\n";
    }
    if (kTestBlocks > print_n)
        std::cout << "  ... (" << (kTestBlocks - print_n) << " more blocks omitted)\n";
    std::cout << "\n";

    // ---- Aggregate results ----
    std::cout << "--- Results (test set: " << kTestSize << " steps / "
              << kTestBlocks << " blocks) ---\n\n";
    std::cout << "Overall step accuracy: " << std::fixed << std::setprecision(1)
              << overall_acc << "%\n\n";

    std::cout << "Per-mode breakdown:\n";
    size_t hardest = 0;
    double hardest_acc = 200.0;
    size_t confuse_a = 0, confuse_b = 0;
    size_t confuse_count = 0;
    for (size_t c = 0; c < kNumClasses; ++c)
    {
        size_t total_c = 0;
        for (size_t p = 0; p < kNumClasses; ++p)
            total_c += confusion[c][p];
        double acc = (total_c > 0) ? 100.0 * confusion[c][c] / total_c : 0.0;
        if (acc < hardest_acc)
        {
            hardest_acc = acc;
            hardest = c;
        }
        for (size_t p = 0; p < kNumClasses; ++p)
        {
            if (p != c && confusion[c][p] > confuse_count)
            {
                confuse_count = confusion[c][p];
                confuse_a = c;
                confuse_b = p;
            }
        }

        std::cout << "  " << kClassNames[c] << "  " << std::setprecision(1)
                  << std::setw(5) << acc << "%  (" << confusion[c][c]
                  << "/" << total_c << ")";
        if (acc >= 99.5)
            std::cout << "  -- near-perfect";
        else if (acc >= 95.0)
            std::cout << "  -- strong";
        else if (acc < 90.0)
        {
            size_t max_conf = 0, max_conf_class = c;
            for (size_t p = 0; p < kNumClasses; ++p)
            {
                if (p != c && confusion[c][p] > max_conf)
                {
                    max_conf = confusion[c][p];
                    max_conf_class = p;
                }
            }
            if (max_conf > 0)
            {
                double conf_pct = 100.0 * max_conf / total_c;
                std::cout << "  -- confused with " << kClassNames[max_conf_class]
                          << " " << std::setprecision(0) << conf_pct << "% of steps";
            }
        }
        std::cout << "\n";
    }

    std::cout << "\nConfusion matrix (rows=actual, cols=predicted; % of row):\n";
    std::cout << "               ";
    for (size_t c = 0; c < kNumClasses; ++c)
        std::cout << kClassNames[c] << "  ";
    std::cout << "\n";
    for (size_t a = 0; a < kNumClasses; ++a)
    {
        size_t total_a = 0;
        for (size_t p = 0; p < kNumClasses; ++p)
            total_a += confusion[a][p];
        std::cout << "  " << kClassNames[a] << " |";
        for (size_t p = 0; p < kNumClasses; ++p)
        {
            double pct = (total_a > 0) ? 100.0 * confusion[a][p] / total_a : 0.0;
            std::cout << std::setw(7) << std::setprecision(1) << pct << "%  ";
        }
        std::cout << "\n";
    }

    // Aggregate lock-on: accuracy in the first M steps of each test block.
    std::cout << "\nLock-on dynamics (accuracy in first M steps of each block):\n";
    std::cout << "  Steps after switch  | Accuracy\n";
    std::cout << "  --------------------+---------\n";
    const size_t margins[] = {3, 5, 10, 20, kBlockSize};
    for (size_t margin : margins)
    {
        size_t ok = 0, total = 0;
        for (size_t b = 0; b < kTestBlocks; ++b)
        {
            const size_t off = b * kBlockSize;
            for (size_t i = 0; i < margin && i < kBlockSize; ++i)
            {
                ++total;
                if (predictions[off + i] == test_labels[off + i]) ++ok;
            }
        }
        double acc = (total > 0) ? 100.0 * ok / total : 0.0;
        if (margin == kBlockSize)
            std::cout << "  Entire block        |  " << std::setprecision(1)
                      << std::setw(5) << acc << "%  (overall)\n";
        else
            std::cout << "  0 - " << std::setw(2) << margin
                      << "              |  " << std::setprecision(1)
                      << std::setw(5) << acc << "%\n";
    }

    // Time-to-lock stats on switch blocks only.
    size_t switch_blocks = 0, locked_switches = 0;
    double ttl_sum = 0.0;
    int ttl_max = 0;
    size_t no_lock = 0;
    for (size_t b = 0; b < kTestBlocks; ++b)
    {
        if (!blocks[b].is_switch) continue;
        ++switch_blocks;
        if (blocks[b].time_to_lock < 0)
        {
            ++no_lock;
        }
        else
        {
            ++locked_switches;
            ttl_sum += blocks[b].time_to_lock;
            if (blocks[b].time_to_lock > ttl_max)
                ttl_max = blocks[b].time_to_lock;
        }
    }

    std::cout << "\nTime-to-lock (TTL = first step of " << kLockK
              << " consecutive correct; switch blocks only):\n";
    std::cout << "  Switch blocks: " << switch_blocks
              << "   locked: " << locked_switches
              << "   never: " << no_lock << "\n";
    if (locked_switches > 0)
    {
        std::cout << "  Mean TTL: " << std::setprecision(1)
                  << (ttl_sum / locked_switches)
                  << "   max TTL: " << ttl_max << "\n";
    }
    std::cout << "\n";

    // ---- What happened (data-driven) ----
    std::cout << "--- What happened ---\n\n";
    std::cout << "The reservoir encodes recent vibration history as a high-dimensional\n";
    std::cout << "state; the HCNN readout maps that state to one of four process modes.\n\n";

    std::cout << "  Capacity:  DIM=" << dim << " (N=" << N << ").\n";
    std::cout << "             Overall step accuracy landed at " << std::setprecision(1)
              << overall_acc << "%";
    if (overall_acc >= 99.5)
        std::cout << " -- near ceiling on this stream.\n\n";
    else if (overall_acc >= 92.0)
        std::cout << " -- strong, with a residual hard pair.\n\n";
    else
        std::cout << " -- capacity/noise leave clear headroom to study.\n\n";

    std::cout << "  Hardest mode: " << kClassNames[hardest]
              << " (" << kClassShapes[hardest] << ") at "
              << std::setprecision(1) << hardest_acc << "% step accuracy.\n";
    if (confuse_count > 0 && confuse_a != confuse_b)
    {
        size_t row_tot = 0;
        for (size_t p = 0; p < kNumClasses; ++p)
            row_tot += confusion[confuse_a][p];
        double pct = (row_tot > 0) ? 100.0 * confuse_count / row_tot : 0.0;
        std::cout << "  Top confusion: " << kClassNames[confuse_a]
                  << " -> " << kClassNames[confuse_b]
                  << " (" << std::setprecision(0) << pct
                  << "% of that mode's steps).\n";
        std::cout << "             Smooth harmonic vs chirp (Cruise/Spin-up) share\n";
        std::cout << "             early-block shape under noise; Chatter/Ramp stay clean.\n";
    }
    std::cout << "\n";

    if (locked_switches > 0)
    {
        std::cout << "  Lock-on:   After a mode switch, mean time-to-lock is "
                  << std::setprecision(1) << (ttl_sum / locked_switches)
                  << " steps\n";
        std::cout << "             (need " << kLockK
                  << " correct in a row). Early-window accuracy (0-3 / 0-5)\n";
        std::cout << "             trails full-block accuracy -- residual state from\n";
        std::cout << "             the previous mode for a few ticks, then LOCKED.\n";
        if (no_lock > 0)
            std::cout << "             " << no_lock
                      << " switch block(s) never reached a " << kLockK
                      << "-step lock.\n";
    }
    std::cout << "\n";
    std::cout << "  Stream view: LOCKED means high block accuracy + confidence;\n";
    std::cout << "             SWITCHING means a mode change still washing through;\n";
    std::cout << "             SETTLING / CONFUSED / NO LOCK mark partial or failed locks.\n";

    std::cout << std::flush;
    return 0;
}
