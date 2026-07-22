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

static constexpr float PI = 3.14159265358979323846f;
static constexpr size_t NUM_CLASSES = 4;

// Industrial process-mode labels (underlying shapes: sine / square / triangle / chirp).
static const char* CLASS_NAMES[NUM_CLASSES] = {
    "Cruise  ", "Chatter ", "Ramp    ", "Spin-up "
};
static const char* CLASS_SHAPES[NUM_CLASSES] = {
    "sine", "square", "triangle", "chirp"
};

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

// Close frequencies so the readout must key on shape, not tone.
static constexpr float CLASS_FREQ[NUM_CLASSES] = { 0.11f, 0.13f, 0.12f, 0.10f };
static constexpr float NOISE_LEVEL = 0.18f;

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
    size_t counts[NUM_CLASSES] = {};
    for (size_t i = 0; i < n; ++i)
        counts[preds[i]]++;
    size_t best = 0;
    for (size_t c = 1; c < NUM_CLASSES; ++c)
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

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    constexpr size_t DIM = 8;
    constexpr size_t N = 1ULL << DIM;
    constexpr size_t warmup = 300;
    constexpr size_t block_size = 40;
    constexpr size_t num_blocks = 600; // total mode blocks in the stream
    constexpr size_t collect = block_size * num_blocks;
    constexpr size_t train_blocks = 420; // 70%
    constexpr size_t test_blocks = num_blocks - train_blocks;
    constexpr size_t train_size = train_blocks * block_size;
    constexpr size_t test_size = test_blocks * block_size;
    constexpr size_t lock_k = 3; // consecutive correct steps => locked

    // How many test blocks to print live (full metrics still cover all).
    constexpr size_t stream_print_blocks = 24;

    constexpr uint64_t seed = 123456;

    std::cout << "=== HypercubeESN: Signal Classification ===\n\n";
    std::cout << "Scenario: an industrial process runs in one of four vibration modes.\n";
    std::cout << "A single vibration channel feeds the reservoir; the readout must name\n";
    std::cout << "the active mode from reservoir state alone -- not from raw samples.\n\n";
    std::cout << "Process modes (shape under the hood):\n";
    std::cout << "  1. Cruise   -- smooth harmonic load          (sine,    f=0.11)\n";
    std::cout << "  2. Chatter  -- hard-edged clutch / relay      (square,  f=0.13)\n";
    std::cout << "  3. Ramp     -- linear load ramp               (triangle,f=0.12)\n";
    std::cout << "  4. Spin-up  -- accelerating runaway           (chirp,   f=0.10)\n\n";
    std::cout << "Stream: " << num_blocks << " random-order blocks of " << block_size
              << " steps (no immediate class repeat).\n";
    std::cout << "Frequencies are deliberately close; noise=" << NOISE_LEVEL
              << " forces classification by shape.\n";
    std::cout << "DIM=" << DIM << " (N=" << N
              << "): strong overall ID; residual errors and lock-on delay stay visible.\n\n";

    std::vector<float> signal(warmup + collect);
    std::vector<size_t> labels(collect);
    std::vector<size_t> block_class(num_blocks);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> noise(-NOISE_LEVEL, NOISE_LEVEL);
    std::uniform_int_distribution<size_t> class_pick(0, NUM_CLASSES - 1);

    // Random mode schedule: uniform class, no immediate repeat (more realistic switches).
    size_t prev = NUM_CLASSES; // invalid sentinel
    for (size_t b = 0; b < num_blocks; ++b)
    {
        size_t c = class_pick(rng);
        if (c == prev)
            c = (c + 1 + class_pick(rng) % (NUM_CLASSES - 1)) % NUM_CLASSES;
        block_class[b] = c;
        prev = c;
    }

    for (size_t t = 0; t < warmup; ++t)
        signal[t] = GenerateWaveform(0, CLASS_FREQ[0] * static_cast<float>(t))
                    + noise(rng);

    for (size_t t = 0; t < collect; ++t)
    {
        size_t b = t / block_size;
        size_t waveform = block_class[b];
        size_t t_in_block = t % block_size;
        labels[t] = waveform;
        float phase = CLASS_FREQ[waveform] * static_cast<float>(t_in_block);
        signal[warmup + t] = GenerateWaveform(waveform, phase) + noise(rng);
    }

    const size_t* test_labels = labels.data() + train_size;

    ESNConfig cfg;
    cfg.reservoir.dim = DIM;
    cfg.reservoir.spectral_radius = 0.95;
    cfg.reservoir.input_scaling = 0.1;
    cfg.readout.num_outputs = NUM_CLASSES;
    cfg.readout.task = ReadoutTask::Classification;
    cfg.readout.epochs = 100;
    cfg.readout.activation = ReadoutActivation::TANH; // TANH / RELU / LEAKY_RELU / NONE
    // Full-state linear feedback (internal). Edit these three for A/B.
    cfg.reservoir.full_state_feedback = false;
    cfg.reservoir.fsf_seed = 4415756;
    cfg.reservoir.fsf_scaling = 0.003f;
    ESN esn(cfg);

    std::cout << "Config: DIM=" << DIM << "  N=" << N
              << "  History Depth=" << cfg.reservoir.history_depth
              << "  Input Scaling=" << cfg.reservoir.input_scaling
              << "  Task=Classification  Classes=" << NUM_CLASSES << "\n";
    if (cfg.reservoir.full_state_feedback)
        std::cout << "  FSF: ON   fsf_seed=" << cfg.reservoir.fsf_seed
                  << "  fsf_scaling=" << cfg.reservoir.fsf_scaling << "\n";
    else
        std::cout << "  FSF: OFF\n";
    std::cout << esn.ReadoutArchSummary();
    std::cout << "\n";

    std::cout << "--- Phase 1: Learn the four process modes ---\n\n";
    esn.ReservoirWarmup(signal.data(), warmup);
    esn.ReservoirRun(signal.data() + warmup, train_size + test_size);

    std::vector<float> float_labels(collect);
    for (size_t t = 0; t < collect; ++t)
        float_labels[t] = static_cast<float>(labels[t]);

    std::cout << "Training on " << train_size << " samples (" << train_blocks
              << " blocks; " << cfg.readout.epochs << " epochs, batch="
              << cfg.readout.batch_size << ", lr_max=" << std::setprecision(4)
              << cfg.readout.lr_max << ")..." << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    esn.Train(float_labels.data(), train_size);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(2) << secs << "s)\n\n";

    // Per-step predictions + confidence on the held-out stream.
    std::vector<size_t> predictions(test_size);
    std::vector<float> confidences(test_size);
    for (size_t t = 0; t < test_size; ++t)
    {
        const std::vector<float> logits = esn.PredictFromRecorded(train_size + t);
        ArgmaxSoftmax(logits, predictions[t], confidences[t]);
    }

    // Per-block summaries on the test stream.
    std::vector<BlockSummary> blocks(test_blocks);
    size_t overall_ok = 0;
    size_t confusion[NUM_CLASSES][NUM_CLASSES] = {};
    for (size_t t = 0; t < test_size; ++t)
    {
        confusion[test_labels[t]][predictions[t]]++;
        if (predictions[t] == test_labels[t])
            ++overall_ok;
    }
    double overall_acc = 100.0 * static_cast<double>(overall_ok) / static_cast<double>(test_size);

    for (size_t b = 0; b < test_blocks; ++b)
    {
        const size_t off = b * block_size;
        BlockSummary& s = blocks[b];
        s.true_class = test_labels[off];
        s.is_switch = (b == 0)
            ? (block_class[train_blocks] != block_class[train_blocks - 1])
            : (test_labels[off] != test_labels[off - 1]);
        s.majority_pred = MajorityClass(predictions.data() + off, block_size);
        s.time_to_lock = TimeToLock(predictions.data() + off, test_labels + off,
                                    block_size, lock_k);

        size_t ok = 0;
        double conf_sum = 0.0;
        for (size_t i = 0; i < block_size; ++i)
        {
            if (predictions[off + i] == test_labels[off + i]) ++ok;
            conf_sum += confidences[off + i];
        }
        s.block_acc = static_cast<double>(ok) / static_cast<double>(block_size);
        s.mean_conf = conf_sum / static_cast<double>(block_size);
    }

    // ---- Phase 2: live block monitor (abbreviated) ----
    std::cout << "--- Phase 2: Monitor the process stream ("
              << test_blocks << " held-out blocks) ---\n\n";
    std::cout << "Each block is scored step-by-step from frozen reservoir state.\n";
    std::cout << "Conf = mean softmax probability of the argmax class.\n";
    std::cout << "TTL  = steps until " << lock_k
              << " consecutive correct predictions (- = never).\n";
    std::cout << "Showing first " << stream_print_blocks
              << " blocks; full metrics follow.\n\n";

    std::cout << "  Blk | True      Pred      | Acc%  Conf  TTL | Status\n";
    std::cout << "  ----+---------------------+-----------------+----------\n";

    size_t print_n = std::min(stream_print_blocks, test_blocks);
    for (size_t b = 0; b < print_n; ++b)
    {
        const BlockSummary& s = blocks[b];
        size_t ok = 0;
        const size_t off = b * block_size;
        for (size_t i = 0; i < block_size; ++i)
            if (predictions[off + i] == test_labels[off + i]) ++ok;
        double acc_pct = 100.0 * static_cast<double>(ok) / static_cast<double>(block_size);

        char ttl_buf[8];
        if (s.time_to_lock < 0)
            std::snprintf(ttl_buf, sizeof(ttl_buf), "  -");
        else
            std::snprintf(ttl_buf, sizeof(ttl_buf), "%3d", s.time_to_lock);

        std::cout << "  " << std::setw(3) << (b + 1)
                  << " | " << CLASS_NAMES[s.true_class]
                  << "  " << CLASS_NAMES[s.majority_pred]
                  << " | " << std::fixed << std::setprecision(0) << std::setw(4) << acc_pct
                  << "  " << std::setprecision(2) << std::setw(4) << s.mean_conf
                  << "  " << ttl_buf
                  << " | " << StatusLabel(s, lock_k) << "\n";
    }
    if (test_blocks > print_n)
        std::cout << "  ... (" << (test_blocks - print_n) << " more blocks omitted)\n";
    std::cout << "\n";

    // ---- Aggregate results ----
    std::cout << "--- Results (test set: " << test_size << " steps / "
              << test_blocks << " blocks) ---\n\n";
    std::cout << "Overall step accuracy: " << std::fixed << std::setprecision(1)
              << overall_acc << "%\n\n";

    std::cout << "Per-mode breakdown:\n";
    size_t hardest = 0;
    double hardest_acc = 200.0;
    size_t confuse_a = 0, confuse_b = 0;
    size_t confuse_count = 0;
    for (size_t c = 0; c < NUM_CLASSES; ++c)
    {
        size_t total_c = 0;
        for (size_t p = 0; p < NUM_CLASSES; ++p)
            total_c += confusion[c][p];
        double acc = (total_c > 0) ? 100.0 * confusion[c][c] / total_c : 0.0;
        if (acc < hardest_acc)
        {
            hardest_acc = acc;
            hardest = c;
        }
        for (size_t p = 0; p < NUM_CLASSES; ++p)
        {
            if (p != c && confusion[c][p] > confuse_count)
            {
                confuse_count = confusion[c][p];
                confuse_a = c;
                confuse_b = p;
            }
        }

        std::cout << "  " << CLASS_NAMES[c] << "  " << std::setprecision(1)
                  << std::setw(5) << acc << "%  (" << confusion[c][c]
                  << "/" << total_c << ")";
        if (acc >= 99.5)
            std::cout << "  -- near-perfect";
        else if (acc >= 95.0)
            std::cout << "  -- strong";
        else if (acc < 90.0)
        {
            size_t max_conf = 0, max_conf_class = c;
            for (size_t p = 0; p < NUM_CLASSES; ++p)
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
                std::cout << "  -- confused with " << CLASS_NAMES[max_conf_class]
                          << " " << std::setprecision(0) << conf_pct << "% of steps";
            }
        }
        std::cout << "\n";
    }

    std::cout << "\nConfusion matrix (rows=actual, cols=predicted; % of row):\n";
    std::cout << "               ";
    for (size_t c = 0; c < NUM_CLASSES; ++c)
        std::cout << CLASS_NAMES[c] << "  ";
    std::cout << "\n";
    for (size_t a = 0; a < NUM_CLASSES; ++a)
    {
        size_t total_a = 0;
        for (size_t p = 0; p < NUM_CLASSES; ++p)
            total_a += confusion[a][p];
        std::cout << "  " << CLASS_NAMES[a] << " |";
        for (size_t p = 0; p < NUM_CLASSES; ++p)
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
    const size_t margins[] = {3, 5, 10, 20, block_size};
    for (size_t margin : margins)
    {
        size_t ok = 0, total = 0;
        for (size_t b = 0; b < test_blocks; ++b)
        {
            const size_t off = b * block_size;
            for (size_t i = 0; i < margin && i < block_size; ++i)
            {
                ++total;
                if (predictions[off + i] == test_labels[off + i]) ++ok;
            }
        }
        double acc = (total > 0) ? 100.0 * ok / total : 0.0;
        if (margin == block_size)
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
    for (size_t b = 0; b < test_blocks; ++b)
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

    std::cout << "\nTime-to-lock (TTL = first step of " << lock_k
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

    std::cout << "  Capacity:  DIM=" << DIM << " (N=" << N << ").\n";
    std::cout << "             Overall step accuracy landed at " << std::setprecision(1)
              << overall_acc << "%";
    if (overall_acc >= 99.5)
        std::cout << " -- near ceiling on this stream.\n\n";
    else if (overall_acc >= 92.0)
        std::cout << " -- strong, with a residual hard pair.\n\n";
    else
        std::cout << " -- capacity/noise leave clear headroom to study.\n\n";

    std::cout << "  Hardest mode: " << CLASS_NAMES[hardest]
              << " (" << CLASS_SHAPES[hardest] << ") at "
              << std::setprecision(1) << hardest_acc << "% step accuracy.\n";
    if (confuse_count > 0 && confuse_a != confuse_b)
    {
        size_t row_tot = 0;
        for (size_t p = 0; p < NUM_CLASSES; ++p)
            row_tot += confusion[confuse_a][p];
        double pct = (row_tot > 0) ? 100.0 * confuse_count / row_tot : 0.0;
        std::cout << "  Top confusion: " << CLASS_NAMES[confuse_a]
                  << " -> " << CLASS_NAMES[confuse_b]
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
        std::cout << "             (need " << lock_k
                  << " correct in a row). Early-window accuracy (0-3 / 0-5)\n";
        std::cout << "             trails full-block accuracy -- residual state from\n";
        std::cout << "             the previous mode for a few ticks, then LOCKED.\n";
        if (no_lock > 0)
            std::cout << "             " << no_lock
                      << " switch block(s) never reached a " << lock_k
                      << "-step lock.\n";
    }
    std::cout << "\n";
    std::cout << "  Stream view: LOCKED means high block accuracy + confidence;\n";
    std::cout << "             SWITCHING means a mode change still washing through;\n";
    std::cout << "             SETTLING / CONFUSED / NO LOCK mark partial or failed locks.\n";

    std::cout << std::flush;
    return 0;
}
