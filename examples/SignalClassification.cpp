/// @file SignalClassification.cpp
/// @brief Multi-class waveform classification from reservoir state.
/// See SignalClassification.md for walkthrough and experiments.

#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <cmath>
#include "ESN.h"

static constexpr float PI = 3.14159265358979323846f;
static constexpr size_t NUM_CLASSES = 4;
static const char* CLASS_NAMES[NUM_CLASSES] = {"Sine    ", "Square  ", "Triangle", "Chirp   "};

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

    default:
        return 0.0f;
    }
}

static constexpr float CLASS_FREQ[NUM_CLASSES] = { 0.11f, 0.13f, 0.12f, 0.10f };
static constexpr float NOISE_LEVEL = 0.15f;

/// Print overall accuracy, per-class breakdown, confusion matrix, and lock-on
/// table (accuracy as a function of steps since the last block transition).
static double analyze_and_print(const std::vector<size_t>& predictions,
                                const size_t* test_labels,
                                size_t test_size,
                                size_t train_size,
                                size_t block_size)
{
    std::cout << "--- Results (test set: " << test_size << " samples) ---\n\n";

    size_t confusion[NUM_CLASSES][NUM_CLASSES] = {};
    size_t correct = 0;
    for (size_t t = 0; t < test_size; ++t)
    {
        confusion[test_labels[t]][predictions[t]]++;
        if (test_labels[t] == predictions[t])
            correct++;
    }

    double accuracy = 100.0 * static_cast<double>(correct) / static_cast<double>(test_size);

    std::cout << "Overall accuracy: " << std::fixed << std::setprecision(1)
              << accuracy << "%\n\n";

    std::cout << "Per-class breakdown:\n";
    for (size_t c = 0; c < NUM_CLASSES; ++c)
    {
        size_t total_c = 0;
        for (size_t p = 0; p < NUM_CLASSES; ++p)
            total_c += confusion[c][p];
        double acc = (total_c > 0) ? 100.0 * confusion[c][c] / total_c : 0.0;
        std::cout << "  " << CLASS_NAMES[c] << "  " << std::setprecision(1)
                  << std::setw(5) << acc << "%  (" << confusion[c][c]
                  << "/" << total_c << ")";

        if (acc >= 99.9)
            std::cout << "  -- perfectly separable";
        else if (acc >= 98.0)
            std::cout << "  -- near-perfect";
        else if (acc < 90.0)
        {
            size_t max_conf = 0;
            size_t max_conf_class = c;
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
                          << " " << std::setprecision(0) << conf_pct << "% of the time";
            }
        }
        std::cout << "\n";
    }

    std::cout << "\nConfusion matrix (rows=actual, cols=predicted):\n";
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

    std::cout << "\nLock-on dynamics (steps after block transition):\n";
    std::cout << "  Steps after switch  | Accuracy\n";
    std::cout << "  --------------------+---------\n";

    const size_t margins[] = {3, 5, 10, 20, block_size};
    for (size_t margin : margins)
    {
        size_t ok = 0, total = 0;

        for (size_t t = 0; t < test_size; ++t)
        {
            size_t global_t = train_size + t;
            size_t pos_in_block = global_t % block_size;

            if (pos_in_block < margin)
            {
                total++;
                if (predictions[t] == test_labels[t]) ok++;
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
    std::cout << "\n";
    return accuracy;
}

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;

    constexpr size_t DIM = 8;
    constexpr size_t N = 1ULL << DIM;
    constexpr size_t warmup = 300;
    constexpr size_t block_size = 40;
    constexpr size_t num_cycles = 75;
    constexpr size_t collect = block_size * NUM_CLASSES * num_cycles;
    constexpr double train_fraction = 0.7;

    constexpr uint64_t seed = 123456;
    std::cout << "=== HypercubeESN: Signal Classification ===\n\n";
    std::cout << "Task: identify which waveform is currently being fed to the reservoir,\n";
    std::cout << "using only the reservoir's internal state -- not the input directly.\n\n";
    std::cout << "Four waveforms cycle in blocks of " << block_size << " steps:\n";
    std::cout << "  Sine (f=0.11)  |  Square (f=0.13)  |  Triangle (f=0.12)  |  Chirp (f=0.10)\n";
    std::cout << "Frequencies are deliberately close; " << NOISE_LEVEL
              << " noise forces classification by shape, not frequency.\n";

    std::vector<float> signal(warmup + collect);
    std::vector<size_t> labels(collect);

    std::mt19937_64 noise_rng(seed);
    std::uniform_real_distribution<float> noise(-NOISE_LEVEL, NOISE_LEVEL);

    for (size_t t = 0; t < warmup; ++t)
        signal[t] = GenerateWaveform(0, CLASS_FREQ[0] * static_cast<float>(t))
                    + noise(noise_rng);

    for (size_t t = 0; t < collect; ++t)
    {
        size_t block_idx = t / block_size;
        size_t waveform = block_idx % NUM_CLASSES;
        size_t t_in_block = t % block_size;

        labels[t] = waveform;
        float phase = CLASS_FREQ[waveform] * static_cast<float>(t_in_block);
        signal[warmup + t] = GenerateWaveform(waveform, phase) + noise(noise_rng);
    }

    size_t train_size = static_cast<size_t>(collect * train_fraction);
    size_t test_size = collect - train_size;
    const size_t* test_labels = labels.data() + train_size;

    ESNConfig cfg;
    cfg.reservoir.dim         = DIM;
    cfg.reservoir.input_scaling = 1.5;
    cfg.readout.num_outputs   = NUM_CLASSES;
    cfg.readout.task          = ReadoutTask::Classification;
    cfg.readout.epochs        = 50;
    cfg.readout.activation    = ReadoutActivation::TANH;  // TANH / RELU / LEAKY_RELU / NONE
    ESN esn(cfg);

    std::cout << "Config: DIM=" << DIM << "  N=" << N << "  History Depth=" << cfg.reservoir.history_depth << "  Input Scaling=" << cfg.reservoir.input_scaling
              << "  Task=Classification  Classes=" << NUM_CLASSES << "\n";

    esn.Warmup(signal.data(), warmup);
    esn.Run(signal.data() + warmup, collect);

    std::vector<float> float_labels(collect);
    for (size_t t = 0; t < collect; ++t)
        float_labels[t] = static_cast<float>(labels[t]);

    std::cout << "Training: " << cfg.readout.epochs << " epochs, batch=" << cfg.readout.batch_size
              << ", lr_max=" << std::setprecision(4) << cfg.readout.lr_max
              << " (cosine floor " << cfg.readout.lr_max * cfg.readout.lr_min_frac << ")\n";
    std::cout << "Training..." << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    esn.Train(float_labels.data(), train_size);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(1) << secs << "s)\n\n";

    std::vector<size_t> predictions(test_size);
    std::vector<float> logits(NUM_CLASSES);
    for (size_t t = 0; t < test_size; ++t)
    {
        esn.PredictRaw(train_size + t, logits.data());
        size_t predicted = 0;
        float best = logits[0];
        for (size_t c = 1; c < NUM_CLASSES; ++c)
        {
            if (logits[c] > best)
            {
                best = logits[c];
                predicted = c;
            }
        }
        predictions[t] = predicted;
    }

    analyze_and_print(predictions, test_labels, test_size, train_size, block_size);

    std::cout << "The HCNN multi-class readout classifies waveforms from the same\n";
    std::cout << "reservoir dynamics, discovering features via convolution on raw state.\n";

    std::cout << std::flush;
    return 0;
}
