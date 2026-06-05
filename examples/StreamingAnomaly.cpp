/// @file StreamingAnomaly.cpp
/// @brief Anomaly detection: learn normal behavior, flag deviations.
/// See StreamingAnomaly.md for walkthrough and experiments.

#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include "ESN.h"

/// Two-harmonic process signal (0.6*sin + 0.2*sin(3x)) with adjustable noise
/// level, DC offset, and frequency scale -- the knobs the anomalies modulate.
static void GenerateProcess(float* out, size_t n, size_t t_start,
                            float noise_level, float dc_drift, float freq_mult,
                            std::mt19937_64& rng)
{
    std::uniform_real_distribution<float> noise(-1.0f, 1.0f);
    for (size_t t = 0; t < n; ++t)
    {
        float phase = 0.1f * freq_mult * static_cast<float>(t_start + t);
        float clean = 0.6f * std::sin(phase) + 0.2f * std::sin(3.0f * phase);
        out[t] = clean + dc_drift + noise_level * noise(rng);
    }
}

static double ComputeRMSE(const float* pred, const float* targets, size_t n)
{
    double mse = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        double err = targets[i] - pred[i];
        mse += err * err;
    }
    return std::sqrt(mse / n);
}

/// One row of the monitoring schedule: process parameters for a single window.
struct Event
{
    const char* label;
    float noise;
    float drift;
    float freq;
};

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    constexpr size_t DIM = 8;
    constexpr size_t N = 1ULL << DIM;
    constexpr size_t warmup = 500;
    constexpr size_t prime_steps = 4000;
    constexpr size_t window = 200;
    constexpr float normal_noise = 0.01f;
    constexpr float anomaly_threshold = 10.0f;

    constexpr uint64_t seed = 123456;
    std::mt19937_64 signal_rng(seed);

    const Event normal = {"Normal     ", 0.01f, 0.0f, 1.0f};
    const Event spike = {"Noise spike", 0.12f, 0.0f, 1.0f};
    const Event drift_evt = {"DC drift   ", 0.01f, 0.30f, 1.0f};
    const Event freq_evt = {"Freq shift ", 0.01f, 0.0f, 1.3f};

    std::vector<Event> schedule;
    for (size_t i = 0; i < 5; ++i) schedule.push_back(normal);
    for (size_t i = 0; i < 3; ++i) schedule.push_back(spike);
    for (size_t i = 0; i < 5; ++i) schedule.push_back(normal);
    for (size_t i = 0; i < 3; ++i) schedule.push_back(drift_evt);
    for (size_t i = 0; i < 5; ++i) schedule.push_back(normal);
    for (size_t i = 0; i < 3; ++i) schedule.push_back(freq_evt);
    for (size_t i = 0; i < 6; ++i) schedule.push_back(normal);

    std::cout << "=== HypercubeESN: Streaming Anomaly Detection ===\n\n";
    std::cout << "Scenario: an industrial process produces a multi-harmonic signal.\n";
    std::cout << "The reservoir learns the normal pattern, then monitors for deviations.\n";
    std::cout << "Three types of anomaly are injected, each for 3 windows, separated\n";
    std::cout << "by normal operation to show both detection and recovery.\n\n";

    std::cout << "Anomaly types:\n";
    std::cout << "  1. Noise spike   -- sensor noise jumps 12x (0.01 -> 0.12)\n";
    std::cout << "  2. DC drift      -- systematic +0.30 offset (e.g. sensor fouling)\n";
    std::cout << "  3. Freq shift    -- process speed changes to 1.3x (e.g. motor issue)\n\n";

    ESNConfig cfg;
    cfg.reservoir.dim = DIM;
    cfg.reservoir.input_scaling = 1.9;
    cfg.readout.task = ReadoutTask::Regression;
    cfg.readout.epochs = 1000;
    cfg.readout.momentum = 0.9;
    cfg.readout.conv_channels = 8;
    cfg.readout.batch_size = 64;
    cfg.readout.activation = ReadoutActivation::TANH; // TANH / RELU / LEAKY_RELU / NONE
    ESN esn(cfg);

    std::cout << "Config: DIM=" << DIM << "  N=" << N << "  History Depth=" << cfg.reservoir.history_depth
        << "  Leak=" << cfg.reservoir.leak_rate
        << "  Input Scaling=" << cfg.reservoir.input_scaling
        << "  Threshold=" << anomaly_threshold << "x baseline\n\n";

    std::cout << "--- Phase 1: Learn what \"normal\" looks like ---\n\n";

    size_t t_global = 0;
    std::vector<float> prime_signal(warmup + prime_steps + 1);
    GenerateProcess(prime_signal.data(), prime_signal.size(), t_global,
                    normal_noise, 0.0f, 1.0f, signal_rng);

    esn.Warmup(prime_signal.data(), warmup);
    esn.Run(prime_signal.data() + warmup, prime_steps);
    t_global += warmup + prime_steps;

    std::vector<float> prime_targets(prime_steps);
    for (size_t t = 0; t < prime_steps; ++t)
        prime_targets[t] = prime_signal[warmup + t + 1];

    size_t train_n = static_cast<size_t>(prime_steps * 0.7);
    size_t test_n = prime_steps - train_n;

    std::cout << "Training on " << train_n << " samples ("
        << cfg.readout.epochs << " epochs, batch=" << cfg.readout.batch_size
        << ", lr_max=" << std::setprecision(4) << cfg.readout.lr_max << ")..." << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    esn.Train(prime_targets.data(), train_n);
    auto t1 = std::chrono::steady_clock::now();
    double train_secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(2) << train_secs << "s)\n\n";

    std::vector<float> prime_pred(test_n);
    for (size_t i = 0; i < test_n; ++i)
        prime_pred[i] = esn.PredictRaw(train_n + i);
    double baseline = ComputeRMSE(prime_pred.data(),
                                  prime_targets.data() + train_n, test_n);
    double threshold = baseline * anomaly_threshold;

    std::cout << "Baseline (prime test, RMSE): " << std::setprecision(6) << baseline
        << "   threshold " << threshold << "\n\n";

    std::cout << "--- Phase 2: Monitor the process (" << schedule.size()
        << " windows of " << window << " steps) ---\n\n";
    std::cout << "Each window is fed to the reservoir, and the readout predicts\n";
    std::cout << "the next value.  An RMSE above " << std::setprecision(0) << anomaly_threshold
        << "x baseline is flagged.\n\n";

    std::cout << "  Window | Condition          |    RMSE     Ratio | Status\n";
    std::cout << "  -------+--------------------+-------------------+---------\n";

    size_t flags = 0;

    for (size_t w = 0; w < schedule.size(); ++w)
    {
        const Event& evt = schedule[w];

        std::vector<float> sig(window + 1);
        GenerateProcess(sig.data(), sig.size(), t_global,
                        evt.noise, evt.drift, evt.freq, signal_rng);
        t_global += window;

        std::vector<float> tgt(window);
        for (size_t t = 0; t < window; ++t)
            tgt[t] = sig[t + 1];

        esn.ClearStates();
        esn.Run(sig.data(), window);

        std::vector<float> pred(window);
        for (size_t t = 0; t < window; ++t)
            pred[t] = esn.PredictRaw(t);
        double rmse = ComputeRMSE(pred.data(), tgt.data(), window);
        double ratio = rmse / baseline;
        bool anom = (rmse > threshold);
        if (anom) ++flags;

        char status[16] = "";
        if (anom) std::snprintf(status, sizeof(status), "** ANOMALY **");

        std::cout << "  " << std::setw(5) << (w + 1)
            << "  | " << evt.label << "        "
            << " | " << std::fixed << std::setprecision(6) << std::setw(10) << rmse
            << "  " << std::setprecision(1) << std::setw(5) << ratio
            << " | " << status << "\n";
    }

    std::cout << "\nFlagged windows: " << flags
        << "  (expected 10 = 9 anomaly windows + 1 washout window\n"
        << "                    where the leaky integrator is still ringing)\n\n";

    std::cout << "--- What happened ---\n\n";
    std::cout << "The reservoir learned to predict normal process output during priming.\n";
    std::cout << "During monitoring, prediction error is the anomaly signal:\n\n";
    std::cout << "  Noise spike:  RMSE jumps ~12x -- random disturbance is unpredictable.\n";
    std::cout << "                Recovery is instant (next normal window back to baseline).\n\n";
    std::cout << "  DC drift:     RMSE rises dramatically -- the model didn't learn this offset.\n";
    std::cout << "                The leaky integrator compounds the error across steps.\n";
    std::cout << "                Takes 1-2 windows to wash out after recovery.\n\n";
    std::cout << "  Freq shift:   RMSE spikes -- changed dynamics break the learned pattern.\n";
    std::cout << "                Slowest recovery: reservoir needs 1-2 extra windows to\n";
    std::cout << "                wash out the altered frequency from its internal state.\n";

    return 0;
}
