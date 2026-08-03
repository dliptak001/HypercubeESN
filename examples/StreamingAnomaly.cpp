/// @file StreamingAnomaly.cpp
/// @brief Anomaly detection: learn normal behavior, flag deviations.
/// See StreamingAnomaly.md for walkthrough and experiments.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "ESN.h"

// =============================================================================
// ESN configuration — primary knobs for this demo (edit here)
// =============================================================================

static ESNConfig MakeESNConfig()
{
    ESNConfig cfg;

    // Reservoir (fixed dynamics)
    cfg.reservoir.dim             = 7;
    cfg.reservoir.seed            = 7934791766227647176;
    cfg.reservoir.history_depth   = 16;
    cfg.reservoir.spectral_radius = 0.985f;
    cfg.reservoir.input_scaling   = 0.1f;

    // Seam: delay-line ages packed into the readout input (power of two, ≤ M)
    cfg.readout_slices = 1;

    // Readout (trainable HCNN)
    cfg.readout.task          = ReadoutTask::Regression;
    cfg.readout.epochs        = 500;
    cfg.readout.batch_size    = 64;
    cfg.readout.momentum      = 0.9f;
    cfg.readout.activation    = ReadoutActivation::TANH;  // TANH / RELU / LEAKY_RELU / NONE
    cfg.readout.num_layers    = 1;
    cfg.readout.conv_channels = 8;
    cfg.readout.use_pooling   = true;

    return cfg;
}

// =============================================================================
// Demo / task parameters (not part of ESNConfig)
// =============================================================================

static constexpr size_t kWarmup            = 500;
static constexpr size_t kPrimeSteps        = 4000;
static constexpr size_t kWindow            = 200;
static constexpr float  kNormalNoise       = 0.01f;
static constexpr float  kAnomalyThreshold  = 10.0f;  // × baseline RMSE
static constexpr uint64_t kStreamSeed      = 123456;

// =============================================================================
// Helpers
// =============================================================================

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

/// Per-window monitor result (feeds the data-driven "What happened" epilogue).
struct WindowResult
{
    Event evt{};
    double ratio = 0.0;
    bool flagged = false;
};

static bool IsNormalEvent(const Event& e)
{
    return e.noise == kNormalNoise && e.drift == 0.0f && e.freq == 1.0f;
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

    std::mt19937_64 signal_rng(kStreamSeed);

    const Event normal    = {"Normal     ", 0.01f, 0.0f, 1.0f};
    const Event spike     = {"Noise spike", 0.12f, 0.0f, 1.0f};
    const Event drift_evt = {"DC drift   ", 0.01f, 0.30f, 1.0f};
    const Event freq_evt  = {"Freq shift ", 0.01f, 0.0f, 1.3f};

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

    std::cout << "Config: DIM=" << dim << "  N=" << N
              << "  History Depth=" << cfg.reservoir.history_depth
              << "  Leak=" << cfg.reservoir.leak_rate
              << "  Input Scaling=" << cfg.reservoir.input_scaling
              << "  Threshold=" << kAnomalyThreshold << "x baseline\n";
    std::cout << esn.ReadoutArchSummary();
    std::cout << "\n";

    std::cout << "--- Phase 1: Learn what \"normal\" looks like ---\n\n";

    size_t t_global = 0;
    std::vector<float> prime_signal(kWarmup + kPrimeSteps + 1);
    GenerateProcess(prime_signal.data(), prime_signal.size(), t_global,
                    kNormalNoise, 0.0f, 1.0f, signal_rng);

    constexpr size_t train_n = static_cast<size_t>(kPrimeSteps * 0.7);
    constexpr size_t test_n  = kPrimeSteps - train_n;

    esn.ReservoirWarmup(prime_signal.data(), kWarmup);
    esn.ReservoirRun(prime_signal.data() + kWarmup, train_n + test_n);  // training + baseline states
    t_global += kWarmup + kPrimeSteps;

    std::vector<float> prime_targets(kPrimeSteps);
    for (size_t t = 0; t < kPrimeSteps; ++t)
        prime_targets[t] = prime_signal[kWarmup + t + 1];

    std::cout << "Training on " << train_n << " samples ("
              << cfg.readout.epochs << " epochs, batch=" << cfg.readout.batch_size
              << ", lr_max=" << std::setprecision(4) << cfg.readout.lr_max << ")..."
              << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    esn.Train(prime_targets.data(), train_n);
    auto t1 = std::chrono::steady_clock::now();
    double train_secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(2) << train_secs << "s)\n\n";

    std::vector<float> prime_pred(test_n);
    for (size_t i = 0; i < test_n; ++i)
        prime_pred[i] = esn.PredictFromRecorded(train_n + i)[0];
    double baseline  = ComputeRMSE(prime_pred.data(), prime_targets.data() + train_n, test_n);
    double threshold = baseline * kAnomalyThreshold;

    std::cout << "Baseline (prime test, RMSE): " << std::setprecision(6) << baseline
              << "   threshold " << threshold << "\n\n";

    std::cout << "--- Phase 2: Monitor the process (" << schedule.size()
              << " windows of " << kWindow << " steps) ---\n\n";
    std::cout << "Each window is fed to the reservoir, and the readout predicts\n";
    std::cout << "the next value.  An RMSE above " << std::setprecision(0) << kAnomalyThreshold
              << "x baseline is flagged.\n\n";

    std::cout << "  Window | Condition          |    RMSE     Ratio | Status\n";
    std::cout << "  -------+--------------------+-------------------+---------\n";

    size_t flags = 0;
    std::vector<WindowResult> results;
    results.reserve(schedule.size());

    for (size_t w = 0; w < schedule.size(); ++w)
    {
        const Event& evt = schedule[w];

        std::vector<float> sig(kWindow + 1);
        GenerateProcess(sig.data(), sig.size(), t_global,
                        evt.noise, evt.drift, evt.freq, signal_rng);
        t_global += kWindow;

        std::vector<float> tgt(kWindow);
        for (size_t t = 0; t < kWindow; ++t)
            tgt[t] = sig[t + 1];

        esn.ReservoirRun(sig.data(), kWindow, /*clear_recorded=*/true);

        std::vector<float> pred(kWindow);
        for (size_t t = 0; t < kWindow; ++t)
            pred[t] = esn.PredictFromRecorded(t)[0];
        double rmse  = ComputeRMSE(pred.data(), tgt.data(), kWindow);
        double ratio = rmse / baseline;
        bool anom    = (rmse > threshold);
        if (anom) ++flags;

        results.push_back(WindowResult{evt, ratio, anom});

        char status[16] = "";
        if (anom) std::snprintf(status, sizeof(status), "** ANOMALY **");

        std::cout << "  " << std::setw(5) << (w + 1)
                  << "  | " << evt.label << "        "
                  << " | " << std::fixed << std::setprecision(6) << std::setw(10) << rmse
                  << "  " << std::setprecision(1) << std::setw(5) << ratio
                  << " | " << status << "\n";
    }

    // Count anomaly-condition windows vs washout flags (flagged normals after a burst).
    size_t anomaly_condition_flags = 0;
    size_t washout_flags = 0;
    for (size_t w = 0; w < results.size(); ++w)
    {
        if (!results[w].flagged) continue;
        if (IsNormalEvent(results[w].evt))
            ++washout_flags;
        else
            ++anomaly_condition_flags;
    }

    std::cout << "\nFlagged windows: " << flags
              << "  (" << anomaly_condition_flags << " anomaly-condition + "
              << washout_flags << " washout on recovering \"normal\")\n\n";

    // ---- What happened (ratios from this run; no hardcoded ratios) ----
    std::cout << "--- What happened ---\n\n";
    std::cout << "The reservoir learned to predict normal process output during priming.\n";
    std::cout << "During monitoring, prediction error is the anomaly signal:\n\n";

    // Walk contiguous non-normal bursts; for each, report ratio range + first recovery window.
    size_t w = 0;
    while (w < results.size())
    {
        if (IsNormalEvent(results[w].evt))
        {
            ++w;
            continue;
        }

        const char* kind = results[w].evt.label;
        double r_min = results[w].ratio;
        double r_max = results[w].ratio;
        size_t burst_end = w;
        while (burst_end < results.size() && !IsNormalEvent(results[burst_end].evt)
               && results[burst_end].evt.label == kind)
        {
            r_min = std::min(r_min, results[burst_end].ratio);
            r_max = std::max(r_max, results[burst_end].ratio);
            ++burst_end;
        }

        // First normal window after the burst (washout / recovery probe).
        const bool has_recovery = (burst_end < results.size() && IsNormalEvent(results[burst_end].evt));
        const double recovery_ratio = has_recovery ? results[burst_end].ratio : 0.0;
        const bool recovery_flagged = has_recovery && results[burst_end].flagged;

        // Trim trailing spaces from the schedule label for prose.
        std::string name = kind;
        while (!name.empty() && name.back() == ' ')
            name.pop_back();

        std::cout << "  " << name << ":  RMSE ratio "
                  << std::fixed << std::setprecision(1)
                  << r_min << "x - " << r_max << "x during the burst";

        if (name == "Noise spike")
        {
            std::cout << " -- random disturbance\n";
            std::cout << "                is unpredictable.";
            if (has_recovery)
            {
                std::cout << " Recovery is essentially instant: the\n";
                std::cout << "                first normal window is only mildly elevated ("
                          << std::setprecision(1) << recovery_ratio << "x)";
                if (recovery_flagged)
                    std::cout << " and flagged.\n";
                else
                    std::cout << ",\n"
                              << "                under the " << std::setprecision(0)
                              << kAnomalyThreshold << "x threshold (not flagged).\n";
            }
            else
            {
                std::cout << "\n";
            }
        }
        else if (name == "DC drift")
        {
            std::cout << " -- the model never saw this offset.\n";
            std::cout << "                Structured bias error.";
            if (has_recovery)
            {
                std::cout << " First residual normal window sits at "
                          << std::setprecision(1) << recovery_ratio << "x";
                if (recovery_flagged)
                    std::cout << " (flagged washout).\n";
                else
                    std::cout << " (under the " << std::setprecision(0)
                              << kAnomalyThreshold << "x threshold, not flagged),\n"
                              << "                then baseline.\n";
            }
            else
            {
                std::cout << "\n";
            }
        }
        else if (name == "Freq shift")
        {
            std::cout << " -- changed dynamics break the pattern.\n";
            if (has_recovery)
            {
                if (recovery_flagged)
                {
                    std::cout << "                Slowest recovery: first \"normal\" window is still "
                              << std::setprecision(1) << recovery_ratio
                              << "x and flagged while\n"
                              << "                recurrent/delay-line state washes out. That washout\n"
                              << "                flag is expected, not a false positive on a new fault.\n";
                }
                else
                {
                    std::cout << "                First normal window recovers to "
                              << std::setprecision(1) << recovery_ratio
                              << "x (under threshold).\n";
                }
            }
            else
            {
                std::cout << "\n";
            }
        }
        else
        {
            std::cout << ".\n";
            if (has_recovery)
            {
                std::cout << "                First normal after: " << std::setprecision(1)
                          << recovery_ratio << "x"
                          << (recovery_flagged ? " (flagged).\n" : " (not flagged).\n");
            }
        }
        std::cout << "\n";

        w = burst_end;
    }

    std::cout << "  Totals:     " << flags << " flagged window(s) = "
              << anomaly_condition_flags << " during anomalies + "
              << washout_flags << " washout.\n";

    return 0;
}
