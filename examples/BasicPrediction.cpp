/// @file BasicPrediction.cpp
/// @brief Simplest end-to-end demo: predict sin(t+1) from reservoir state.
/// See BasicPrediction.md for walkthrough and experiments.

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "ESN.h"

// =============================================================================
// ESN configuration — primary knobs for this demo (edit here)
// =============================================================================

static ESNConfig MakeESNConfig()
{
    ESNConfig cfg;

    // Reservoir (fixed dynamics)
    cfg.reservoir.dim             = 8;
    cfg.reservoir.seed            = 4112530987988204306ull;
    cfg.reservoir.history_depth   = 4;
    cfg.reservoir.spectral_radius = 0.9f;
    cfg.reservoir.input_scaling   = 0.1f;

    // Seam: delay-line ages packed into the readout input (power of two, ≤ M)
    cfg.readout_slices = 1;

    // Readout (trainable HCNN)
    cfg.readout.task        = ReadoutTask::Regression;
    cfg.readout.epochs      = 1500;
    cfg.readout.batch_size  = 16;
    cfg.readout.activation  = ReadoutActivation::TANH;  // TANH / RELU / LEAKY_RELU / NONE
    cfg.readout.use_pooling = true;

    return cfg;
}

// =============================================================================
// Demo / task parameters (not part of ESNConfig)
// =============================================================================

static constexpr size_t kWarmup         = 200;
static constexpr size_t kCollect        = 2000;
static constexpr size_t kHorizon        = 1;
static constexpr double kTrainFraction  = 0.7;
static constexpr size_t kSamplePreds    = 10;

// =============================================================================

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    const ESNConfig cfg = MakeESNConfig();
    ESN esn(cfg);

    std::cout << "=== HypercubeESN: Sine Wave Prediction ===\n\n";
    std::cout << "Task: predict the next value of sin(0.1t) from the reservoir's\n";
    std::cout << "internal state.  The readout never sees the input directly -- it\n";
    std::cout << "learns the input-to-output mapping entirely from reservoir dynamics.\n\n";

    const size_t total = kWarmup + kCollect + kHorizon;
    std::vector<float> signal(total);
    for (size_t t = 0; t < total; ++t)
        signal[t] = std::sin(0.1f * static_cast<float>(t));

    std::vector<float> targets(kCollect);
    for (size_t t = 0; t < kCollect; ++t)
        targets[t] = signal[kWarmup + t + kHorizon];

    const size_t train_size = static_cast<size_t>(kCollect * kTrainFraction);
    const size_t test_size  = kCollect - train_size;
    const float  lr_floor   = cfg.readout.lr_max * cfg.readout.lr_min_frac;

    // Demo schedule (not in ReadoutArchSummary) + architecture once.
    std::cout << "Stream:  warmup=" << kWarmup
              << "  collect=" << kCollect
              << "  train=" << train_size
              << "  test=" << test_size
              << "  horizon=" << kHorizon << "\n";
    std::cout << "Train:   " << cfg.readout.epochs << " epochs"
              << "  batch=" << cfg.readout.batch_size
              << "  lr_max=" << cfg.readout.lr_max
              << "  (cosine, floor=" << lr_floor << ")\n\n";
    std::cout << esn.ReadoutArchSummary() << "\n";

    esn.ReservoirWarmup(signal.data(), kWarmup);
    esn.ReservoirRun(signal.data() + kWarmup, train_size + test_size);

    std::cout << "Training on " << train_size << " steps..." << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    esn.Train(targets.data(), train_size);
    auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(1) << secs << "s)\n\n";

    const double r2    = esn.R2(targets.data(), train_size, test_size);
    const double nrmse = esn.NRMSE(targets.data(), train_size, test_size);

    std::cout << "--- Results (held-out test, " << test_size << " steps) ---\n\n";
    std::cout << "R2:    " << std::fixed << std::setprecision(6) << r2;
    if (r2 > 0.9999) std::cout << "  (effectively perfect)";
    else if (r2 > 0.99) std::cout << "  (excellent)";
    else if (r2 > 0.9) std::cout << "  (good)";
    std::cout << "\n";
    std::cout << "NRMSE: " << std::setprecision(6) << nrmse;
    if (nrmse < 0.001) std::cout << "  (sub-0.1% error)";
    else if (nrmse < 0.01) std::cout << "  (under 1% error)";
    else if (nrmse < 0.1) std::cout << "  (under 10% error)";
    std::cout << "\n\n";

    std::cout << "Sample predictions (first " << kSamplePreds
              << " test steps, indices " << train_size
              << ".." << (train_size + kSamplePreds - 1) << "):\n\n";
    std::cout << "  Step  |   Actual   |  Predicted  |    Error\n";
    std::cout << "  ------+------------+-------------+-----------\n";
    double sum_abs_err = 0.0;
    for (size_t i = 0; i < kSamplePreds; ++i)
    {
        const float actual    = targets[train_size + i];
        const float predicted = esn.PredictFromRecorded(train_size + i)[0];
        const float error     = actual - predicted;
        sum_abs_err += static_cast<double>(std::fabs(error));
        std::cout << "  " << std::setw(5) << (train_size + i)
                  << " | " << std::showpos << std::setprecision(5) << std::setw(10) << actual
                  << " | " << std::setw(11) << predicted
                  << " | " << std::setw(10) << error
                  << std::noshowpos << "\n";
    }
    const double mean_abs_err = sum_abs_err / static_cast<double>(kSamplePreds);
    std::cout << "  ------+------------+-------------+-----------\n";
    std::cout << std::fixed << std::setprecision(6)
              << "  Mean |error| over " << kSamplePreds << " samples: "
              << mean_abs_err << "\n";

    std::cout << "\nThe HCNN readout learned sin(t+1) from the reservoir's dynamics,\n";
    std::cout << "discovering features via convolution on the hypercube state.\n";

    return 0;
}
