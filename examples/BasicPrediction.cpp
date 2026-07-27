/// @file BasicPrediction.cpp
/// @brief Simplest end-to-end demo: predict sin(t+1) from reservoir state.
/// See BasicPrediction.md for walkthrough and experiments.

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "ESN.h"

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;

    constexpr size_t DIM = 8;
    constexpr size_t N = 1ULL << DIM;
    constexpr size_t warmup = 200;
    constexpr size_t collect = 2000;
    constexpr size_t horizon = 1;
    constexpr double train_fraction = 0.7;

    std::cout << "=== HypercubeESN: Sine Wave Prediction ===\n\n";
    std::cout << "Task: predict the next value of sin(0.1t) from the reservoir's\n";
    std::cout << "internal state.  The readout never sees the input directly -- it\n";
    std::cout << "learns the input-to-output mapping entirely from reservoir dynamics.\n\n";

    const size_t total = warmup + collect + horizon;
    std::vector<float> signal(total);
    for (size_t t = 0; t < total; ++t)
        signal[t] = std::sin(0.1f * static_cast<float>(t));

    std::vector<float> targets(collect);
    for (size_t t = 0; t < collect; ++t)
        targets[t] = signal[warmup + t + horizon];

    size_t train_size = static_cast<size_t>(collect * train_fraction);
    size_t test_size = collect - train_size;

    ESNConfig cfg;
    cfg.reservoir.dim         = DIM;
    cfg.reservoir.seed = 84745874578;
    cfg.reservoir.history_depth = 4;
    cfg.readout_slices = 1;
    cfg.reservoir.spectral_radius = 0.9;
    cfg.reservoir.input_scaling = 0.1;
    // Full-state linear feedback (internal). Edit these three for A/B.
    cfg.reservoir.full_state_feedback = false;
    cfg.reservoir.fsf_seed = 4415756;
    cfg.reservoir.fsf_scaling = 0.003f;
    cfg.readout.task          = ReadoutTask::Regression;
    cfg.readout.epochs        = 1500;
    cfg.readout.batch_size    = 16;
    cfg.readout.activation    = ReadoutActivation::TANH;  // TANH / RELU / LEAKY_RELU / NONE
    cfg.readout.use_pooling = true;
    ESN esn(cfg);

    std::cout << "  Config: N=" << N << "  history_depth=" << cfg.reservoir.history_depth << "\n";
    if (cfg.reservoir.full_state_feedback)
        std::cout << "  FSF: ON   fsf_seed=" << cfg.reservoir.fsf_seed
                  << "  fsf_scaling=" << cfg.reservoir.fsf_scaling << "\n";
    else
        std::cout << "  FSF: OFF\n";
    std::cout << "  Readout in: " << esn.ReadoutBlockCount() << " block(s) x " << N
              << " = " << esn.ReadoutInputWidth() << " values"
              << "  (slices=" << cfg.readout_slices
              << ", aux=" << cfg.aux_input_dim
              << ", pooling=" << (cfg.readout.use_pooling ? "on" : "off") << ")\n";
    std::cout << "  Training: " << cfg.readout.epochs << " epochs, batch=" << cfg.readout.batch_size
              << ", lr=" << cfg.readout.lr_max
              << " (cosine, floor=" << (cfg.readout.lr_max * cfg.readout.lr_min_frac) << ")\n";
    std::cout << esn.ReadoutArchSummary();

    esn.ReservoirWarmup(signal.data(), warmup);
    esn.ReservoirRun(signal.data() + warmup, train_size + test_size);

    std::cout << "  Training..." << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    esn.Train(targets.data(), train_size);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(1) << secs << "s)\n";

    double r2 = esn.R2(targets.data(), train_size, test_size);
    double nrmse = esn.NRMSE(targets.data(), train_size, test_size);

    std::cout << "  R2:    " << std::fixed << std::setprecision(6) << r2;
    if (r2 > 0.9999) std::cout << "  (effectively perfect)";
    else if (r2 > 0.99) std::cout << "  (excellent)";
    else if (r2 > 0.9) std::cout << "  (good)";
    std::cout << "\n";
    std::cout << "  NRMSE: " << std::setprecision(6) << nrmse;
    if (nrmse < 0.001) std::cout << "  (sub-0.1% error)";
    else if (nrmse < 0.01) std::cout << "  (under 1% error)";
    else if (nrmse < 0.1) std::cout << "  (under 10% error)";
    std::cout << "\n\n";

    constexpr size_t n_samples = 10;
    std::cout << "Sample predictions (test set):\n\n";
    std::cout << "  Step  |   Actual   |  Predicted  |    Error\n";
    std::cout << "  ------+------------+-------------+-----------\n";
    double sum_abs_err = 0.0;
    for (size_t i = 0; i < n_samples; ++i)
    {
        float actual = targets[train_size + i];
        float predicted = esn.PredictFromRecorded(train_size + i)[0];
        float error = actual - predicted;
        sum_abs_err += static_cast<double>(std::fabs(error));
        std::cout << "  " << std::setw(5) << (train_size + i)
                  << " | " << std::showpos << std::setprecision(5) << std::setw(10) << actual
                  << " | " << std::setw(11) << predicted
                  << " | " << std::setw(10) << error
                  << std::noshowpos << "\n";
    }
    const double mean_abs_err = sum_abs_err / static_cast<double>(n_samples);
    std::cout << "  ------+------------+-------------+-----------\n";
    std::cout << std::fixed << std::setprecision(6)
              << "  Mean |error| over " << n_samples << " samples: "
              << mean_abs_err << "\n";

    std::cout << "\nThe HCNN readout learned sin(t+1) from the reservoir's dynamics,\n";
    std::cout << "discovering features via convolution on the hypercube state.\n";

    return 0;
}

