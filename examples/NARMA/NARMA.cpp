/// @file NARMA.cpp
/// @brief NARMA-N system-identification benchmark on the hypercube reservoir.
/// See NARMA.md for the walkthrough, the recurrence, and reference bands.
///
/// NARMA (Nonlinear Auto-Regressive Moving Average) is the classic reservoir
/// stress test: reproduce y(t) from a white input u(t) when y depends on a
/// long nonlinear history of itself and on the delayed input u(t-N). It probes
/// memory depth and nonlinear mixing at once.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "ESN.h"
#include "NARMA_N.h"

// ---- A/B compile switch ------------------------------------------------------
// 0 = legacy NARMA: no squashing; the order-dependent beta/delta schedule keeps
//     it bounded but weakens the nonlinearity as order grows, so high-order
//     series get smoother/easier -- difficulty does NOT scale with order.
// 1 = tanh-wrapped NARMA: the outer tanh bounds y(t), so coefficients stay fixed
//     at every order and difficulty scales honestly with order (the standard
//     literature form for NARMA-20/30).
// Override at build time with -DNARMA_TANH_WRAP=1, or edit the default here.
#ifndef NARMA_TANH_WRAP
#define NARMA_TANH_WRAP 1
#endif

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;

    constexpr size_t DIM         = 8;
    constexpr size_t N           = 1ULL << DIM;
    constexpr size_t narma_order = 30;          // fixed order for the history-depth sweep
    constexpr size_t collect     = 8000;        // states fed to the readout (80/20 split)
    constexpr uint64_t data_seed = 1939;        // signal-side RNG seed

    // history_depth (M) sweep points: below, around, and beyond the NARMA order,
    // to map where the delay line can finally hold the full lag history (the knee
    // sits near M = order). history_depth is capped at 64 by Reservoir::Create.
    const std::vector<size_t> sweep_M = {1, 2, 4, 8, 16, 24, 32, 48};

    // Second sweep dimension: reservoir-init seed. The target series depends on
    // narma_order + data_seed only (NOT the reservoir seed), so every (seed, M)
    // cell scores the byte-identical task -- the spread across seeds at a fixed M
    // is the run-to-run variance, which tells us whether the M-curve shape is real.
    const std::vector<uint64_t> sweep_seeds = {73895, 73896, 73897, 73898, 73899};

    std::cout << "=== HypercubeESN: NARMA-" << narma_order
              << " history_depth (M) x seed sweep ===\n\n";
    std::cout << "Task: reproduce the NARMA-" << narma_order
              << " output y(t) from its white input u(t),\n";
    std::cout << "sweeping the reservoir delay-line depth M while holding the\n";
    std::cout << "target series fixed -- an isolated test of memory depth.\n\n";

    // ---- Build the NARMA task ONCE -------------------------------------------
    // The task does not depend on M, so every trial scores the byte-identical
    // target series; the only thing that varies across runs is history_depth.
    NARMATaskConfig tc{narma_order, data_seed};
    tc.tanh_wrap = (NARMA_TANH_WRAP != 0);
    NARMATask task = MakeNARMATask(DIM, tc, collect);

    std::cout << "  Variant: " << (task.tanh_wrap
                  ? "tanh-wrapped (fixed coeffs -- honest order-scaling)"
                  : "legacy (scheduled coeffs -- nonlinearity weakens with order)")
              << "\n";
    std::cout << "  Series:  warmup=" << task.warmup
              << "  collect=" << task.collect
              << "  (train=" << task.tr << ", test=" << task.te << ")\n";
    std::cout << "  Coeffs:  alpha=" << task.coeffs.alpha
              << " beta=" << task.coeffs.beta
              << " gamma=" << task.coeffs.gamma
              << " delta=" << task.coeffs.delta
              << "  u in [" << task.coeffs.u_low << ", " << task.coeffs.u_high << "]\n";

    // ---- Base ESN config -----------------------------------------------------
    // Shared by every trial; only reservoir.seed and reservoir.history_depth
    // change per cell, so any NRMSE difference is attributable to those alone.
    ESNConfig base;
    base.reservoir.dim = DIM;
    base.reservoir.verbose = false;   // 40 trials -- suppress the per-trial SR banner
    base.reservoir.spectral_radius = 0.92;
    base.reservoir.leak_rate = 1.0;
    base.reservoir.input_scaling = 0.5;

    base.readout.task       = ReadoutTask::Regression;
    base.readout.epochs     = 600;
    base.readout.batch_size = 128;     // CPU cores saturate at batch >= 128
    base.readout.activation = ReadoutActivation::TANH;

    std::cout << "\n  Config: DIM=" << DIM << " N=" << N
              << "  sr=" << base.reservoir.spectral_radius
              << " leak=" << base.reservoir.leak_rate
              << " input_scaling=" << base.reservoir.input_scaling << "\n";
    std::cout << "  Training: " << base.readout.epochs << " epochs, batch="
              << base.readout.batch_size << ", lr=" << base.readout.lr_max
              << " (cosine, floor=" << (base.readout.lr_max * base.readout.lr_min_frac)
              << ")\n";
    std::cout << "  Sweep M:  ";
    for (size_t m : sweep_M) std::cout << m << ' ';
    std::cout << "\n  Seeds:    ";
    for (uint64_t sd : sweep_seeds) std::cout << sd << ' ';
    std::cout << "\n";

    // ---- Run the sweep (seed x M) --------------------------------------------
    const size_t nM = sweep_M.size();
    const size_t nS = sweep_seeds.size();
    std::vector<std::vector<double>> nrmse(nM, std::vector<double>(nS, 0.0));
    double target_mean = 0.0;  // identical across every cell (same target series)

    for (size_t si = 0; si < nS; ++si) {
        std::cout << "\n  === seed " << sweep_seeds[si]
                  << "  (" << (si + 1) << "/" << nS << ") ===\n" << std::flush;
        for (size_t mi = 0; mi < nM; ++mi) {
            ESNConfig cfg = base;
            cfg.reservoir.seed = sweep_seeds[si];
            cfg.reservoir.history_depth = sweep_M[mi];

            NARMATrialResult res = RunNARMATrial(cfg, task);
            target_mean = res.target_mean;
            nrmse[mi][si] = res.nrmse;
            std::cout << std::fixed
                      << "    M=" << std::setw(2) << sweep_M[mi]
                      << ": NRMSE=" << std::setprecision(4) << res.nrmse
                      << "  R2=" << std::setprecision(4) << res.r2
                      << "  (" << std::setprecision(1) << res.train_secs << "s)\n";
        }
    }

    // ---- Aggregate per M across seeds (mean / sample-std / min / max) ---------
    auto stats = [](const std::vector<double>& v) {
        double mn = v[0], mx = v[0], sum = 0.0;
        for (double x : v) { sum += x; mn = std::min(mn, x); mx = std::max(mx, x); }
        const double mean = sum / static_cast<double>(v.size());
        double var = 0.0;
        for (double x : v) var += (x - mean) * (x - mean);
        var = (v.size() > 1) ? var / static_cast<double>(v.size() - 1) : 0.0;
        return std::array<double, 4>{mean, std::sqrt(var), mn, mx};
    };

    std::cout << std::fixed;
    std::cout << "\n=== NARMA-" << narma_order << " history_depth x seed sweep"
              << "  (DIM=" << DIM << ", N=" << N << ", " << nS << " seeds"
              << ", train mean " << std::setprecision(4) << target_mean << ") ===\n";
    std::cout << "    " << std::setw(4) << "M"
              << "  " << std::setw(7) << "mean"
              << "  " << std::setw(7) << "std"
              << "  " << std::setw(7) << "min"
              << "  " << std::setw(7) << "max" << "\n";
    std::cout << "    " << std::setw(4) << "----"
              << "  " << std::setw(7) << "-------"
              << "  " << std::setw(7) << "-------"
              << "  " << std::setw(7) << "-------"
              << "  " << std::setw(7) << "-------" << "\n";
    for (size_t mi = 0; mi < nM; ++mi) {
        const std::array<double, 4> s = stats(nrmse[mi]);
        std::cout << "    " << std::setw(4) << sweep_M[mi]
                  << "  " << std::setw(7) << std::setprecision(4) << s[0]
                  << "  " << std::setw(7) << std::setprecision(4) << s[1]
                  << "  " << std::setw(7) << std::setprecision(4) << s[2]
                  << "  " << std::setw(7) << std::setprecision(4) << s[3]
                  << "\n";
    }

    // ---- Raw NRMSE matrix (rows = M, cols = seed) ----------------------------
    std::cout << "\n  Raw NRMSE (rows M, cols seed):\n";
    std::cout << "    " << std::setw(4) << "M";
    for (uint64_t sd : sweep_seeds) std::cout << "  " << std::setw(9) << sd;
    std::cout << "\n";
    for (size_t mi = 0; mi < nM; ++mi) {
        std::cout << "    " << std::setw(4) << sweep_M[mi];
        for (size_t si = 0; si < nS; ++si)
            std::cout << "  " << std::setw(9) << std::setprecision(4) << nrmse[mi][si];
        std::cout << "\n";
    }

    std::cout << "\nReconstruction quality should track memory depth, saturating once\n";
    std::cout << "M >= the NARMA order (the delay line can then hold the full lag history).\n";

    return 0;
}
