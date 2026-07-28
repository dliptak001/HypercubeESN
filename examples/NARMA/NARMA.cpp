/// @file NARMA.cpp
/// @brief NARMA-N open-loop system-identification validator on HypercubeESN.
/// See NARMA.md for the protocol, shared op-point, and campaign results.
///
/// Usage: NARMA.exe [order]
///   order — optional NARMA recurrence order N (>= 2); default kDefaultNarmaOrder.
///
/// Default campaign path: tanh-wrapped NARMA, fixed M and knobs, multi-seed
/// survey (best-5 of 20 featured). Only the CLI order and reservoir.seed vary.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
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

// Default order when main is run with no arguments. Override at the command line:
//   NARMA.exe [order]
//   NARMA.exe 30
namespace {
constexpr size_t kDefaultNarmaOrder = 50;

void PrintUsage(const char* argv0)
{
    std::cerr << "Usage: " << (argv0 ? argv0 : "NARMA") << " [order]\n"
              << "  order  NARMA recurrence order N (integer >= 2).\n"
              << "         Optional; default is " << kDefaultNarmaOrder << ".\n"
              << "  -h, --help  Show this message.\n";
}

// Parse optional order from argv.
// Returns 0 on success, 0 after --help (caller should exit 0), 1 on bad args.
// On help/error, usage is already printed; out_order is only valid on success.
int ParseNarmaOrder(int argc, char* argv[], size_t& out_order, bool& is_help)
{
    is_help = false;
    out_order = kDefaultNarmaOrder;
    if (argc <= 1)
        return 0;

    const std::string a1 = argv[1] ? argv[1] : "";
    if (a1 == "-h" || a1 == "--help" || a1 == "/?") {
        PrintUsage(argv[0]);
        is_help = true;
        return 0;
    }
    if (argc > 2) {
        std::cerr << "NARMA: unexpected extra arguments.\n";
        PrintUsage(argv[0]);
        return 1;
    }

    char* end = nullptr;
    const unsigned long v = std::strtoul(a1.c_str(), &end, 10);
    if (end == a1.c_str() || (end && *end != '\0') || v < 2ul) {
        std::cerr << "NARMA: order must be an integer >= 2 (got \"" << a1 << "\").\n";
        PrintUsage(argv[0]);
        return 1;
    }
    out_order = static_cast<size_t>(v);
    return 0;
}
} // namespace

int main(int argc, char* argv[])
{
    size_t narma_order = kDefaultNarmaOrder;
    bool help = false;
    if (const int rc = ParseNarmaOrder(argc, argv, narma_order, help); rc != 0)
        return rc;
    if (help)
        return 0;

    // Shared campaign op-point — keep in lockstep with examples/NARMA/NARMA.md
    // ("Shared configuration"). Only narma_order (CLI) and reservoir.seed vary.
    constexpr size_t DIM         = 10;          // N = 1024
    constexpr size_t N           = 1ULL << DIM;
    constexpr size_t collect     = 32000;       // train 25600 / test 6400
    constexpr size_t warmup      = 300;
    constexpr uint64_t data_seed = 1939;        // signal-side RNG (series fixed across res seeds)
    constexpr size_t history_M   = 32;          // history_depth for the campaign

    // Single-M "sweep" (campaign is multi-seed at fixed M, not an M ladder).
    const std::vector<size_t> sweep_M = {history_M};

    // 20 reservoir seeds (same list for N30/N50/N70). Formula (73896+k)*(k+2), k=0..19.
    // Values: 147792 221691 295592 369495 443400 517307 591216 665127 739040 812955
    //         886872 960791 1034712 1108635 1182560 1256487 1330416 1404347 1478280 1552215
    const std::vector<uint64_t> sweep_reservoir_seeds = {
        73896*2,  73897*3,  73898*4,  73899*5,  73900*6,
        73901*7,  73902*8,  73903*9,  73904*10, 73905*11,
        73906*12, 73907*13, 73908*14, 73909*15, 73910*16,
        73911*17, 73912*18, 73913*19, 73914*20, 73915*21,
    };
    // Spotlight size: best-k of n (featured in NARMA.md); full pool always logged.
    constexpr size_t k_best = 5;

    std::cout << "=== HypercubeESN: NARMA-" << narma_order
              << " multi-seed survey (fixed M) ===\n\n";
    std::cout << "Task: open-loop system identification — reproduce y(t) from u(t).\n";
    std::cout << "Fixed op-point; " << sweep_reservoir_seeds.size()
              << " reservoir seeds; series held fixed (data_seed).\n"
              << "Featured metric: best-" << k_best
              << " of those seeds (lowest test NRMSE).\n\n";

    // ---- Build the NARMA task ONCE -------------------------------------------
    // Independent of reservoir seed: every trial scores the byte-identical series.
    NARMATaskConfig tc{narma_order, data_seed};
    tc.tanh_wrap = (NARMA_TANH_WRAP != 0); // campaign: 1 (fixed α β γ δ + outer tanh)
    NARMATask task = MakeNARMATask(DIM, tc, collect, warmup);

    std::cout << "  Variant: " << (task.tanh_wrap
                  ? "tanh-wrapped (fixed coeffs -- honest order-scaling)"
                  : "legacy (scheduled coeffs -- nonlinearity weakens with order)")
              << "\n";
    std::cout << "  Series:  warmup=" << task.warmup
              << "  collect=" << task.collect
              << "  (train=" << task.tr << ", test=" << task.te << ")"
              << "  data_seed=" << data_seed << "\n";
    std::cout << "  Coeffs:  alpha=" << task.coeffs.alpha
              << " beta=" << task.coeffs.beta
              << " gamma=" << task.coeffs.gamma
              << " delta=" << task.coeffs.delta
              << "  u in [" << task.coeffs.u_low << ", " << task.coeffs.u_high << "]\n";

    // ---- Base ESN config (NARMA.md shared configuration) --------------------
    // Per trial: only reservoir.seed changes (history_depth stays at history_M).
    ESNConfig base;
    base.reservoir.dim = DIM;
    base.reservoir.verbose = false;            // suppress per-trial SR banner
    base.reservoir.spectral_radius = 0.99f;
    base.reservoir.input_scaling = 0.03f;
    base.reservoir.leak_rate = 1.0f;
    base.reservoir.history_depth = history_M;
    base.reservoir.bias_scaling = 0.02f;       // pin campaign default

    base.readout_slices = 2;                   // B=2 → HCNN start dim 11
    base.readout.seed = 3423555;               // fixed HCNN init (isolate res seed)
    base.readout.conv_channels = 16;
    base.readout.num_layers = 1;
    base.readout.use_pooling = true;
    base.readout.pool_type = ReadoutPoolType::Max;
    base.readout.task = ReadoutTask::Regression;
    base.readout.activation = ReadoutActivation::TANH;
    base.readout.epochs = 600;
    base.readout.batch_size = 128;
    base.readout.lr_max = 0.0015f;
    base.readout.lr_min_frac = 0.005f;         // floor = 0.0015 * 0.005 = 7.5e-06
    // Best-epoch restore scores the full training set (holdout_frac=0) — train MSE,
    // not a validation split. Test NRMSE remains a clean held-out metric.
    base.readout.restore_best_epoch = true;
    base.readout.best_epoch_holdout_frac = 0.0f;

    std::cout << "\n  Config: DIM=" << DIM << " N=" << N
              << "  M=" << history_M
              << "  sr=" << base.reservoir.spectral_radius
              << " leak=" << base.reservoir.leak_rate
              << " input_scaling=" << base.reservoir.input_scaling
              << " bias_scaling=" << base.reservoir.bias_scaling << "\n";
    std::cout << "  Readout: slices=" << base.readout_slices
              << "  readout.seed=" << base.readout.seed
              << "  restore_best_epoch="
              << (base.readout.restore_best_epoch ? "true" : "false")
              << " (train-MSE; holdout_frac="
              << base.readout.best_epoch_holdout_frac << ")\n";
    std::cout << "  Training: " << base.readout.epochs << " epochs, batch="
              << base.readout.batch_size << ", lr=" << base.readout.lr_max
              << " (cosine, floor=" << (base.readout.lr_max * base.readout.lr_min_frac)
              << ")\n";
    std::cout << "  Res seeds (" << sweep_reservoir_seeds.size() << "): ";
    for (uint64_t sd : sweep_reservoir_seeds) std::cout << sd << ' ';
    std::cout << "\n";

    // Architecture is shared across seeds (only reservoir.seed changes).
    // Probe once so logs show stack + param count without per-trial noise.
    {
        ESNConfig probe = base;
        probe.reservoir.seed =
            sweep_reservoir_seeds.empty() ? 42ULL : sweep_reservoir_seeds.front();
        probe.reservoir.history_depth =
            sweep_M.empty() ? history_M : sweep_M.front();
        ESN probe_esn(probe);
        std::cout << probe_esn.ReadoutArchSummary();
    }

    // ---- Multi-seed survey (fixed M; optional multi-M via sweep_M) ----------
    const size_t nM = sweep_M.size();
    const size_t nS = sweep_reservoir_seeds.size();
    const size_t nTrials = nS;
    std::vector<std::vector<double>> nrmse(nM, std::vector<double>(nTrials, 0.0));
    double target_mean = 0.0;  // identical across every cell (same target series)

    for (size_t si = 0; si < nS; ++si) {
        const size_t ti = si;
        std::cout << "\n  === res_seed " << sweep_reservoir_seeds[si]
                  << "  (" << (ti + 1) << "/" << nTrials << ") ===\n" << std::flush;
        for (size_t mi = 0; mi < nM; ++mi) {
            ESNConfig cfg = base;
            cfg.reservoir.seed = sweep_reservoir_seeds[si];
            cfg.reservoir.history_depth = sweep_M[mi];

            NARMATrialResult res = RunNARMATrial(cfg, task);
            target_mean = res.target_mean;
            nrmse[mi][ti] = res.nrmse;
            std::cout << std::fixed;
            if (nM > 1)
                std::cout << "    M=" << std::setw(2) << sweep_M[mi] << ": ";
            else
                std::cout << "    ";
            std::cout << "NRMSE=" << std::setprecision(4) << res.nrmse
                      << "  R2=" << std::setprecision(4) << res.r2
                      << "  (" << std::setprecision(1) << res.train_secs << "s)\n";
        }
    }

    // ---- Aggregate (full pool + best-k spotlight) ---------------------------
    // Full raw table always reports every trial. Summary tables:
    //   all-nTrials  — multi-seed mean / sample-std / min / max
    //   best-k_best  — lowest-NRMSE trials (featured in NARMA.md)
    auto stats = [](const std::vector<double>& v) {
        double mn = v[0], mx = v[0], sum = 0.0;
        for (double x : v) { sum += x; mn = std::min(mn, x); mx = std::max(mx, x); }
        const double mean = sum / static_cast<double>(v.size());
        double var = 0.0;
        for (double x : v) var += (x - mean) * (x - mean);
        var = (v.size() > 1) ? var / static_cast<double>(v.size() - 1) : 0.0;
        return std::array<double, 4>{mean, std::sqrt(var), mn, mx};
    };

    auto print_stats_table = [&](const char* title, bool best_only) {
        std::cout << "\n  " << title << "\n";
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
            std::vector<double> vals;
            if (!best_only) {
                vals = nrmse[mi];
            } else {
                const size_t k = std::min(k_best, nTrials);
                std::vector<size_t> order(nTrials);
                for (size_t ti = 0; ti < nTrials; ++ti) order[ti] = ti;
                std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k),
                                  order.end(),
                                  [&](size_t a, size_t b) {
                                      return nrmse[mi][a] < nrmse[mi][b];
                                  });
                vals.reserve(k);
                for (size_t j = 0; j < k; ++j)
                    vals.push_back(nrmse[mi][order[j]]);
            }
            const std::array<double, 4> s = stats(vals);
            std::cout << "    " << std::setw(4) << sweep_M[mi]
                      << "  " << std::setw(7) << std::setprecision(4) << s[0]
                      << "  " << std::setw(7) << std::setprecision(4) << s[1]
                      << "  " << std::setw(7) << std::setprecision(4) << s[2]
                      << "  " << std::setw(7) << std::setprecision(4) << s[3]
                      << "\n";
        }
    };

    std::cout << std::fixed;
    std::cout << "\n=== NARMA-" << narma_order << " multi-seed survey"
              << "  (DIM=" << DIM << ", N=" << N
              << ", M=" << history_M
              << ", " << nS << " res seeds"
              << ", " << nTrials << " trials total"
              << ", train mean " << std::setprecision(4) << target_mean << ") ===\n";
    std::cout << "  Stats: all-" << nTrials << " pool + best-"
              << std::min(k_best, nTrials) << " of " << nTrials
              << " (lowest test NRMSE; featured multi-seed band).\n";

    {
        std::string all_title = "All " + std::to_string(nTrials)
            + " trials (mean/std = multi-seed typical):";
        print_stats_table(all_title.c_str(), /*best_only=*/false);
    }
    {
        const size_t k = std::min(k_best, nTrials);
        std::string best_title = "Best " + std::to_string(k) + " of " + std::to_string(nTrials)
            + " (lowest NRMSE; max column = worst of top-" + std::to_string(k) + "):";
        print_stats_table(best_title.c_str(), /*best_only=*/true);
    }

    // Best-k trial list (which seeds made the spotlight).
    {
        const size_t k = std::min(k_best, nTrials);
        std::cout << "\n  Best-" << k << " seeds (lowest test NRMSE first):\n";
        for (size_t mi = 0; mi < nM; ++mi) {
            std::vector<size_t> order(nTrials);
            for (size_t ti = 0; ti < nTrials; ++ti) order[ti] = ti;
            std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k),
                              order.end(),
                              [&](size_t a, size_t b) {
                                  return nrmse[mi][a] < nrmse[mi][b];
                              });
            if (nM > 1)
                std::cout << "    M=" << sweep_M[mi] << ":";
            else
                std::cout << "   ";
            for (size_t j = 0; j < k; ++j) {
                const size_t ti = order[j];
                std::cout << "  [" << (j + 1) << "] "
                          << sweep_reservoir_seeds[ti]
                          << "=" << std::setprecision(4) << nrmse[mi][ti];
            }
            std::cout << "\n";
        }
    }

    // ---- Raw NRMSE (cols = res_seed; optional M row label if multi-M) --------
    std::cout << "\n  Raw NRMSE — all " << nTrials << " trials:\n";
    std::cout << "    ";
    if (nM > 1)
        std::cout << std::setw(4) << "M";
    for (size_t si = 0; si < nS; ++si)
        std::cout << "  " << std::setw(9) << sweep_reservoir_seeds[si];
    std::cout << "\n";
    for (size_t mi = 0; mi < nM; ++mi) {
        std::cout << "    ";
        if (nM > 1)
            std::cout << std::setw(4) << sweep_M[mi];
        for (size_t ti = 0; ti < nTrials; ++ti)
            std::cout << "  " << std::setw(9) << std::setprecision(4) << nrmse[mi][ti];
        std::cout << "\n";
    }

    std::cout << "\nOpen-loop test NRMSE on the held-out "
              << task.te << " steps (train " << task.tr
              << "). Same series for every seed; only reservoir.seed varies.\n";

    return 0;
}
