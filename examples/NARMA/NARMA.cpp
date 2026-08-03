/// @file NARMA.cpp
/// @brief NARMA-N open-loop system-identification validator on HypercubeESN.
/// See NARMA.md for the protocol, shared op-point, and campaign results.
///
/// Usage: NARMA.exe [order]
///   order - optional NARMA recurrence order N (>= 2); default kDefaultNarmaOrder.
///
/// Campaign path: tanh-wrapped NARMA, fixed op-point. Edit kReservoirSeeds:
///   1 seed  - spot run
///   3 seeds - literature band (mean/std/min/max over the three trials)
/// Only the CLI order and each trial's reservoir.seed vary.

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

// =============================================================================
// Campaign configuration - primary knobs (edit here; keep lockstep with NARMA.md)
// =============================================================================
//
// Layout:
//   1. Task / series     - what NARMA order series is built from
//   2. Base ESN op-point - MakeBaseESNConfig(); fixed across seeds/orders
//   3. Reservoir seeds   - explicit list: 1 = spot, 3 = literature stats
//
// Per trial only reservoir.seed changes. CLI order is the only runtime task knob;
// series data_seed stays fixed so every reservoir seed scores the same u/y series.

namespace campaign {

// --- 1. Task / series --------------------------------------------------------

/// Default NARMA recurrence order when main is run with no arguments.
constexpr size_t kDefaultNarmaOrder = 50;

/// Collect steps (train 80% / test 20% via MakeNARMATask).
constexpr size_t kCollect = 32000;

/// Reservoir washout before collect.
constexpr size_t kWarmup = 300;

/// Signal-side RNG - fixes the entire u/y series across all reservoir seeds.
constexpr uint64_t kDataSeed = 1939;

// --- 2. Base ESN op-point (shared across 30/50/70 and all seeds) --------------

/// Fixed HypercubeESN configuration for the campaign.
/// Per-trial override: only reservoir.seed (set in main).
inline ESNConfig MakeBaseESNConfig()
{
    ESNConfig cfg;

    // Reservoir (fixed dynamics; seed set per trial)
    cfg.reservoir.dim             = 10;  // N = 1024
    cfg.reservoir.verbose         = false;  // suppress per-trial SR banner
    cfg.reservoir.spectral_radius = 0.99f;
    cfg.reservoir.input_scaling   = 0.03f;
    cfg.reservoir.leak_rate       = 1.0f;
    cfg.reservoir.history_depth   = 29;//32;
    cfg.reservoir.bias_scaling    = 0.02f;  // pin campaign default

    // Seam: delay-line ages packed into the readout (B=2 -> HCNN start dim 11)
    cfg.readout_slices = 2;

    // Readout (trainable HCNN) - fixed seed so multi-seed spread is reservoir-side
    cfg.readout.seed                    = 73423555; //3423555;
    cfg.readout.task                    = ReadoutTask::Regression;
    cfg.readout.activation              = ReadoutActivation::TANH;
    cfg.readout.conv_channels           = 16;
    cfg.readout.num_layers              = 1;
    cfg.readout.use_pooling             = true;
    cfg.readout.pool_type               = ReadoutPoolType::Max;
    cfg.readout.epochs                  = 600;
    cfg.readout.batch_size              = 128;
    cfg.readout.lr_max                  = 0.0015f;
    cfg.readout.lr_min_frac             = 0.005f;  // floor = 7.5e-06
    // Best-epoch restore scores full training set (holdout_frac=0) - train MSE,
    // not a validation split. Test NRMSE remains a clean held-out metric.
    cfg.readout.restore_best_epoch      = true;
    cfg.readout.best_epoch_holdout_frac = 0.0f;

    return cfg;
}

// --- 3. Reservoir seeds ------------------------------------------------------
//
// Provide the full list here. Length is the survey mode:
//   1 entry  - spot run (single trial; stats are trivial)
//   3 entries - literature band (mean / sample-std / min / max over the three)
//
// No other sizes. No best-k selection - every listed seed is reported in full.

inline constexpr uint64_t kReservoirSeeds[] = {
    7934791766227647176ull,  // spot: leave only this line; literature: keep all three
    8982357012682103037ull,
    3079493423467196890ull,
};
    // inline constexpr uint64_t kReservoirSeeds[] = {
    //     4112530987988204306ull,  // spot: leave only this line; literature: keep all three
    //     8982357012682103037ull,
    //     15208094364242385359ull,
    // };

inline constexpr size_t kNumReservoirSeeds =
    sizeof(kReservoirSeeds) / sizeof(kReservoirSeeds[0]);

static_assert(kNumReservoirSeeds == 1 || kNumReservoirSeeds == 3,
              "campaign::kReservoirSeeds must have 1 (spot) or 3 (literature) entries");

} // namespace campaign

// =============================================================================
// CLI helpers
// =============================================================================

namespace {

void PrintUsage(const char* argv0)
{
    std::cerr << "Usage: " << (argv0 ? argv0 : "NARMA") << " [order]\n"
              << "  order  NARMA recurrence order N (integer >= 2).\n"
              << "         Optional; default is " << campaign::kDefaultNarmaOrder << ".\n"
              << "  -h, --help  Show this message.\n";
}

// Parse optional order from argv.
// Returns 0 on success (including --help), 1 on bad args.
// On help/error, usage is already printed; out_order is only valid when not help.
int ParseNarmaOrder(int argc, char* argv[], size_t& out_order, bool& is_help)
{
    is_help = false;
    out_order = campaign::kDefaultNarmaOrder;
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

/// mean / sample-std / min / max over a non-empty vector.
std::array<double, 4> Stats(const std::vector<double>& v)
{
    double mn = v[0], mx = v[0], sum = 0.0;
    for (double x : v) {
        sum += x;
        if (x < mn) mn = x;
        if (x > mx) mx = x;
    }
    const double mean = sum / static_cast<double>(v.size());
    double var = 0.0;
    for (double x : v)
        var += (x - mean) * (x - mean);
    var = (v.size() > 1) ? var / static_cast<double>(v.size() - 1) : 0.0;
    return {mean, std::sqrt(var), mn, mx};
}

void PrintStatsLine(const std::vector<double>& vals)
{
    const std::array<double, 4> s = Stats(vals);
    std::cout << "    mean=" << std::setw(7) << std::setprecision(4) << s[0]
              << "  std=" << std::setw(7) << std::setprecision(4) << s[1]
              << "  min=" << std::setw(7) << std::setprecision(4) << s[2]
              << "  max=" << std::setw(7) << std::setprecision(4) << s[3]
              << "\n";
}

} // namespace

// =============================================================================

int main(int argc, char* argv[])
{
    using namespace campaign;

    size_t narma_order = kDefaultNarmaOrder;
    bool help = false;
    if (const int rc = ParseNarmaOrder(argc, argv, narma_order, help); rc != 0)
        return rc;
    if (help)
        return 0;

    const ESNConfig base = MakeBaseESNConfig();
    const size_t nTrials = kNumReservoirSeeds;
    const size_t N = size_t{1} << base.reservoir.dim;
    const float lr_floor = base.readout.lr_max * base.readout.lr_min_frac;
    const char* mode = (nTrials == 1) ? "spot (1 seed)" : "literature (3 seeds)";

    std::cout << "=== HypercubeESN: NARMA-" << narma_order
              << "  [" << mode << "] ===\n\n";
    std::cout << "Task: open-loop system identification - reproduce y(t) from u(t).\n";
    std::cout << "Fixed op-point; " << nTrials
              << " reservoir seed(s); series held fixed (data_seed).\n";
    if (nTrials == 3)
        std::cout << "Stats: mean / sample-std / min / max over all 3 seeds "
                     "(no best-k selection).\n\n";
    else
        std::cout << "Spot run: single trial (edit kReservoirSeeds to 3 for a band).\n\n";

    // ---- Build the NARMA task ONCE -------------------------------------------
    // Independent of reservoir seed: every trial scores the byte-identical series.
    NARMATaskConfig tc{narma_order, kDataSeed};
    tc.tanh_wrap = (NARMA_TANH_WRAP != 0); // campaign: 1 (fixed a b g d + outer tanh)
    NARMATask task = MakeNARMATask(base.reservoir.dim, tc, kCollect, kWarmup);

    // Demo schedule + op-point banner (architecture once below).
    std::cout << "Variant: " << (task.tanh_wrap
                  ? "tanh-wrapped (fixed coeffs -- honest order-scaling)"
                  : "legacy (scheduled coeffs -- nonlinearity weakens with order)")
              << "\n";
    std::cout << "Stream:  warmup=" << task.warmup
              << "  collect=" << task.collect
              << "  train=" << task.tr
              << "  test=" << task.te
              << "  data_seed=" << kDataSeed << "\n";
    std::cout << "Coeffs:  alpha=" << task.coeffs.alpha
              << "  beta=" << task.coeffs.beta
              << "  gamma=" << task.coeffs.gamma
              << "  delta=" << task.coeffs.delta
              << "  u in [" << task.coeffs.u_low << ", " << task.coeffs.u_high << "]\n";
    std::cout << "Train:   " << base.readout.epochs << " epochs"
              << "  batch=" << base.readout.batch_size
              << "  lr_max=" << base.readout.lr_max
              << "  (cosine, floor=" << lr_floor << ")\n";
    std::cout << "Op-point: M=" << base.reservoir.history_depth
              << "  sr=" << base.reservoir.spectral_radius
              << "  leak=" << base.reservoir.leak_rate
              << "  input_scaling=" << base.reservoir.input_scaling
              << "  bias_scaling=" << base.reservoir.bias_scaling
              << "  readout_slices=" << base.readout_slices
              << "  readout.seed=" << base.readout.seed << "\n";
    std::cout << "Survey:  " << nTrials << " res seed(s)"
              << "  restore_best_epoch="
              << (base.readout.restore_best_epoch ? "true" : "false")
              << " (holdout_frac=" << base.readout.best_epoch_holdout_frac << ")\n";
    std::cout << "Seeds:  ";
    for (size_t i = 0; i < nTrials; ++i)
        std::cout << (i ? " " : "") << kReservoirSeeds[i];
    std::cout << "\n\n";

    // Architecture is shared across seeds (only reservoir.seed changes).
    // Probe once so logs show stack + param count without per-trial noise.
    {
        ESNConfig probe = base;
        probe.reservoir.seed = kReservoirSeeds[0];
        ESN probe_esn(probe);
        std::cout << probe_esn.ReadoutArchSummary() << "\n";
    }

    // ---- Trials (1 or 3 seeds) ----------------------------------------------
    std::vector<double> nrmse(nTrials, 0.0);
    double target_mean = 0.0;  // identical across every trial (same target series)

    for (size_t ti = 0; ti < nTrials; ++ti) {
        std::cout << "\n  === res_seed " << kReservoirSeeds[ti]
                  << "  (" << (ti + 1) << "/" << nTrials << ") ===\n" << std::flush;

        ESNConfig cfg = base;
        cfg.reservoir.seed = kReservoirSeeds[ti];

        NARMATrialResult res = RunNARMATrial(cfg, task);
        target_mean = res.target_mean;
        nrmse[ti] = res.nrmse;
        std::cout << std::fixed
                  << "    NRMSE=" << std::setprecision(4) << res.nrmse
                  << "  R2=" << std::setprecision(4) << res.r2
                  << "  (" << std::setprecision(1) << res.train_secs << "s)\n";
    }

    // ---- Aggregate (every listed seed; no best-k) ---------------------------
    std::cout << std::fixed;
    std::cout << "\n=== NARMA-" << narma_order << " summary"
              << "  (DIM=" << base.reservoir.dim << ", N=" << N
              << ", M=" << base.reservoir.history_depth
              << ", " << nTrials << " seed(s)"
              << ", train mean " << std::setprecision(4) << target_mean << ") ===\n";

    if (nTrials == 1) {
        std::cout << "  Spot NRMSE: " << std::setprecision(4) << nrmse[0] << "\n";
    } else {
        std::cout << "  All " << nTrials
                  << " seeds (mean/std/min/max - literature band):\n";
        PrintStatsLine(nrmse);
    }

    std::cout << "\n  Per-seed NRMSE:\n";
    for (size_t ti = 0; ti < nTrials; ++ti) {
        std::cout << "    seed " << kReservoirSeeds[ti]
                  << "  NRMSE=" << std::setprecision(4) << nrmse[ti] << "\n";
    }

    std::cout << "\nOpen-loop test NRMSE on the held-out "
              << task.te << " steps (train " << task.tr
              << "). Same series for every seed; only reservoir.seed varies.\n";

    return 0;
}
