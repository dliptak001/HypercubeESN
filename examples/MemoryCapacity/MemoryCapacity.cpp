/// @file MemoryCapacity.cpp
/// @brief Driver for the memory-capacity (MC) diagnostic. See MemoryCapacity.md.
///
/// This is a thin orchestration + reporting layer over MemoryCapacityMeter (the
/// engine, in MemoryCapacity.h). The engine measures MC at one operating point
/// given a ReservoirConfig; this file builds the operating points for each run
/// mode, fans them out via RunSweep, and formats the tables. Edit main() to pick
/// a mode and its parameters.
///
/// Run modes:
///   RunDetailed   — one operating point, full per-lag r²(k) table.
///   RunGridSweep  — sr × leak × history_depth grid; ordered table + per-M grids.
///   RunSeedSurvey — fixed op-point, sweep reservoir seed; realization variance.
///   RunDepthProbe — per-lag r²(k) curves for several depths, side-by-side.

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "MemoryCapacity.h"

namespace
{
    using mc::MCConfig;
    using mc::MCResult;
    using mc::MeasureOptions;
    using mc::MemoryCapacityMeter;
    using mc::RunSweep;
    using mc::SweepOptions;

    /// Shared "Fixed: ..." experiment line, printed by every mode so its output
    /// self-describes the MCConfig it ran under.
    void PrintExperimentLine(const MCConfig& mc, float input_scaling)
    {
        std::cout << "Fixed: in=1 is=" << input_scaling
            << " warmup=" << mc.t_warmup << " collect=" << mc.t_collect
            << " Kmax=" << mc.k_max << " ridge=" << mc.ridge_lambda << "\n";
    }

    /// Live progress callback: a single rewritten "completed d/total" line.
    auto MakeProgress(std::size_t every = 1)
    {
        return [every](std::size_t done, std::size_t total)
        {
            if (done % every == 0 || done == total)
                std::cout << "\r  completed " << done << "/" << total << " cells" << std::flush;
        };
    }

    // ---------------------------------------------------------------------------
    // Mode 1: one operating point, full per-lag r²(k) table.
    // ---------------------------------------------------------------------------
    [[maybe_unused]] void RunDetailed(const MemoryCapacityMeter& meter, const ReservoirConfig& rcfg)
    {
        const MCConfig& mc = meter.Config();
        const std::size_t N = meter.Size();

        std::cout << "=== HypercubeESN: Memory Capacity ===\n\n";
        std::cout << "Reservoir : DIM=" << meter.Dim() << " N=" << N
            << "  sr=" << rcfg.spectral_radius
            << "  is=" << rcfg.input_scaling
            << "  leak=" << rcfg.leak_rate
            << "  hist=" << rcfg.history_depth
            << "  seed=" << rcfg.seed << "\n";
        std::cout << "Features  : " << meter.Features() << " (first " << meter.Features() << " vertices)\n";
        std::cout << "Drive     : T_warmup=" << mc.t_warmup
            << "  T_collect=" << mc.t_collect
            << "  K_max=" << mc.k_max
            << "  M(usable)=" << meter.Samples() << "\n";
        std::cout << "Split     : M_train=" << meter.TrainRows()
            << "  M_test=" << meter.TestRows()
            << "  train/F=" << std::fixed << std::setprecision(2)
            << (static_cast<double>(meter.TrainRows()) / static_cast<double>(meter.Features())) << "\n";
        std::cout << std::defaultfloat;
        std::cout << "Regression: ridge lambda=" << mc.ridge_lambda
            << "  input_seed=0x" << std::hex << mc.input_seed << std::dec << "\n\n";

        std::cout << "Measuring (full " << mc.k_max << "-lag curve) ... " << std::flush;
        const MCResult r = meter.Measure(rcfg, MeasureOptions{
                                             /*early_stop=*/false, /*kmax=*/0
                                         });
        std::cout << "done.\n\n" << std::flush;

        if (!r.pd)
        {
            std::cerr << "ERROR: train Gram not positive definite. Increase ridge lambda.\n";
            return;
        }

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  k    r2(test)\n";
        std::cout << "  ---  --------\n";
        for (std::size_t k = 1; k <= mc.k_max; ++k)
            std::cout << "  " << std::setw(3) << k
                << "  " << std::setw(8) << r.r2[k - 1] << "\n";

        std::cout << "\n=== Summary ===\n";
        std::cout << "Total MC = " << std::setprecision(3) << r.total_mc
            << "  (theoretical max F=" << meter.Features() << ")\n";
        std::cout << "Last lag with r^2 > 0.50 : k=" << r.k50 << "\n";
        std::cout << "Last lag with r^2 > 0.10 : k=" << r.k10 << "\n";
        std::cout << "Last lag with r^2 > 0.01 : k=" << r.k01 << "\n";
        std::cout << std::defaultfloat;
    }

    // ---------------------------------------------------------------------------
    // Mode 2: sr × leak × history_depth grid.
    // ---------------------------------------------------------------------------
    [[maybe_unused]] void RunGridSweep(const MemoryCapacityMeter& meter, ReservoirConfig base,
                      const std::vector<float>& spectral_radii,
                      const std::vector<float>& leak_rates,
                      const std::vector<std::size_t>& history_depths,
                      const SweepOptions& sweep_opts = {})
    {
        const MCConfig& mc = meter.Config();
        const std::size_t N = meter.Size();
        const std::size_t nsr = spectral_radii.size();
        const std::size_t nleak = leak_rates.size();
        const std::size_t nhist = history_depths.size();
        const std::size_t cells = nsr * nleak * nhist;
        if (cells == 0) return;

        // history_depth must be a positive power of 2 — the Reservoir ctor throws
        // otherwise. Fail fast here rather than from inside a worker thread.
        for (std::size_t hd : history_depths)
            if (hd == 0 || (hd & (hd - 1)) != 0)
            {
                std::cerr << "RunGridSweep: history_depth " << hd
                    << " is not a positive power of 2 — aborting.\n";
                return;
            }

        // Flat layout: ((i_sr * nleak) + j_leak) * nhist + k_hist.
        std::vector<ReservoirConfig> configs;
        configs.reserve(cells);
        for (std::size_t i = 0; i < nsr; ++i)
            for (std::size_t j = 0; j < nleak; ++j)
                for (std::size_t k = 0; k < nhist; ++k)
                {
                    ReservoirConfig c = base;
                    c.spectral_radius = spectral_radii[i];
                    c.leak_rate = leak_rates[j];
                    c.history_depth = history_depths[k];
                    configs.push_back(c);
                }

        const std::size_t workers = mc::ResolveWorkerCount(cells, meter.PerCellBytes(), sweep_opts);

        std::cout << "=== HypercubeESN: MC Profile (DIM=" << meter.Dim()
            << " N=" << N << " F=" << meter.Features() << " is=" << base.input_scaling << ") ===\n";
        PrintExperimentLine(mc, base.input_scaling);
        std::cout << nsr << " sr x " << nleak << " leak x " << nhist
            << " hist = " << cells << " cells | " << workers
            << " workers | ~" << std::fixed << std::setprecision(2)
            << (static_cast<double>(meter.PerCellBytes()) / 1e9) << " GB/cell, est peak ~"
            << (static_cast<double>(meter.PerCellBytes() * workers) / 1e9) << " GB\n\n"
            << std::defaultfloat << std::flush;

        const std::vector<MCResult> results = RunSweep(meter, configs, sweep_opts, MakeProgress());
        std::cout << "\n\n";

        auto cell = [&](std::size_t i, std::size_t j, std::size_t k) -> const MCResult&
        {
            return results[(i * nleak + j) * nhist + k];
        };

        // ---- Results table (ordered: sr, then leak, then hist) ----
        std::cout << "    sr   leak  hist  realSR    TotalMC  k>.5  k>.1  k>.01\n";
        std::cout << "  -----  -----  ----  ------  ---------  ----  ----  -----\n";
        for (std::size_t i = 0; i < nsr; ++i)
            for (std::size_t j = 0; j < nleak; ++j)
                for (std::size_t k = 0; k < nhist; ++k)
                {
                    const MCResult& m = cell(i, j, k);
                    std::cout << std::fixed << std::setprecision(2)
                        << "  " << std::setw(5) << spectral_radii[i]
                        << "  " << std::setw(5) << leak_rates[j]
                        << "  " << std::setw(4) << history_depths[k]
                        << "  " << std::setw(6) << std::setprecision(4) << m.realized_sr << "  ";
                    if (m.oom)
                    {
                        std::cout << "  OOM (lower max_workers/ram_budget)\n";
                        continue;
                    }
                    if (!m.pd)
                    {
                        std::cout << "  not-PD (raise ridge)\n";
                        continue;
                    }
                    std::cout << std::setw(9) << std::setprecision(3) << m.total_mc
                        << "  " << std::setw(4) << m.k50
                        << "  " << std::setw(4) << m.k10
                        << "  " << std::setw(5) << m.k01 << "\n";
                }

        // ---- One TotalMC grid (rows=sr, cols=leak) per history depth ----
        for (std::size_t k = 0; k < nhist; ++k)
        {
            std::cout << "\n=== TotalMC grid (M=" << history_depths[k]
                << ", is=" << base.input_scaling << ", rows=sr, cols=leak) ===\n";
            std::cout << std::fixed << std::setprecision(2) << std::setw(9) << "sr\\leak";
            for (float lk : leak_rates) std::cout << std::setw(8) << lk;
            std::cout << "\n";
            for (std::size_t i = 0; i < nsr; ++i)
            {
                std::cout << "  " << std::setw(5) << std::setprecision(2) << spectral_radii[i] << "  ";
                for (std::size_t j = 0; j < nleak; ++j)
                {
                    const MCResult& m = cell(i, j, k);
                    if (m.oom || !m.pd) std::cout << std::setw(8) << "n/a";
                    else std::cout << std::setw(8) << std::setprecision(2) << m.total_mc;
                }
                std::cout << "\n";
            }
        }
        std::cout << std::defaultfloat << std::flush;
    }

    // ---------------------------------------------------------------------------
    // Mode 3: fixed op-point, sweep the reservoir seed over [seed_start, seed_end].
    // ---------------------------------------------------------------------------
    [[maybe_unused]] void RunSeedSurvey(const MemoryCapacityMeter& meter, ReservoirConfig base,
                       std::uint64_t seed_start, std::uint64_t seed_end,
                       const SweepOptions& sweep_opts = {})
    {
        const MCConfig& mc = meter.Config();
        const std::size_t N = meter.Size();

        if (seed_end < seed_start)
        {
            std::cerr << "RunSeedSurvey: seed_end < seed_start — nothing to do.\n";
            return;
        }
        const std::size_t count = static_cast<std::size_t>(seed_end - seed_start + 1);

        std::vector<ReservoirConfig> configs;
        configs.reserve(count);
        for (std::size_t idx = 0; idx < count; ++idx)
        {
            ReservoirConfig c = base;
            c.seed = seed_start + idx;
            configs.push_back(c);
        }

        const std::size_t workers = mc::ResolveWorkerCount(count, meter.PerCellBytes(), sweep_opts);

        std::cout << "=== HypercubeESN: MC Seed Survey (DIM=" << meter.Dim()
            << " N=" << N << " F=" << meter.Features() << ") ===\n";
        std::cout << "Fixed op-point: sr=" << base.spectral_radius << " leak=" << base.leak_rate
            << " M=" << base.history_depth << " is=" << base.input_scaling
            << " | warmup=" << mc.t_warmup << " collect=" << mc.t_collect
            << " Kmax=" << mc.k_max << " ridge=" << mc.ridge_lambda << "\n";
        std::cout << "seeds [" << seed_start << ".." << seed_end << "] = " << count
            << " cells | " << workers << " workers | ~" << std::fixed << std::setprecision(2)
            << (static_cast<double>(meter.PerCellBytes()) / 1e9) << " GB/cell, est peak ~"
            << (static_cast<double>(meter.PerCellBytes() * workers) / 1e9) << " GB\n\n"
            << std::defaultfloat << std::flush;

        const std::vector<MCResult> results = RunSweep(meter, configs, sweep_opts, MakeProgress(10));
        std::cout << "\n\n";

        // ---- Per-seed table (seed order) ----
        std::cout << "        seed  realSR    TotalMC  k>.5  k>.1  k>.01\n";
        std::cout << "  ----------  ------  ---------  ----  ----  -----\n";
        std::cout << std::fixed;
        for (std::size_t idx = 0; idx < count; ++idx)
        {
            const MCResult& m = results[idx];
            std::cout << "  " << std::setw(10) << (seed_start + idx)
                << "  " << std::setprecision(4) << std::setw(6) << m.realized_sr << "  ";
            if (m.oom)
            {
                std::cout << "  OOM (lower max_workers/ram_budget)\n";
                continue;
            }
            if (!m.pd)
            {
                std::cout << "  not-PD (raise ridge)\n";
                continue;
            }
            std::cout << std::setprecision(3) << std::setw(9) << m.total_mc
                << "  " << std::setw(4) << m.k50
                << "  " << std::setw(4) << m.k10
                << "  " << std::setw(5) << m.k01 << "\n";
        }
        std::cout << std::defaultfloat;

        // ---- Summary over valid (pd && !oom) cells ----
        std::size_t nvalid = 0, best_idx = 0;
        double best_mc = -1.0, sum = 0.0, mn = 0.0, mx = 0.0;
        for (std::size_t idx = 0; idx < count; ++idx)
        {
            const MCResult& m = results[idx];
            if (m.oom || !m.pd) continue;
            if (nvalid == 0) mn = mx = m.total_mc;
            else
            {
                mn = std::min(mn, m.total_mc);
                mx = std::max(mx, m.total_mc);
            }
            sum += m.total_mc;
            if (m.total_mc > best_mc)
            {
                best_mc = m.total_mc;
                best_idx = idx;
            }
            ++nvalid;
        }

        std::cout << "\n=== Summary (" << nvalid << "/" << count << " valid) ===\n";
        if (nvalid == 0)
        {
            std::cout << "  no valid cells (all OOM or not-PD)\n";
            return;
        }
        const double mean = sum / static_cast<double>(nvalid);
        double var = 0.0;
        for (std::size_t idx = 0; idx < count; ++idx)
        {
            const MCResult& m = results[idx];
            if (m.oom || !m.pd) continue;
            const double d = m.total_mc - mean;
            var += d * d;
        }
        const double sd = (nvalid > 1) ? std::sqrt(var / static_cast<double>(nvalid - 1)) : 0.0;
        std::cout << std::fixed << std::setprecision(3)
            << "  best seed = " << (seed_start + best_idx) << "  TotalMC = " << best_mc
            << "  (realSR " << std::setprecision(4) << results[best_idx].realized_sr << ")\n"
            << std::setprecision(3)
            << "  TotalMC over seeds: mean=" << mean << "  std=" << sd
            << "  min=" << mn << "  max=" << mx << "\n"
            << std::defaultfloat << std::flush;

        // ---- Followup: SR-band filtered chart + top-5 ranking ----
        // The per-seed rescale lands each realization at a slightly different
        // realized SR; a few seeds drift far enough from the target that their MC
        // is no longer at the same operating point. This disregards any cell whose
        // realized SR strays more than kSrBandFrac from the target (base.spectral_radius),
        // then re-summarizes and ranks the survivors. The reporting above is unchanged.
        constexpr double kSrBandFrac = 0.0005; // ±0.05% of target sr
        const double target_sr = static_cast<double>(base.spectral_radius);
        const double sr_tol = kSrBandFrac * target_sr;

        std::vector<std::size_t> kept;
        kept.reserve(count);
        for (std::size_t idx = 0; idx < count; ++idx)
        {
            const MCResult& m = results[idx];
            if (m.oom || !m.pd) continue;
            if (std::abs(static_cast<double>(m.realized_sr) - target_sr) <= sr_tol)
                kept.push_back(idx);
        }

        std::cout << "\n=== Followup: realSR within " << std::fixed << std::setprecision(2)
            << (kSrBandFrac * 100.0) << "% of target sr=" << std::setprecision(4) << target_sr
            << " (band [" << (target_sr - sr_tol) << ", " << (target_sr + sr_tol) << "]) ===\n"
            << std::defaultfloat;
        std::cout << "Kept " << kept.size() << "/" << nvalid
            << " valid cells (" << (nvalid - kept.size()) << " dropped as off-band).\n\n";

        if (kept.empty())
        {
            std::cout << "  no cells inside the SR band\n" << std::flush;
            return;
        }

        std::cout << "        seed  realSR    TotalMC  k>.5  k>.1  k>.01\n";
        std::cout << "  ----------  ------  ---------  ----  ----  -----\n";
        std::cout << std::fixed;
        for (std::size_t idx : kept)
        {
            const MCResult& m = results[idx];
            std::cout << "  " << std::setw(10) << (seed_start + idx)
                << "  " << std::setprecision(4) << std::setw(6) << m.realized_sr
                << "  " << std::setprecision(3) << std::setw(9) << m.total_mc
                << "  " << std::setw(4) << m.k50
                << "  " << std::setw(4) << m.k10
                << "  " << std::setw(5) << m.k01 << "\n";
        }
        std::cout << std::defaultfloat;

        double f_sum = 0.0, f_mn = results[kept[0]].total_mc, f_mx = results[kept[0]].total_mc;
        for (std::size_t idx : kept)
        {
            const double v = results[idx].total_mc;
            f_sum += v;
            f_mn = std::min(f_mn, v);
            f_mx = std::max(f_mx, v);
        }
        const double f_mean = f_sum / static_cast<double>(kept.size());
        double f_var = 0.0;
        for (std::size_t idx : kept)
        {
            const double d = results[idx].total_mc - f_mean;
            f_var += d * d;
        }
        const double f_sd = (kept.size() > 1)
                                ? std::sqrt(f_var / static_cast<double>(kept.size() - 1))
                                : 0.0;
        std::cout << std::fixed << std::setprecision(3)
            << "\n  TotalMC over band seeds: mean=" << f_mean << "  std=" << f_sd
            << "  min=" << f_mn << "  max=" << f_mx << "\n"
            << std::defaultfloat;

        std::vector<std::size_t> ranked = kept;
        std::sort(ranked.begin(), ranked.end(),
                  [&](std::size_t a, std::size_t b) { return results[a].total_mc > results[b].total_mc; });
        const std::size_t top_n = std::min<std::size_t>(5, ranked.size());

        std::cout << "\n  Top " << top_n << " (band, by TotalMC):\n";
        std::cout << "  rank        seed  realSR    TotalMC  k>.5  k>.1  k>.01\n";
        std::cout << "  ----  ----------  ------  ---------  ----  ----  -----\n";
        std::cout << std::fixed;
        for (std::size_t r = 0; r < top_n; ++r)
        {
            const std::size_t idx = ranked[r];
            const MCResult& m = results[idx];
            std::cout << "  " << std::setw(4) << (r + 1)
                << "  " << std::setw(10) << (seed_start + idx)
                << "  " << std::setprecision(4) << std::setw(6) << m.realized_sr
                << "  " << std::setprecision(3) << std::setw(9) << m.total_mc
                << "  " << std::setw(4) << m.k50
                << "  " << std::setw(4) << m.k10
                << "  " << std::setw(5) << m.k01 << "\n";
        }
        std::cout << std::defaultfloat << std::flush;
    }

    // ---------------------------------------------------------------------------
    // Mode 4: per-lag r²(k) curves for several depths, side-by-side.
    // ---------------------------------------------------------------------------
    // Confirms whether a Total-MC dip at some M is real dynamics (r²(1) intact,
    // smoothly faster-decaying tail) or an indexing/injection glitch (a
    // discontinuity, or an across-the-board depression hitting the lag-1 term).
    [[maybe_unused]] void RunDepthProbe(const MemoryCapacityMeter& meter, ReservoirConfig base,
                       float sr, float leak, const std::vector<std::size_t>& depths,
                       std::size_t kmax_probe)
    {
        const MCConfig& mc = meter.Config();
        const std::size_t N = meter.Size();
        kmax_probe = std::min(kmax_probe, mc.k_max);

        base.spectral_radius = sr;
        base.leak_rate = leak;

        std::cout << "=== Per-lag r2(k) probe (DIM=" << meter.Dim() << " N=" << N
            << " F=" << meter.Features() << " sr=" << sr << " leak=" << leak << ") ===\n";
        std::cout << "Same drive/split/ridge as the sweep; only M varies.\n\n";

        std::vector<MCResult> curves(depths.size());
        for (std::size_t d = 0; d < depths.size(); ++d)
        {
            ReservoirConfig c = base;
            c.history_depth = depths[d];
            curves[d] = meter.Measure(c, MeasureOptions{
                                          /*early_stop=*/false, /*kmax=*/kmax_probe
                                      });

            const MCResult& m = curves[d];
            if (!m.pd)
            {
                std::cout << "  M=" << depths[d] << ": train Gram not PD — skipped\n";
                continue;
            }
            double partial = 0.0;
            int first_below_half = 0, first_below_001 = 0;
            for (std::size_t k = 0; k < kmax_probe; ++k)
            {
                partial += m.r2[k];
                if (!first_below_half && m.r2[k] < 0.5) first_below_half = static_cast<int>(k + 1);
                if (!first_below_001 && m.r2[k] < 0.01) first_below_001 = static_cast<int>(k + 1);
            }
            std::cout << std::fixed << std::setprecision(4)
                << "  M=" << std::setw(2) << depths[d]
                << "  realSR=" << m.realized_sr
                << "  r2(1)=" << m.r2[0]
                << "  first k<0.5=" << std::setw(3) << first_below_half
                << "  first k<0.01=" << std::setw(3) << first_below_001
                << "  Σr2[1.." << kmax_probe << "]=" << std::setprecision(3) << partial << "\n";
        }

        // Side-by-side per-lag table.
        std::cout << "\n    k";
        for (std::size_t d = 0; d < depths.size(); ++d)
            std::cout << "      M=" << std::setw(2) << depths[d];
        std::cout << "\n  ---";
        for (std::size_t d = 0; d < depths.size(); ++d) std::cout << "  --------";
        std::cout << "\n" << std::fixed << std::setprecision(4);
        for (std::size_t k = 0; k < kmax_probe; ++k)
        {
            std::cout << "  " << std::setw(3) << (k + 1);
            for (std::size_t d = 0; d < depths.size(); ++d)
                std::cout << "  " << std::setw(8) << (curves[d].pd ? curves[d].r2[k] : 0.0);
            std::cout << "\n";
        }
        std::cout << std::defaultfloat << std::flush;
    }
} // namespace

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    // ---- Experiment definition (the MCConfig fixes drive/split/ridge/lags) ----
    // Defaults mirror a representative text-LM operating point so MC numbers are
    // comparable to it. Edit fields here to change the *experiment*; edit the
    // ReservoirConfig below to change the *operating point*.
    MCConfig mccfg;
    // mccfg.k_max = 2000;  // (defaults shown in MemoryCapacity.h)

    // ---- Base reservoir operating point ----
    // The √DIM-normalized baseline is ~0.069 ≈ 0.02·√12 (reproduces the
    // pre-normalization is=0.02 MC grid); this active run probes slightly below it
    // at 0.06. Raise input_scaling to probe the hard-drive / supercritical regime.
    constexpr std::size_t DIM = 11;

    ReservoirConfig base;
    base.dim = DIM;
    base.seed = 738956;
    base.num_inputs = 1;
    base.spectral_radius = 0.99f;
    base.leak_rate = 1.0f;
    base.input_scaling = 0.06f;
    base.history_depth = 8;

    MemoryCapacityMeter meter(DIM, mccfg);

    // --- Mode 1: single detailed run with the full per-lag r²(k) table ---
    // RunDetailed(meter, base);

    // --- Mode 4: per-lag r²(k) shape, several depths side-by-side ---
    //   RunDepthProbe(meter, base, 0.95f, 1.00f, {1, 2, 4}, 80);

    // --- Mode 3: seed survey at the base op-point (inclusive seed range) ---
    //   RunSeedSurvey(meter, base, 73890, 73890 + 10);

    // --- Mode 2: sr × leak × history-depth grid (hist must be powers of 2) ---
    // The base op-point supplies seed + input_scaling; the grid varies sr/leak/hist.
    RunGridSweep(meter, base,
                      {0.9f, 0.95f, 1.0f, 1.1f}, // spectral radii
                      {1.00f}, // leak rates
                      {1, 2, 4, 8, 16, 32, 64}); // history depths (M)

    return 0;
}
