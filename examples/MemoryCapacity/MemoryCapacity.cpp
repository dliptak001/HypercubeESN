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
///   RunDetailed   — one operating point, summary + sparse per-lag r²(k).
///   RunGridSweep  — sr × leak × history_depth grid; ordered table + pivots.
///   RunSeedSurvey — fixed op-point, sweep reservoir seed; realization variance.
///   RunDepthProbe — per-lag r²(k) curves for several depths, side-by-side.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
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

    // -------------------------------------------------------------------------
    // Shared reporting helpers
    // -------------------------------------------------------------------------

    void PrintFSF(const ReservoirConfig& base)
    {
        if (base.full_state_feedback)
            std::cout << "FSF: ON   fsf_seed=" << base.fsf_seed
                      << "  fsf_scaling=" << base.fsf_scaling << "\n";
        else
            std::cout << "FSF: OFF\n";
    }

    /// Self-describing banner used by every mode (meter + op-point + FSF).
    void PrintRunHeader(const char* title,
                        const MemoryCapacityMeter& meter,
                        const ReservoirConfig& base)
    {
        const MCConfig& mc = meter.Config();
        const double train_over_f =
            static_cast<double>(meter.TrainRows()) /
            static_cast<double>(std::max<std::size_t>(1, meter.Features()));

        std::cout << "=== HypercubeESN: " << title << " ===\n";
        PrintFSF(base);
        std::cout << "Meter   : warmup=" << mc.t_warmup
                  << "  collect=" << mc.t_collect
                  << "  Kmax=" << mc.k_max
                  << "  ridge=" << mc.ridge_lambda
                  << "  train_frac=" << mc.train_frac
                  << "  input_seed=0x" << std::hex << mc.input_seed << std::dec << "\n";
        std::cout << "Layout  : DIM=" << meter.Dim()
                  << "  N=" << meter.Size()
                  << "  F=" << meter.Features()
                  << "  M_usable=" << meter.Samples()
                  << "  M_train=" << meter.TrainRows()
                  << "  M_test=" << meter.TestRows()
                  << "  train/F=" << std::fixed << std::setprecision(2) << train_over_f
                  << std::defaultfloat << "\n";
        std::cout << "Op-point: sr=" << base.spectral_radius
                  << "  leak=" << base.leak_rate
                  << "  is=" << base.input_scaling
                  << "  hist=" << base.history_depth
                  << "  seed=" << base.seed << "\n";
        // ASCII-only metrics glossary: Windows consoles often mis-decode UTF-8
        // (CP1252/OEM), turning Σ/²/× into garbage in CLion and cmd.
        std::cout << "Metrics : TotalMC = sum r^2(k)  |  MC/F = util. of F features"
                  << "  |  k>.5/.1/.01 = last lag above threshold"
                  << "  |  realSR = post-rescale SR\n";
        std::cout << "          open tail (*) = r^2 still above decay floor at last scored lag"
                  << " -> TotalMC is a lower bound\n\n";
    }

    /// Live progress on stderr (keeps stdout tables clean in IDEs).
    auto MakeProgress(std::size_t every = 1)
    {
        return [every](std::size_t done, std::size_t total)
        {
            if (done % every == 0 || done == total)
                std::cerr << "\r  completed " << done << "/" << total << " cells" << std::flush;
            if (done == total)
                std::cerr << "\n";
        };
    }

    /// Tail marker for tables: '*' when open_tail (TotalMC lower bound),
    /// 'e' when early-stopped with closed tail, ' ' otherwise.
    char TailMark(const MCResult& m)
    {
        if (m.oom || !m.pd) return ' ';
        if (m.open_tail) return '*';
        if (m.early_stopped) return 'e';
        return ' ';
    }

    void PrintMcCell(const MCResult& m, std::size_t F, bool with_horizons = true)
    {
        if (m.oom)
        {
            std::cout << "  OOM (lower max_workers/ram_budget)";
            return;
        }
        if (!m.pd)
        {
            std::cout << "  not-PD (raise ridge)";
            return;
        }
        const double util = (F > 0) ? (m.total_mc / static_cast<double>(F)) : 0.0;
        std::cout << std::fixed
                  << std::setw(9) << std::setprecision(3) << m.total_mc
                  << TailMark(m)
                  << "  " << std::setw(6) << std::setprecision(3) << util;
        if (with_horizons)
        {
            std::cout << "  " << std::setw(4) << m.k50
                      << "  " << std::setw(4) << m.k10
                      << "  " << std::setw(5) << m.k01;
        }
    }

    void PrintOpenTailLegend(const std::vector<MCResult>& results)
    {
        std::size_t n_open = 0, n_early = 0, n_full = 0, n_bad = 0;
        for (const MCResult& m : results)
        {
            if (m.oom || !m.pd) { ++n_bad; continue; }
            if (m.open_tail) ++n_open;
            else if (m.early_stopped) ++n_early;
            else ++n_full;
        }
        std::cout << "\nTail notes: * = open tail (TotalMC lower bound; r^2 still live at last lag)"
                  << "  e = early-stop (curve decayed)"
                  << "  (blank) = full k_max scored, closed tail\n";
        std::cout << "  cells: open*=" << n_open
                  << "  early-e=" << n_early
                  << "  full=" << n_full
                  << "  bad=" << n_bad << "\n";
    }

    // Whether to include lag k in a sparse detailed dump.
    bool SparseLag(std::size_t k, std::size_t k_max)
    {
        if (k <= 20) return true;
        if (k <= 100 && (k % 5) == 0) return true;
        if (k <= 500 && (k % 10) == 0) return true;
        if ((k % 50) == 0) return true;
        return k == k_max;
    }

    // ---------------------------------------------------------------------------
    // Mode 1: one operating point, summary + sparse per-lag r²(k).
    // ---------------------------------------------------------------------------
    [[maybe_unused]] void RunDetailed(const MemoryCapacityMeter& meter, const ReservoirConfig& rcfg)
    {
        const MCConfig& mc = meter.Config();
        const std::size_t F = meter.Features();

        PrintRunHeader("Memory Capacity (detailed)", meter, rcfg);
        std::cout << "Measuring (full " << mc.k_max << "-lag curve, early_stop=off) ... "
                  << std::flush;
        const MCResult r = meter.Measure(rcfg, MeasureOptions{
                                             /*early_stop=*/false, /*kmax=*/0
                                         });
        std::cout << "done.\n\n";

        if (!r.pd)
        {
            std::cerr << "ERROR: train Gram not positive definite. Increase ridge lambda.\n";
            return;
        }

        const double util = r.total_mc / static_cast<double>(F);
        std::cout << "=== Summary ===\n";
        std::cout << std::fixed << std::setprecision(3)
                  << "Total MC = " << r.total_mc
                  << "  MC/F = " << util
                  << "  (F=" << F << ")\n";
        std::cout << "realSR   = " << std::setprecision(4) << r.realized_sr << "\n";
        std::cout << "Last lag with r^2 > 0.50 : k=" << r.k50 << "\n";
        std::cout << "Last lag with r^2 > 0.10 : k=" << r.k10 << "\n";
        std::cout << "Last lag with r^2 > 0.01 : k=" << r.k01 << "\n";
        std::cout << "lags scored = " << r.lags_scored
                  << "  r2_tail = " << std::setprecision(4) << r.r2_tail;
        if (r.open_tail)
            std::cout << "  OPEN TAIL (*) - TotalMC is a lower bound (memory still live at k_max)\n";
        else
            std::cout << "  closed tail\n";
        std::cout << std::defaultfloat;

        std::cout << "\n  k    r2(test)   (sparse: 1..20 all, then every 5/10/50 + k_max)\n";
        std::cout << "  ---  --------\n";
        std::cout << std::fixed << std::setprecision(4);
        for (std::size_t k = 1; k <= mc.k_max; ++k)
        {
            if (!SparseLag(k, mc.k_max)) continue;
            std::cout << "  " << std::setw(3) << k
                      << "  " << std::setw(8) << r.r2[k - 1] << "\n";
        }
        std::cout << std::defaultfloat << std::flush;
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
        const std::size_t F = meter.Features();
        const std::size_t nsr = spectral_radii.size();
        const std::size_t nleak = leak_rates.size();
        const std::size_t nhist = history_depths.size();
        const std::size_t cells = nsr * nleak * nhist;
        if (cells == 0) return;

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

        PrintRunHeader("MC Profile (grid)", meter, base);
        std::cout << "Grid    : " << nsr << " sr x " << nleak << " leak x " << nhist
                  << " hist = " << cells << " cells  |  " << workers
                  << " workers  |  ~" << std::fixed << std::setprecision(2)
                  << (static_cast<double>(meter.PerCellBytes()) / 1e9) << " GB/cell, est peak ~"
                  << (static_cast<double>(meter.PerCellBytes() * workers) / 1e9) << " GB\n"
                  << "Axes    : sr={";
        for (std::size_t i = 0; i < nsr; ++i)
            std::cout << (i ? "," : "") << spectral_radii[i];
        std::cout << "}  leak={";
        for (std::size_t j = 0; j < nleak; ++j)
            std::cout << (j ? "," : "") << leak_rates[j];
        std::cout << "}  M={";
        for (std::size_t k = 0; k < nhist; ++k)
            std::cout << (k ? "," : "") << history_depths[k];
        std::cout << "}  (seed+is fixed from op-point)\n\n" << std::defaultfloat << std::flush;

        const std::vector<MCResult> results = RunSweep(meter, configs, sweep_opts, MakeProgress());
        std::cout << "\n";

        auto cell = [&](std::size_t i, std::size_t j, std::size_t k) -> const MCResult&
        {
            return results[(i * nleak + j) * nhist + k];
        };

        // ---- Results table (ordered: sr, then leak, then hist) ----
        std::cout << "    sr   leak  hist  realSR    TotalMC   MC/F  k>.5  k>.1  k>.01\n";
        std::cout << "  -----  -----  ----  ------  ---------  ------  ----  ----  -----\n";
        for (std::size_t i = 0; i < nsr; ++i)
            for (std::size_t j = 0; j < nleak; ++j)
                for (std::size_t k = 0; k < nhist; ++k)
                {
                    const MCResult& m = cell(i, j, k);
                    std::cout << std::fixed << std::setprecision(2)
                              << "  " << std::setw(5) << spectral_radii[i]
                              << "  " << std::setw(5) << leak_rates[j]
                              << "  " << std::setw(4) << history_depths[k]
                              << "  " << std::setw(6) << std::setprecision(4) << m.realized_sr
                              << "  ";
                    PrintMcCell(m, F);
                    std::cout << "\n";
                }

        // ---- Pivot grids (auto-shape when an axis is a singleton) ----
        auto print_mc_or_na = [&](const MCResult& m)
        {
            if (m.oom || !m.pd)
                std::cout << std::setw(9) << "n/a";
            else
                std::cout << std::setw(8) << std::setprecision(2) << m.total_mc << TailMark(m);
        };

        if (nleak == 1 && nsr >= 1 && nhist >= 1)
        {
            // Default campaign shape: depth × sr (matches MemoryCapacity.md)
            std::cout << "\n=== TotalMC grid (M x sr, leak=" << leak_rates[0]
                      << ", is=" << base.input_scaling << ") ===\n";
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << "M\\sr";
            for (float sr : spectral_radii) std::cout << std::setw(9) << sr;
            std::cout << "\n";
            for (std::size_t k = 0; k < nhist; ++k)
            {
                std::cout << "  " << std::setw(4) << history_depths[k] << "  ";
                for (std::size_t i = 0; i < nsr; ++i)
                    print_mc_or_na(cell(i, 0, k));
                std::cout << "\n";
            }
        }
        else if (nsr == 1 && nleak >= 1 && nhist >= 1)
        {
            std::cout << "\n=== TotalMC grid (M x leak, sr=" << spectral_radii[0]
                      << ", is=" << base.input_scaling << ") ===\n";
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << "M\\leak";
            for (float lk : leak_rates) std::cout << std::setw(9) << lk;
            std::cout << "\n";
            for (std::size_t k = 0; k < nhist; ++k)
            {
                std::cout << "  " << std::setw(4) << history_depths[k] << "  ";
                for (std::size_t j = 0; j < nleak; ++j)
                    print_mc_or_na(cell(0, j, k));
                std::cout << "\n";
            }
        }
        else
        {
            // Full 3-axis: one sr × leak grid per depth
            for (std::size_t k = 0; k < nhist; ++k)
            {
                std::cout << "\n=== TotalMC grid (M=" << history_depths[k]
                          << ", is=" << base.input_scaling << ", rows=sr, cols=leak) ===\n";
                std::cout << std::fixed << std::setprecision(2) << std::setw(9) << "sr\\leak";
                for (float lk : leak_rates) std::cout << std::setw(9) << lk;
                std::cout << "\n";
                for (std::size_t i = 0; i < nsr; ++i)
                {
                    std::cout << "  " << std::setw(5) << std::setprecision(2) << spectral_radii[i]
                              << "  ";
                    for (std::size_t j = 0; j < nleak; ++j)
                        print_mc_or_na(cell(i, j, k));
                    std::cout << "\n";
                }
            }
        }

        PrintOpenTailLegend(results);
        std::cout << std::defaultfloat << std::flush;
    }

    // ---------------------------------------------------------------------------
    // Mode 3: fixed op-point, sweep the reservoir seed over [seed_start, seed_end].
    // ---------------------------------------------------------------------------
    [[maybe_unused]] void RunSeedSurvey(const MemoryCapacityMeter& meter, ReservoirConfig base,
                       std::uint64_t seed_start, std::uint64_t seed_end,
                       const SweepOptions& sweep_opts = {})
    {
        const std::size_t F = meter.Features();

        if (seed_end < seed_start)
        {
            std::cerr << "RunSeedSurvey: seed_end < seed_start - nothing to do.\n";
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

        PrintRunHeader("MC Seed Survey", meter, base);
        std::cout << "Seeds   : [" << seed_start << ".." << seed_end << "] = " << count
                  << " cells  |  " << workers << " workers  |  ~" << std::fixed << std::setprecision(2)
                  << (static_cast<double>(meter.PerCellBytes()) / 1e9) << " GB/cell, est peak ~"
                  << (static_cast<double>(meter.PerCellBytes() * workers) / 1e9) << " GB\n\n"
                  << std::defaultfloat << std::flush;

        const std::vector<MCResult> results = RunSweep(meter, configs, sweep_opts, MakeProgress(10));
        std::cout << "\n";

        std::cout << "        seed  realSR    TotalMC   MC/F  k>.5  k>.1  k>.01\n";
        std::cout << "  ----------  ------  ---------  ------  ----  ----  -----\n";
        for (std::size_t idx = 0; idx < count; ++idx)
        {
            const MCResult& m = results[idx];
            std::cout << "  " << std::setw(10) << (seed_start + idx)
                      << "  " << std::fixed << std::setprecision(4) << std::setw(6) << m.realized_sr
                      << "  ";
            PrintMcCell(m, F);
            std::cout << "\n";
        }
        std::cout << std::defaultfloat;

        // ---- Summary over valid (pd && !oom) cells ----
        std::vector<double> vals;
        vals.reserve(count);
        std::size_t best_idx = 0;
        double best_mc = -1.0;
        double sum_k01 = 0.0;
        for (std::size_t idx = 0; idx < count; ++idx)
        {
            const MCResult& m = results[idx];
            if (m.oom || !m.pd) continue;
            vals.push_back(m.total_mc);
            sum_k01 += static_cast<double>(m.k01);
            if (m.total_mc > best_mc)
            {
                best_mc = m.total_mc;
                best_idx = idx;
            }
        }

        std::cout << "\n=== Summary (" << vals.size() << "/" << count << " valid) ===\n";
        if (vals.empty())
        {
            std::cout << "  no valid cells (all OOM or not-PD)\n";
            return;
        }

        std::sort(vals.begin(), vals.end());
        const double mean = std::accumulate(vals.begin(), vals.end(), 0.0) /
                            static_cast<double>(vals.size());
        double var = 0.0;
        for (double v : vals)
        {
            const double d = v - mean;
            var += d * d;
        }
        const double sd = (vals.size() > 1)
                              ? std::sqrt(var / static_cast<double>(vals.size() - 1))
                              : 0.0;
        const double median = (vals.size() % 2 == 1)
                                  ? vals[vals.size() / 2]
                                  : 0.5 * (vals[vals.size() / 2 - 1] + vals[vals.size() / 2]);
        const double mean_k01 = sum_k01 / static_cast<double>(vals.size());

        std::cout << std::fixed << std::setprecision(3)
                  << "  best seed = " << (seed_start + best_idx) << "  TotalMC = " << best_mc
                  << "  MC/F = " << (best_mc / static_cast<double>(F))
                  << "  (realSR " << std::setprecision(4) << results[best_idx].realized_sr << ")\n"
                  << std::setprecision(3)
                  << "  TotalMC over seeds: mean=" << mean << "  median=" << median
                  << "  std=" << sd << "  min=" << vals.front() << "  max=" << vals.back() << "\n"
                  << "  mean k>.01 = " << std::setprecision(1) << mean_k01 << "\n"
                  << std::defaultfloat << std::flush;

        // ---- Followup: SR-band filtered chart + top-5 ranking ----
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
        std::cout << "Kept " << kept.size() << "/" << vals.size()
                  << " valid cells (" << (vals.size() - kept.size()) << " dropped as off-band).\n\n";

        if (kept.empty())
        {
            std::cout << "  no cells inside the SR band\n" << std::flush;
            PrintOpenTailLegend(results);
            return;
        }

        std::cout << "        seed  realSR    TotalMC   MC/F  k>.5  k>.1  k>.01\n";
        std::cout << "  ----------  ------  ---------  ------  ----  ----  -----\n";
        for (std::size_t idx : kept)
        {
            const MCResult& m = results[idx];
            std::cout << "  " << std::setw(10) << (seed_start + idx)
                      << "  " << std::fixed << std::setprecision(4) << std::setw(6) << m.realized_sr
                      << "  ";
            PrintMcCell(m, F);
            std::cout << "\n";
        }
        std::cout << std::defaultfloat;

        std::vector<double> fvals;
        fvals.reserve(kept.size());
        double f_sum_k01 = 0.0;
        for (std::size_t idx : kept)
        {
            fvals.push_back(results[idx].total_mc);
            f_sum_k01 += static_cast<double>(results[idx].k01);
        }
        std::sort(fvals.begin(), fvals.end());
        const double f_mean = std::accumulate(fvals.begin(), fvals.end(), 0.0) /
                              static_cast<double>(fvals.size());
        double f_var = 0.0;
        for (double v : fvals)
        {
            const double d = v - f_mean;
            f_var += d * d;
        }
        const double f_sd = (fvals.size() > 1)
                                ? std::sqrt(f_var / static_cast<double>(fvals.size() - 1))
                                : 0.0;
        const double f_median = (fvals.size() % 2 == 1)
                                    ? fvals[fvals.size() / 2]
                                    : 0.5 * (fvals[fvals.size() / 2 - 1] + fvals[fvals.size() / 2]);

        std::cout << std::fixed << std::setprecision(3)
                  << "\n  TotalMC over band seeds: mean=" << f_mean
                  << "  median=" << f_median << "  std=" << f_sd
                  << "  min=" << fvals.front() << "  max=" << fvals.back() << "\n"
                  << "  mean k>.01 = " << std::setprecision(1)
                  << (f_sum_k01 / static_cast<double>(kept.size())) << "\n"
                  << std::defaultfloat;

        std::vector<std::size_t> ranked = kept;
        std::sort(ranked.begin(), ranked.end(),
                  [&](std::size_t a, std::size_t b)
                  { return results[a].total_mc > results[b].total_mc; });
        const std::size_t top_n = std::min<std::size_t>(5, ranked.size());

        std::cout << "\n  Top " << top_n << " (band, by TotalMC):\n";
        std::cout << "  rank        seed  realSR    TotalMC   MC/F  k>.5  k>.1  k>.01\n";
        std::cout << "  ----  ----------  ------  ---------  ------  ----  ----  -----\n";
        for (std::size_t r = 0; r < top_n; ++r)
        {
            const std::size_t idx = ranked[r];
            const MCResult& m = results[idx];
            std::cout << "  " << std::setw(4) << (r + 1)
                      << "  " << std::setw(10) << (seed_start + idx)
                      << "  " << std::fixed << std::setprecision(4) << std::setw(6) << m.realized_sr
                      << "  ";
            PrintMcCell(m, F);
            std::cout << "\n";
        }
        PrintOpenTailLegend(results);
        std::cout << std::defaultfloat << std::flush;
    }

    // ---------------------------------------------------------------------------
    // Mode 4: per-lag r²(k) curves for several depths, side-by-side.
    // ---------------------------------------------------------------------------
    [[maybe_unused]] void RunDepthProbe(const MemoryCapacityMeter& meter, ReservoirConfig base,
                       float sr, float leak, const std::vector<std::size_t>& depths,
                       std::size_t kmax_probe)
    {
        const MCConfig& mc = meter.Config();
        const std::size_t F = meter.Features();
        kmax_probe = std::min(kmax_probe, mc.k_max);

        base.spectral_radius = sr;
        base.leak_rate = leak;

        PrintRunHeader("Per-lag r2(k) depth probe", meter, base);
        std::cout << "Probe   : sr=" << sr << "  leak=" << leak
                  << "  kmax_probe=" << kmax_probe
                  << "  depths={";
        for (std::size_t d = 0; d < depths.size(); ++d)
            std::cout << (d ? "," : "") << depths[d];
        std::cout << "}  (early_stop=off; same drive/split/ridge)\n\n" << std::flush;

        std::vector<MCResult> curves(depths.size());
        for (std::size_t d = 0; d < depths.size(); ++d)
        {
            ReservoirConfig c = base;
            c.history_depth = depths[d];
            std::cerr << "  measuring M=" << depths[d] << " (" << (d + 1) << "/"
                      << depths.size() << ") ...\n";
            curves[d] = meter.Measure(c, MeasureOptions{
                                          /*early_stop=*/false, /*kmax=*/kmax_probe
                                      });

            const MCResult& m = curves[d];
            if (!m.pd)
            {
                std::cout << "  M=" << depths[d] << ": train Gram not PD - skipped\n";
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
                      << "  sum_r2[1.." << kmax_probe << "]=" << std::setprecision(3) << partial
                      << "  MC/F=" << (partial / static_cast<double>(F));
            if (m.open_tail) std::cout << "  *open";
            std::cout << "\n";
        }

        // Side-by-side per-lag table (sparse rows for long probes).
        std::cout << "\n    k";
        for (std::size_t d = 0; d < depths.size(); ++d)
            std::cout << "      M=" << std::setw(2) << depths[d];
        std::cout << "\n  ---";
        for (std::size_t d = 0; d < depths.size(); ++d) std::cout << "  --------";
        std::cout << "\n" << std::fixed << std::setprecision(4);
        for (std::size_t k = 0; k < kmax_probe; ++k)
        {
            if (kmax_probe > 80 && !SparseLag(k + 1, kmax_probe)) continue;
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
    // Defaults match MemoryCapacity.md Results (DIM 11 grid). Edit fields here to
    // change the *experiment*; edit the ReservoirConfig below for the op-point.
    MCConfig mccfg;
    // mccfg.k_max = 2000;  // (defaults shown in MemoryCapacity.h)

    // ---- Base reservoir operating point ----
    // Doc Results table: is=0.06 (weak drive / memory-margin regime).
    // For A_lorentz free-run op-points try ~0.2; retune per activation and task.
    constexpr std::size_t DIM = 11;

    ReservoirConfig base;
    base.dim = DIM;
    base.seed = 738956;
    base.num_inputs = 1;
    base.spectral_radius = 0.99f;
    base.leak_rate = 1.0f;
    base.input_scaling = 0.06f; // tanh baseline (md Results); A(x) often wants ~0.2
    base.history_depth = 8;

    // Full-state linear feedback (internal). Edit these three for A/B.
    base.full_state_feedback = false;
    base.fsf_seed = 4415756;
    base.fsf_scaling = 0.003f;

    MemoryCapacityMeter meter(DIM, mccfg);

    // --- Mode 1: single detailed run (summary + sparse r²) ---
    // RunDetailed(meter, base);

    // --- Mode 4: per-lag r²(k) shape, several depths side-by-side ---
    //   RunDepthProbe(meter, base, 0.95f, 1.00f, {1, 2, 4}, 80);

    // --- Mode 3: seed survey at the base op-point (inclusive seed range) ---
    //   RunSeedSurvey(meter, base, 73890, 73890 + 10);

    // --- Mode 2: sr × leak × history-depth grid (hist must be powers of 2) ---
    // Default campaign: matches MemoryCapacity.md Results (leak singleton → M×sr pivot).
    RunGridSweep(meter, base,
                      {0.9f, 0.95f, 1.0f, 1.1f}, // spectral radii
                      {1.00f},                   // leak rates
                      {1, 2, 4, 8, 16, 32, 64}); // history depths (M)

    return 0;
}
