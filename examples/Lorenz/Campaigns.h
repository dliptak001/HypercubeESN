#pragma once

#include "Lorenz.h"

#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// Campaign entry points -- plain functions. Call them from main.cpp (copy/paste).
// No CLI framework: pick what to run by editing main.
// First argument is always reservoir DIM (N = 2^DIM); restored on exit.
// ---------------------------------------------------------------------------

/// Aggregates from one multi-seed survey (means are mean-of-trial-means).
/// Filled by Campaign_SeedSurvey when @p out is non-null.
struct SurveySummary
{
    bool ok = false;
    size_t history_depth = 0; ///< config::HISTORY_DEPTH at survey time
    size_t num_inputs = 0;    ///< drive channels (always kNumDriveChannels)
    size_t num_trials = 0;    ///< parallel ESN seeds
    int num_runs = 0;         ///< free-runs per trial
    size_t n_trials_ok = 0;   ///< trials with >=1 valid free-run
    uint64_t base_seed = 0;
    uint64_t orbit_seed = 0;

    /// Trial freerun means use top 10% of ICs per metric (see README).
    /// Primary metrics only: VPT, duty, VPT*duty, RMSE.
    double mean_vpt = 0;
    double std_vpt = 0;
    double mean_rmse = 0;
    double std_rmse = 0;
    double mean_duty = 0;
    double std_duty = 0;
    double mean_vpt_x_duty = 0;
    double std_vpt_x_duty = 0;

    double wall_seconds = 0;  ///< survey wall time (parallel pool)
};

/// Multi-seed train + free-run survey. Prints reports to stdout; progress on stderr.
/// @param dim  Reservoir hypercube dim (N = 2^dim). Restored on exit.
/// @param out  If non-null, filled with crunched aggregates (for roll-ups).
/// @param completion_beep  Windows Beep when finished (false for multi-step sweeps).
/// @return 0 on success.
int Campaign_SeedSurvey(size_t dim,
                        size_t num_threads = 0,
                        int num_runs = 50,
                        uint64_t base_seed = 21978990,
                        uint64_t orbit_seed = 72983498,
                        bool completion_beep = true,
                        SurveySummary* out = nullptr);

/// Single-seed train/load + free-run CSV dumps under RUNS_DIR/traces/.
/// CSV columns: step, lt, err, locked, pred_x/y/z, true_x/y/z, drive_*.
/// When @p target_orbit != 0: free-run that orbit once (fixed seed), print every
/// generative step to stdout, write one CSV for Python overlay plots.
/// When @p target_orbit == 0: dump up to @p max_freeruns free-runs (CSV only).
/// @param dim  Reservoir hypercube dim (N = 2^dim). Restored on exit.
int Campaign_Trace(size_t dim,
                   uint64_t esn_seed,
                   int max_freeruns = 30,
                   uint64_t target_orbit = 0,
                   uint64_t orbit_seed = 72983498);

/// Load readout weights from file + one free-run on an explicit attractor IC
/// (not an orbit seed). Does **not** train. Weights stem: @p weights_stem if
/// non-null/non-empty, else config::LOAD_WEIGHTS_STEM (path without .hcnw).
/// Edge warmup then free-run past span on the IC stream.
/// Writes one CSV under RUNS_DIR/traces/ (plottable with plot_freerun_overlay.py).
/// Restores DIM / HISTORY_DEPTH / SPECTRAL_RADIUS / INPUT_SCALING on exit.
/// Channel gains are fixed (`config::INPUT_SCALE_CH`).
/// @param dim            Reservoir hypercube dim (N = 2^dim); must match model.
/// @param history_depth  Reservoir delay-line M (1..64); must match model.
/// @param esn_seed       Reservoir seed; must match the run that produced weights.
/// @param ic_x,ic_y,ic_z Attractor IC (same space as IcFromOrbitSeed).
/// @param weights_stem   Optional override of config::LOAD_WEIGHTS_STEM.
/// @param spectral_radius If > 0, set config::SPECTRAL_RADIUS; <= 0 keeps config.
/// @param input_scaling   If > 0, set config::INPUT_SCALING; <= 0 keeps config.
int FreeRun(size_t dim,
            size_t history_depth,
            uint64_t esn_seed,
            double ic_x, double ic_y, double ic_z,
            const char* weights_stem = nullptr,
            float spectral_radius = 0.f,
            float input_scaling = 0.f);

/// Aggregate from one @ref FreeRunSurvey (top-10% freerun pool means).
struct FreeRunSurveySummary
{
    bool ok = false;
    uint64_t esn_seed = 0;
    size_t n_valid = 0;
    double mean_vpt = 0;
    double mean_duty = 0;
    double mean_vpt_x_duty = 0; ///< primary seed-ranking score
    double mean_rmse = 0;
    double best_vpt_x_duty = 0;
    double best_ic_x = 0, best_ic_y = 0, best_ic_z = 0;
    uint64_t best_orbit_seed = 0;
};

/// Load weights, free-run @p num_runs remixed orbits (from @p orbit_seed),
/// print aggregate stats (top-10% pool) and top orbits by VPT*duty with IC
/// triples ready for @ref FreeRun. No train; no per-step CSV (use FreeRun to plot).
/// Writes a leaderboard CSV under RUNS_DIR/surveys/.
/// Restores DIM / HISTORY_DEPTH / SPECTRAL_RADIUS / INPUT_SCALING on exit.
/// Channel gains are fixed (`config::INPUT_SCALE_CH`).
/// @param out  Optional; filled with top-10% means and top freerun IC.
/// @param spectral_radius If > 0, set config::SPECTRAL_RADIUS; <= 0 keeps config.
/// @param input_scaling   If > 0, set config::INPUT_SCALING; <= 0 keeps config.
int FreeRunSurvey(size_t dim,
                  size_t history_depth,
                  uint64_t esn_seed,
                  int num_runs,
                  uint64_t orbit_seed = 72983498,
                  const char* weights_stem = nullptr,
                  int top_k = 10,
                  FreeRunSurveySummary* out = nullptr,
                  float spectral_radius = 0.f,
                  float input_scaling = 0.f);

/// Train-only complement to @ref FreeRun: remixed orbits for @p epochs starting
/// from @p target_orbit remix seed; save readout to @p weights_stem (or default
/// under MODEL_SAVE_DIR). No free-run. Restores DIM, HISTORY_DEPTH, EPOCHS on exit.
int Train(size_t dim,
          size_t history_depth,
          uint64_t esn_seed,
          uint64_t target_orbit,
          size_t epochs,
          const char* weights_stem = nullptr);

/// Parallel train+freerun seed search (no weight save). Spawns @p num_threads
/// workers over @p num_seeds ESN seeds derived from @p base_esn_seed via
/// SplitMix64 (decorrelated; not base+i). Each worker: Train in memory →
/// @p freerun_runs Unseen freeruns; aggregates use top-10% freerun pool per
/// metric. Interim seed reports suppressed; stderr heartbeats only (mutexed).
/// @p base_orbit_seed is the **shared remix root** for train and freerun
/// (each phase advances its own orbit stream via mix64 — not a single orbit).
/// Final report: full seed table + top_k highlights for VPT, duty, and
/// VPT*duty (stdout and surveys/*.csv + *.txt). Dynamics overrides:
/// @p spectral_radius / @p input_scaling >0 set for the run (RAII restore);
/// 0 keeps config. Always trains (no load / no save). Refuses
/// LOAD_TRAINED_WEIGHTS and SAVE_TRAINED_WEIGHTS. Requires
/// Lorenz::kReadoutNumThreads == 1 (HCNN single-threaded per network).
/// @p num_threads is capped to hardware_concurrency and to @p num_seeds.
int ParallelSeedSweep(size_t dim,
                      size_t history_depth,
                      uint64_t base_esn_seed,
                      size_t num_seeds,
                      size_t num_threads,
                      size_t epochs,
                      int freerun_runs,
                      uint64_t base_orbit_seed = 9333312947715283458ull,
                      int top_k = 10,
                      float spectral_radius = 0.f,
                      float input_scaling = 0.f);

/// Parallel orbit search for one ESN seed (train once, no lasting weight save).
/// Trains @p base_esn_seed once (remix root @p base_orbit_seed), then freeruns
/// @p num_orbits fixed orbits in parallel: orbit_i = Mix64(base_orbit ^ FNV*(i+1)).
/// One freerun per orbit (raw metrics, no top-10% pool). Ranks orbit seeds by
/// VPT / duty / VPT*duty. Survey CSV/TXT (and the matching stdout table) keep
/// only the top 100 and bottom 10 by VPT*duty — not the full N-orbit table.
/// Temporary readout stem only so workers can load the trained model; removed
/// after the sweep. Same dynamics overrides / refuse LOAD+SAVE flags / HCNN
/// threads=1 / thread cap as ParallelSeedSweep. @p num_threads capped to HW
/// and @p num_orbits.
int ParallelOrbitSweep(size_t dim,
                       size_t history_depth,
                       uint64_t base_esn_seed,
                       uint64_t base_orbit_seed,
                       size_t num_orbits,
                       size_t num_threads,
                       size_t epochs,
                       int top_k = 10,
                       float spectral_radius = 0.f,
                       float input_scaling = 0.f);
