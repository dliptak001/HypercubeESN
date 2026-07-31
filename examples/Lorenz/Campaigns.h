#pragma once

#include "Lorenz.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>

// ---------------------------------------------------------------------------
// Campaign entry points -- plain functions. Call them from main.cpp (copy/paste).
// No CLI framework: pick what to run by editing main.
// First argument is always reservoir DIM (N = 2^DIM); restored on exit.
// ---------------------------------------------------------------------------

/// Aggregates from one multi-seed survey (means are mean-of-trial-means).
/// Filled by Campaign_SeedSurvey when @p out is non-null; used by M-sweep roll-up.
struct SurveySummary
{
    bool ok = false;
    size_t history_depth = 0; ///< config::HISTORY_DEPTH at survey time
    DriveLayout drive_layout = DriveLayout::XyzXz;
    size_t num_inputs = 0;    ///< channels for drive_layout
    size_t num_trials = 0;    ///< parallel ESN seeds
    int num_runs = 0;         ///< free-runs per trial
    size_t n_trials_ok = 0;   ///< trials with >=1 valid free-run
    uint64_t base_seed = 0;
    uint64_t orbit_seed = 0;

    /// Trial freerun means use best half of ICs per metric (see README).
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
/// Restores DIM / HISTORY_DEPTH / SPECTRAL_RADIUS / INPUT_SCALING /
/// DRIVE_LAYOUT / INPUT_SCALE_CH on exit.
/// @param dim            Reservoir hypercube dim (N = 2^dim); must match model.
/// @param history_depth  Reservoir delay-line M (1..64); must match model.
/// @param esn_seed       Reservoir seed; must match the run that produced weights.
/// @param ic_x,ic_y,ic_z Attractor IC (same space as IcFromOrbitSeed).
/// @param weights_stem   Optional override of config::LOAD_WEIGHTS_STEM.
/// @param spectral_radius If > 0, set config::SPECTRAL_RADIUS; <= 0 keeps config.
/// @param input_scaling   If > 0, set config::INPUT_SCALING; <= 0 keeps config.
/// @param drive_layout    If set, set config::DRIVE_LAYOUT; nullopt keeps config.
/// @param drive_gains     If non-empty, set config::INPUT_SCALE_CH (exact n_in entries).
int FreeRun(size_t dim,
            size_t history_depth,
            uint64_t esn_seed,
            double ic_x, double ic_y, double ic_z,
            const char* weights_stem = nullptr,
            float spectral_radius = 0.f,
            float input_scaling = 0.f,
            std::optional<DriveLayout> drive_layout = std::nullopt,
            std::initializer_list<float> drive_gains = {});

/// Aggregate from one @ref FreeRunSurvey (best-half freerun pool means).
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
/// print aggregate stats (best-half pool) and top orbits by VPT*duty with IC
/// triples ready for @ref FreeRun. No train; no per-step CSV (use FreeRun to plot).
/// Writes a leaderboard CSV under RUNS_DIR/surveys/.
/// Restores DIM / HISTORY_DEPTH / SPECTRAL_RADIUS / INPUT_SCALING /
/// DRIVE_LAYOUT / INPUT_SCALE_CH on exit.
/// @param out  Optional; filled with best-half means and top freerun IC.
/// @param spectral_radius If > 0, set config::SPECTRAL_RADIUS; <= 0 keeps config.
/// @param input_scaling   If > 0, set config::INPUT_SCALING; <= 0 keeps config.
/// @param drive_layout    If set, set config::DRIVE_LAYOUT; nullopt keeps config.
/// @param drive_gains     If non-empty, set config::INPUT_SCALE_CH (exact n_in entries).
int FreeRunSurvey(size_t dim,
                  size_t history_depth,
                  uint64_t esn_seed,
                  int num_runs,
                  uint64_t orbit_seed = 72983498,
                  const char* weights_stem = nullptr,
                  int top_k = 10,
                  FreeRunSurveySummary* out = nullptr,
                  float spectral_radius = 0.f,
                  float input_scaling = 0.f,
                  std::optional<DriveLayout> drive_layout = std::nullopt,
                  std::initializer_list<float> drive_gains = {});

/// Train-only complement to @ref FreeRun: remixed orbits for @p epochs starting
/// from @p target_orbit remix seed; save readout to @p weights_stem (or default
/// under MODEL_SAVE_DIR). No free-run. Restores DIM, HISTORY_DEPTH, EPOCHS on exit.
int Train(size_t dim,
          size_t history_depth,
          uint64_t esn_seed,
          uint64_t target_orbit,
          size_t epochs,
          const char* weights_stem = nullptr);

/// For each ESN seed: optional Train → FreeRunSurvey; rank seeds by mean VPT*duty
/// (best-half freerun pool). Weight stem per seed:
///   {MODEL_SAVE_DIR}/lorenz_seed{S}_D{dim}_M{M}
/// Restores DIM / HISTORY_DEPTH / EPOCHS / SPECTRAL_RADIUS / INPUT_SCALING /
/// INPUT_SCALE_CH on exit.
/// @param do_train          If true, Train each seed first; if false, only survey existing stems.
/// @param spectral_radius   If > 0, set config::SPECTRAL_RADIUS for the whole sweep.
///                          If <= 0 (default), keep the current config value.
/// @param input_scaling     If > 0, set config::INPUT_SCALING for the whole sweep.
///                          If <= 0 (default), keep the current config value.
/// @param drive_layout      If set, set config::DRIVE_LAYOUT for the sweep
///                          (XyzXz / XyzXy / Quadratic8). nullopt (default) keeps config.
/// @param drive_gains       If non-empty, set config::INPUT_SCALE_CH for the sweep.
///                          Must have exactly NumDriveChannels(active layout) entries
///                          (layout feature order; e.g. XyzXz/XyzXy = 4). Empty (default)
///                          keeps current channel gains.
int SeedSweep(size_t dim,
              size_t history_depth,
              std::initializer_list<uint64_t> esn_seeds,
              size_t epochs,
              int freerun_runs,
              uint64_t train_orbit = 9333312947715283458ull,
              uint64_t freerun_orbit_seed = 72983498ull,
              int top_k = 10,
              bool do_train = true,
              float spectral_radius = 0.f,
              float input_scaling = 0.f,
              std::optional<DriveLayout> drive_layout = std::nullopt,
              std::initializer_list<float> drive_gains = {});

/// Sequential surveys for each M in @p history_depths, then a comparative roll-up
/// table (mean-of-trial-means) computed from SurveySummary rows -- not estimated.
/// @param dim  Reservoir hypercube dim (N = 2^dim), fixed for the whole sweep. Restored on exit.
int Campaign_HistoryDepthSweep(size_t dim,
                               std::initializer_list<size_t> history_depths,
                               size_t num_threads = 0,
                               int num_runs = 50,
                               uint64_t base_seed = 21978990,
                               uint64_t orbit_seed = 72983498);

/// A/B drive layouts at one fixed history depth M: XyzXz (4-in) then Quadratic8 (8-in).
/// Matched dim/M/seeds; trains separately per arm. Code-computed roll-up.
/// @param dim            Reservoir hypercube dim (N = 2^dim). Restored on exit.
/// @param history_depth  Fixed M for both arms (1..64). Restored on exit.
int Campaign_DriveLayoutAB(size_t dim,
                           size_t history_depth,
                           size_t num_threads = 0,
                           int num_runs = 50,
                           uint64_t base_seed = 21978990,
                           uint64_t orbit_seed = 72983498);

/// A/B spectral radius at fixed dim/M: arm A = @p sr_a then arm B = @p sr_b.
/// Matched seeds/drive; trains separately per arm (SeedSurvey).
/// Code-computed roll-up + CSV/TXT under RESULTS_DIR. Restores DIM, M, SR on exit.
/// @param dim            Reservoir hypercube dim (N = 2^dim).
/// @param history_depth  Fixed M for both arms (1..64).
/// @param sr_a, sr_b     Spectral radii (must be finite and > 0).
int Campaign_SpectralRadiusAB(size_t dim,
                              size_t history_depth,
                              float sr_a,
                              float sr_b,
                              size_t num_threads = 0,
                              int num_runs = 50,
                              uint64_t base_seed = 21978990,
                              uint64_t orbit_seed = 72983498);

/// A/B per-channel drive gains at fixed dim/M: arm A = @p gains_a then arm B = @p gains_b.
/// Each list must have exactly @c NumDriveChannels(DRIVE_LAYOUT) entries (layout
/// feature order; e.g. XyzXz = 4: [x,y,z,xz]). Values must be finite and >= 0.
/// Matched seeds/SR; trains separately per arm (SeedSurvey).
/// Restores DIM, M, and INPUT_SCALE_CH on exit. Writes GainAB_*.csv/txt under RESULTS_DIR.
int Campaign_DriveGainAB(size_t dim,
                         size_t history_depth,
                         std::initializer_list<float> gains_a,
                         std::initializer_list<float> gains_b,
                         size_t num_threads = 0,
                         int num_runs = 50,
                         uint64_t base_seed = 21978990,
                         uint64_t orbit_seed = 72983498);
