#pragma once

#include "Lorenz.h" // FreeRunProtocol

#include <cstddef>
#include <cstdint>
#include <initializer_list>

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
    FreeRunProtocol protocol = FreeRunProtocol::Unseen;
    uint64_t base_seed = 0;
    uint64_t orbit_seed = 0;

    double mean_vpt = 0;      ///< mean of per-trial mean VPT (lt)
    double std_vpt = 0;       ///< sample std of per-trial mean VPT
    double mean_rmse = 0;
    double std_rmse = 0;
    double mean_duty = 0;
    double std_duty = 0;
    double mean_n_relock = 0;
    double std_n_relock = 0;

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

/// Single-seed train/load + free-run CSV dumps under examples/Lorenz/traces/.
/// @param dim  Reservoir hypercube dim (N = 2^dim). Restored on exit.
int Campaign_Trace(size_t dim,
                   uint64_t esn_seed,
                   int max_freeruns = 30,
                   uint64_t target_orbit = 0,
                   uint64_t orbit_seed = 72983498);

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
/// Matched dim/M/seeds/protocol; trains separately per arm. Code-computed roll-up.
/// @param dim            Reservoir hypercube dim (N = 2^dim). Restored on exit.
/// @param history_depth  Fixed M for both arms (1..64). Restored on exit.
int Campaign_DriveLayoutAB(size_t dim,
                           size_t history_depth,
                           size_t num_threads = 0,
                           int num_runs = 50,
                           uint64_t base_seed = 21978990,
                           uint64_t orbit_seed = 72983498);
