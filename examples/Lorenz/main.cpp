#include "Campaigns.h"

// ---------------------------------------------------------------------------
// Paste the campaign call(s) you want here. Rebuild. Run Lorenz.exe (no args).
// Model knobs: examples/Lorenz/Lorenz.h  config::
// Campaign helpers: examples/Lorenz/Campaigns.h
// First campaign arg is always reservoir DIM (N = 2^DIM).
// ---------------------------------------------------------------------------
int main()
{
    // Multi-seed survey (0 threads => hardware_concurrency)
    // return Campaign_SeedSurvey(/*dim=*/11, /*num_threads=*/0, /*num_runs=*/50);

    // A/B drive layouts at fixed M: XyzXz (4-in) vs Quadratic8 (8-in)
    return Campaign_DriveLayoutAB(/*dim=*/11, /*M=*/24, /*threads=*/16, /*runs=*/50);

    // Sweep reservoir history depth M (list is an argument, not CLI):
    // return Campaign_HistoryDepthSweep(/*dim=*/11, {2, 4, 6}, /*threads=*/16, /*runs=*/50);

    // Longer overnight-style M grid:
    // return Campaign_HistoryDepthSweep(
    //     /*dim=*/12, {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24},
    //     /*threads=*/16, /*runs=*/50);

    // --- other campaigns ---
    // return Campaign_Trace(/*dim=*/11, /*esn_seed=*/21978990, /*max_freeruns=*/30);
}
