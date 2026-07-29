#include "Campaigns.h"

// ---------------------------------------------------------------------------
// Paste the campaign call(s) you want here. Rebuild. Run Lorenz.exe (no args).
// Model knobs: examples/Lorenz/Lorenz.h  config::
// Campaign helpers: examples/Lorenz/Campaigns.h
// ---------------------------------------------------------------------------
int main()
{
    // Multi-seed survey (0 threads => hardware_concurrency)
    // return Campaign_SeedSurvey(/*num_threads=*/0, /*num_runs=*/50);

    // Sweep reservoir history depth M (list is an argument, not CLI):
    return Campaign_HistoryDepthSweep({4, 8, 16, 24, 32}, /*threads=*/16, /*runs=*/50);

    // --- other campaigns ---
    // return Campaign_Trace(/*esn_seed=*/21978990, /*max_freeruns=*/30);
}
