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
    // return Campaign_DriveLayoutAB(/*dim=*/11, /*M=*/24, /*threads=*/16, /*runs=*/50);

    // Sweep reservoir history depth M (list is an argument, not CLI):
    // return Campaign_HistoryDepthSweep(/*dim=*/11, {2, 4, 6}, /*threads=*/16, /*runs=*/50);

    // Longer overnight-style M grid:
    //return Campaign_HistoryDepthSweep(/*dim=*/12, {8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 22, 24}, /*threads=*/16, /*runs=*/50);

    // Train seed 21978990, free-run fixed orbit 9333312947715283458:
    // every step printed + CSV under C:/HypercubeESN/results/traces/
    // Plot (absolute path is also printed at end of run):
    //   python examples/Lorenz/plot_freerun_overlay.py "C:/HypercubeESN/results/traces/seed21978990_orbit9333312947715283458.csv"
    // return Campaign_Trace(/*dim=*/12,
    //                       /*esn_seed=*/221978990,
    //                       /*max_freeruns=*/1,
    //                       /*target_orbit=*/9333312947715283458ull);


    // Pipeline: Train → FreeRunSurvey (rank ICs) → FreeRun (plot one IC).
    constexpr size_t kDim = 12;
    constexpr size_t kM = 10;
    constexpr uint64_t kSeed = 221978990ull;
    constexpr const char* kStem =
        R"(C:\HypercubeESN\models\lorenz_seed221978990_D12_M10)";

    return Train(kDim, kM, kSeed, /*target_orbit=*/9333312947715283458ull, /*epochs=*/400, kStem);

    // Load weights; score many Unseen freeruns; print top ICs + leaderboard CSV.
    // return FreeRunSurvey(kDim, kM, kSeed, /*num_runs=*/1000,
    //                      /*orbit_seed=*/72983498ull, kStem, /*top_k=*/10);

    // After survey, paste a top IC for err/x/y/z overlay plot:
    // return FreeRun(kDim, kM, kSeed, /*ic_x=*/0.43, /*ic_y=*/0.30, /*ic_z=*/0.64, kStem);
}
