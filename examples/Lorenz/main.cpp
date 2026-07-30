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


    // Spectral-radius A/B at fixed dim/M (SeedSurvey per arm; roll-up under campaigns/).
    // Default house SR is 0.99; pair against a slightly contractive arm.
    return Campaign_SpectralRadiusAB(/*dim=*/12, /*history_depth=*/12,
                                     /*sr_a=*/0.98f, /*sr_b=*/0.99f,
                                     /*num_threads=*/0, /*num_runs=*/50);

    // Pipeline: Train → FreeRunSurvey → FreeRun, or multi-seed:
    // SeedSweep (Train+survey each seed, rank by mean VPT*duty).
    // constexpr size_t kDim = 12;
    // constexpr size_t kM = 12;
    // return SeedSweep(/*dim=*/kDim, /*history_depth=*/kM,
    //                  /*esn_seeds=*/{121978990ull, 221978990ull, 321978990ull,
    //                                 421978990ull, 521978990ull},
    //                  /*epochs=*/400,
    //                  /*freerun_runs=*/1000,
    //                  /*train_orbit=*/933312947715283458ull,
    //                  /*freerun_orbit_seed=*/729893498ull,
    //                  /*top_k=*/10,
    //                  /*do_train=*/true);

    // Single-seed survey only (weights already on disk):
    // return FreeRunSurvey(12, 12, 221978990ull, 1000, 72983498ull,
    //     R"(C:\HypercubeESN\models\lorenz_seed221978990_D12_M12)", 10);

    // Plot one IC after ranking:
    // return FreeRun(12, 12, seed, ic_x, ic_y, ic_z, stem);
}
