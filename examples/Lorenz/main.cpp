#include "Campaigns.h"

// ---------------------------------------------------------------------------
// Paste the campaign call(s) you want here. Rebuild. Run Lorenz.exe (no args).
// Model knobs: examples/Lorenz/Lorenz.h  config::
// Campaign helpers: examples/Lorenz/Campaigns.h
// First campaign arg is always reservoir DIM (N = 2^DIM).
// Drive gains: config::INPUT_SCALE_CH (constexpr; edit Lorenz.h).
// ---------------------------------------------------------------------------
int main()
{
    // Multi-seed survey (0 threads => hardware_concurrency)
    // return Campaign_SeedSurvey(/*dim=*/11, /*num_threads=*/0, /*num_runs=*/50);

    // Train seed 21978990, free-run fixed orbit 9333312947715283458:
    // every step printed + CSV under C:/HypercubeESN/results/traces/
    // Plot (absolute path is also printed at end of run):
    //   python examples/Lorenz/plot_freerun_overlay.py "C:/HypercubeESN/results/traces/seed21978990_orbit9333312947715283458.csv"
    // return Campaign_Trace(/*dim=*/12,
    //                       /*esn_seed=*/221978990,
    //                       /*max_freeruns=*/1,
    //                       /*target_orbit=*/9333312947715283458ull);


    // Pipeline: ParallelSeedSweep (overnight seed search) → Train/FreeRunSurvey
    // → FreeRun (plot one IC). Or ParallelOrbitSweep for orbit ranking on one seed.
    constexpr size_t kDim = 10;
    constexpr size_t kM = 2;

    // Parallel overnight seed search (train per seed; heartbeats on stderr):
    return ParallelSeedSweep(/*dim=*/kDim, /*history_depth=*/kM,
                             /*base_esn_seed=*/1002999015000000000ull,
                             /*num_seeds=*/16*50,
                             /*num_threads=*/16,
                             /*epochs=*/300,
                             /*freerun_runs=*/1000,
                             /*base_orbit_seed=*/9333312947715283458ull,
                             /*top_k=*/10,
                             /*spectral_radius=*/0.999f,
                             /*input_scaling=*/0.015f);

    // One ESN seed: train once, rank Mix64 orbits (one freerun each, parallel):
    // return ParallelOrbitSweep(/*dim=*/kDim, /*history_depth=*/kM,
    //                           /*base_esn_seed=*/13265426551472630865ull, // top from seed sweep
    //                           /*base_orbit_seed=*/9333312947715283458ull,
    //                           /*num_orbits=*/1000,
    //                           /*num_threads=*/16,
    //                           /*epochs=*/300,
    //                           /*top_k=*/10,
    //                           /*spectral_radius=*/0.999f,
    //                           /*input_scaling=*/0.015f);

    // Single-seed survey only (weights already on disk):
    // return FreeRunSurvey(/*dim=*/12, /*history_depth=*/32,
    //                      /*esn_seed=*/10121978990ull,
    //                      /*num_runs=*/1000,
    //                      /*orbit_seed=*/72983498ull,
    //                      /*weights_stem=*/R"(C:\HypercubeESN\models\lorenz_seed10121978990_D12_M32)",
    //                      /*top_k=*/10,
    //                      /*out=*/nullptr,
    //                      /*spectral_radius=*/0.99f,
    //                      /*input_scaling=*/0.015f);

    // Plot one IC after ranking:
    // return FreeRun(/*dim=*/12, /*history_depth=*/32, /*esn_seed=*/10121978990ull,
    //                /*ic_x=*/..., /*ic_y=*/..., /*ic_z=*/...,
    //                /*weights_stem=*/R"(C:\HypercubeESN\models\lorenz_seed10121978990_D12_M32)",
    //                /*spectral_radius=*/0.99f,
    //                /*input_scaling=*/0.015f);
}
