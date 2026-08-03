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
    // Manual Pipeline: SeedSweep() → Train (save weights) → OrbitSweep (load weights) → FreeRun (load weights, plot).
    constexpr size_t kDim = 10;
    constexpr size_t kM = 2;
    constexpr float kSpectralRadius = 0.999f;
    constexpr float kInputScaling = 0.015f;
    constexpr uint64_t kBaseOrbitSeed = 9333312947715283458ull;
    constexpr int numThreads = 16;

    /////////////////////////////////////////////////////////////////////////
    // Stage 1
    /////////////////////////////////////////////////////////////////////////
    // return SeedSweep(/*dim=*/kDim,
    //                  /*history_depth=*/kM,
    //                  /*base_esn_seed=*/1002999015000000000ull,
    //                  /*num_seeds=*/numThreads * 50,
    //                  /*num_threads=*/numThreads,
    //                  /*epochs=*/100,
    //                  /*freerun_runs=*/1000,
    //                  /*base_orbit_seed=*/kBaseOrbitSeed,
    //                  /*top_k=*/10,
    //                  /*spectral_radius=*/kSpectralRadius,
    //                  /*input_scaling=*/kInputScaling);

    /////////////////////////////////////////////////////////////////////////
    // Stage 2
    /////////////////////////////////////////////////////////////////////////
    constexpr uint64_t kEsn = 7934791766227647176ull; // one of the top ranked seeds from SeedSweep
    const std::string weights = DefaultWeightStem(kEsn, kDim, kM);
    const char* const kWeightsPath = weights.c_str();
    if (const int tr = Train(/*dim=*/kDim,
                                     /*history_depth=*/kM,
                                     /*esn_seed=*/kEsn,
                                     /*target_orbit=*/kBaseOrbitSeed,
                                     /*epochs=*/100,
                                     /*weights_stem=*/kWeightsPath))
        return tr;

    /////////////////////////////////////////////////////////////////////////
    // Stage 3
    /////////////////////////////////////////////////////////////////////////
    return OrbitSweep(/*dim=*/kDim,
                              /*history_depth=*/kM,
                              /*base_esn_seed=*/kEsn,
                              /*spectral_radius=*/kSpectralRadius,
                              /*input_scaling=*/kInputScaling,
                              /*base_orbit_seed=*/kBaseOrbitSeed,
                              /*num_orbits=*/200000,
                              /*num_threads=*/numThreads,
                              /*weights_stem=*/kWeightsPath,
                              /*top_k=*/10);


    /////////////////////////////////////////////////////////////////////////
    // Stage 4
    /////////////////////////////////////////////////////////////////////////
    // Free-run one OrbitSweep winner by orbit_seed (full double IC; not float paste).
    // freerun_steps: 0 = config::FREE_RUN_WINDOW_SIZE (2000); larger for long plots.
    // Example: top by VxD from par_orbit_sweep ...170727 (idx 188138).
    return FreeRun(/*dim=*/kDim,
                   /*history_depth=*/kM,
                   /*esn_seed=*/kEsn,
                   /*spectral_radius=*/kSpectralRadius,
                   /*input_scaling=*/kInputScaling,
                   /*orbit_seed=*/11526500681396796181ull,
                   /*weights_stem=*/kWeightsPath,
                   /*freerun_steps=*/2000);
}
