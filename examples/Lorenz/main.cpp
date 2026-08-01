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
    // Pipeline: SeedSweep → Train (save) → OrbitSweep (load) → FreeRun (plot).
    constexpr size_t kDim = 10;
    constexpr size_t kM = 2;
    constexpr uint64_t kEsn = 4112530987988204306ull; // reservoir / weight seed
    // Orbit seed: RNG key → Lorenz IC via IcFromOrbitSeed (not FreeRun's IC).
    // Train: start of the per-epoch orbit chain. OrbitSweep: base of Mix64 list.
    constexpr uint64_t kBaseOrbitSeed = 9333312947715283458ull;
    // Trained readout path, no extension — Train writes, OrbitSweep/FreeRun load:
    //   <path>.hcnw  +  <path>.arch.json
    constexpr const char* kWeightsPath =
        R"(C:\HypercubeESN\models\lorenz_seed4112530987988204306_D10_M2)";

    // Parallel overnight seed search (train per seed; heartbeats on stderr):
    // return SeedSweep(/*dim=*/kDim,
    //                  /*history_depth=*/kM,
    //                  /*base_esn_seed=*/1002999015000000000ull,
    //                  /*num_seeds=*/16 * 50,
    //                  /*num_threads=*/16,
    //                  /*epochs=*/300,
    //                  /*freerun_runs=*/1000,
    //                  /*base_orbit_seed=*/kBaseOrbitSeed,
    //                  /*top_k=*/10,
    //                  /*spectral_radius=*/0.999f,
    //                  /*input_scaling=*/0.015f);

    // Fit readout once (comment out once kWeightsPath.hcnw exists).
    if (const int tr = Train(/*dim=*/kDim,
                             /*history_depth=*/kM,
                             /*esn_seed=*/kEsn,
                             /*target_orbit=*/kBaseOrbitSeed,
                             /*epochs=*/300,
                             /*weights_stem=*/kWeightsPath))
        return tr;

    // Load weights; rank Mix64 orbits (one freerun each, parallel):
    // return OrbitSweep(/*dim=*/kDim,
    //                   /*history_depth=*/kM,
    //                   /*base_esn_seed=*/kEsn,
    //                   /*base_orbit_seed=*/kBaseOrbitSeed,
    //                   /*num_orbits=*/100000,
    //                   /*num_threads=*/16,
    //                   /*weights_stem=*/kWeightsPath,
    //                   /*top_k=*/10,
    //                   /*spectral_radius=*/0.999f,
    //                   /*input_scaling=*/0.015f);

    // Plot one IC (load-only). Example: OrbitSweep #1 by VxD.
    return FreeRun(/*dim=*/kDim,
                   /*history_depth=*/kM,
                   /*esn_seed=*/kEsn,
                   /*ic_x=*/0.624263, /*ic_y=*/-0.868602, /*ic_z=*/0.587179,
                   /*weights_stem=*/kWeightsPath,
                   /*spectral_radius=*/0.999f,
                   /*input_scaling=*/0.015f);
}
