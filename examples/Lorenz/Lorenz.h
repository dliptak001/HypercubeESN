#pragma once

#include "EnsembleESN.h"
#include <cstddef>
#include <vector>
#include <iostream>

// ============================================================================
//  CONFIGURATION — edit these, then rebuild. Everything else is mechanism.
// ============================================================================
namespace config
{
    // ---- (A) Reservoir / model ----
    constexpr size_t DIM = 8; // hypercube dimension
    constexpr uint64_t SEED = 673895; // reservoir seed
    constexpr float SPECTRAL_RADIUS = 0.90f; // A(x): ~0.90,  tanh(x): ~0.95 (tune per arm)
    constexpr float INPUT_SCALING = 0.10f; // shared across all input channels
    constexpr float LEAK_RATE = 1.0f; // 1.0 = continuous flow;
    constexpr size_t HISTORY_DEPTH = 16; // delay-line depth
    constexpr float LORENTZ_GAMMA = 1.1f; // set 0.0f for the tanh baseline arm
    constexpr float LORENTZ_INV_SIGMA2 = 250.0f; // 1/sigma^2 of the envelope

    // ---- (B) Readout (HCNN), trained ONLINE (single-sample, multi-epoch) ----
    constexpr size_t WASHOUT = 100;
    constexpr float LR = 0.0015f; // per-step online learning rate (Adam)
    constexpr float WEIGHT_DECAY = 0.0f; // L2 on readout weights
    constexpr size_t EPOCHS = 600;
}

class Lorenz
{
public:
    Lorenz();

private:
    static EnsembleConfig MakeConfig();

    EnsembleConfig esn_config_;
    EnsembleESN esn;
};
