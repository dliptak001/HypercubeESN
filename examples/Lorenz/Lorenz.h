#pragma once

#include "EnsembleESN.h"
#include "LorenzAttractor.h"
#include "JanusCursor.h"
#include <cstddef>
#include <vector>
#include <iostream>

#include "LorenzDatastream.h"

// ============================================================================
//  CONFIGURATION — consolidation point for primary variables of interest
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
    constexpr size_t RESERVOIR_WARMUP_STEPS = 100;

    // ---- (B) Readout (HCNN), trained ONLINE (single-sample, multi-epoch) ----
    constexpr float LR = 0.0015f; // per-step online learning rate (Adam)
    constexpr float WEIGHT_DECAY = 0.0f; // L2 on readout weights
    constexpr size_t EPOCHS = 600;

    // ---- (C) Data stream (Lorenz-63 integration + reflecting scan) ----
    constexpr size_t STREAM_LENGTH = 10000;
    constexpr int32_t CURSOR_SPAN = 5000;
    constexpr int32_t CURSOR_FOCUS_INDEX = STREAM_LENGTH - CURSOR_SPAN - 1000;
    constexpr LorenzAttractor::State INITIAL_LORENZ_STATE = {0.5, 0.5, 0.5};
    constexpr double DT = 0.02; // RK4 integration step (canonical Lorenz-63)

    constexpr double KAPPA = 0.2;
}

class Lorenz
{
public:
    Lorenz();

    void Train();

private:
    EnsembleConfig esn_config_; // we may not need to keep this around...
    EnsembleESN esn;
    LorenzDatastream data_stream_;

    static void ExtractInputsFromPastFutureStates(float inputs[8], const LorenzDatastreamResult& past_future_states);
    static EnsembleConfig MakeEnsembleConfig();
    static LorenzDatastreamConfig MakeDatastreamConfig();
};
