#pragma once

#include "EnsembleESN.h"
#include "LorenzAttractor.h"
#include "LorenzDatastream.h"
#include <cstddef>
#include <cstdint>

// ============================================================================
//  CONFIGURATION — consolidation point for primary variables of interest
// ============================================================================
namespace config
{
    // ---- Reservoir / model ----
    constexpr size_t DIM = 8; // hypercube dimension
    constexpr uint64_t SEED = 7673895; // reservoir seed
    constexpr float SPECTRAL_RADIUS = 0.9f; // A(x): ~0.90,  tanh(x): ~0.95 (tune per arm)
    constexpr float INPUT_SCALING = 0.05;//0.10f; // shared across all input channels
    constexpr float LEAK_RATE = 1.0f;
    constexpr size_t HISTORY_DEPTH = 8; // delay-line depth
    constexpr float LORENTZ_GAMMA = 0.0f; // set 0.0f for the tanh baseline arm

    // ---- Ensemble ESN ----
    constexpr double KAPPA = 0.01; // ramp ceiling (kappa_max); the per-epoch value comes from KappaProfile
    constexpr Combine COMBINE = Combine::Mean;  // Consensus statistic

    // ---- Readout (HCNN), trained ONLINE (single-sample, multi-epoch) ----
    constexpr float LEARNING_RATE = 0.0001f; // peak per-step online learning rate (Adam); annealed by LrProfile
    constexpr float LEARNING_RATE_MIN = 0.00002f; // anneal floor reached at the final epoch
    constexpr size_t LR_HOLD_EPOCHS = 1; // hold at peak through the fast-improvement phase, then cosine-decay
    constexpr size_t EPOCHS = 200;

    // ---- Data stream (Lorenz-63 integration + Janus cursor window) ----
    constexpr size_t STREAM_LENGTH = 20000;
    constexpr int32_t CURSOR_SPAN = 15000;
    constexpr int32_t CURSOR_CENTER_INDEX = STREAM_LENGTH - CURSOR_SPAN/2 - 1000;
    constexpr LorenzAttractor::State INITIAL_LORENZ_STATE = {0.5, 0.5, 0.5};
    constexpr double DT = 0.02; // RK4 integration step (canonical Lorenz-63)

    // ---- Stage control ----
    constexpr size_t RESERVOIR_WARMUP_STEPS = 100;
}

/// @brief Experiment driver: online Janus-cursor training of an EnsembleESN on
/// the Lorenz-63 attractor.
///
/// Owns the full pipeline and wires it from the config:: constants above:
///
///   LorenzDatastream (normalized float stream + dual Janus cursors)
///       |  8 channels: [past xyz, future xyz, distance, past x*z]
///       v
///   EnsembleESN (M members, consensus feedback, online readout updates)
///
/// Each Train() epoch resets the cursors, sets the epoch's coupling kappa from
/// the saturating ramp KappaProfile, warms the reservoirs up (coupled — the
/// epoch's kappa is live during warmup by design), then sweeps the cursor
/// window once teacher-forced at horizon 1, reporting the prequential RMSE.
/// The generative free-run stage (the consumer of ExtractInputs_FreeRun) is
/// not built yet.
class Lorenz
{
public:
    /// Builds the ensemble and the datastream from the config:: constants.
    Lorenz();

    /// Runs config::EPOCHS teacher-forced training passes over the cursor
    /// window, printing one line per epoch: kappa and the prequential train
    /// RMSE (normalized units, over all 3 channels x all steps of the sweep).
    void Train();

private:
    EnsembleConfig esn_config_;
    EnsembleESN esn_;
    LorenzDatastream data_stream_;

    /// Packs the 8 input channels from real (teacher-forced) data.
    static void ExtractInputs_Training(float inputs[8], const LorenzDatastreamResult& past_future_states);
    /// Same channel layout, but the future half (channels 3-5) comes from the
    /// ensemble's last consensus output instead of the datastream.
    static void ExtractInputs_FreeRun(float inputs[8], const LorenzDatastreamResult& past_future_states,
                                      const float* consensus);
    /// Copies the current future sample — the horizon-1 target — into targets.
    static void ExtractTargets(float targets[3], const NormalizedState& future_state);
    /// Assembles the EnsembleESN config from the config:: constants.
    static EnsembleConfig MakeEnsembleConfig();
    /// Assembles the LorenzDatastream config from the config:: constants.
    static LorenzDatastreamConfig MakeDatastreamConfig();
    /// Saturating coupling ramp kappa_max*k*x^2/(1 + k*x^2), x = current_epoch/epochs:
    /// rises steeply, asymptotes strictly below kappa_max.
    static double KappaProfile(double kappa_max, double k, size_t epochs, size_t current_epoch);
    /// Per-epoch learning-rate schedule: hold lr_max through hold_epochs (the
    /// fast-improvement phase), then cosine-anneal (CosineLR) to lr_min at the
    /// final epoch — lowers the single-sample gradient-noise floor late in training.
    static float LrProfile(float lr_max, float lr_min, size_t hold_epochs, size_t epochs, size_t current_epoch);
};
