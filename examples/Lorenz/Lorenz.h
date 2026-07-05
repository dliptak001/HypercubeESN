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
    constexpr float INPUT_SCALING = 0.05; //0.10f; // shared across all input channels
    constexpr float LEAK_RATE = 1.0f;
    constexpr size_t HISTORY_DEPTH = 8; // delay-line depth

    // ---- Ensemble ESN ----
    constexpr double KAPPA = 0.2; // ramp ceiling (kappa_max); the per-epoch value comes from KappaProfile
    constexpr Combine COMBINE = Combine::Mean; // Consensus statistic

    // ---- Readout (HCNN), trained ONLINE (single-sample, multi-epoch) ----
    constexpr float LEARNING_RATE = 0.0001f; // peak per-step online learning rate (Adam); annealed by LrProfile
    constexpr float LEARNING_RATE_MIN = 0.00002f; // anneal floor reached at the final epoch
    constexpr size_t EPOCHS = 100;

    // ---- Data stream (Lorenz-63 integration + Janus cursor window) ----
    constexpr size_t STREAM_LENGTH = 20000;
    constexpr size_t FREE_RUN_WINDOW_SIZE = 2000;
    constexpr int32_t TRAINING_WINDOW_SIZE = STREAM_LENGTH - 2*FREE_RUN_WINDOW_SIZE;
    constexpr int32_t CURSOR_CENTER_INDEX = STREAM_LENGTH - TRAINING_WINDOW_SIZE / 2 - FREE_RUN_WINDOW_SIZE;
    constexpr LorenzAttractor::State INITIAL_LORENZ_STATE = {0.15, 0.75, 0.5};
    constexpr double DT = 0.02; // RK4 integration step (canonical Lorenz-63)

    // ---- Stage control ----
    constexpr size_t RESERVOIR_WARMUP_STEPS = 100;

    // ---- Free-run scoring ----
    constexpr float VPT_THRESHOLD = 0.2f; // channel-RMS error (normalized units) ending the valid-prediction time; provisional
    constexpr double LYAPUNOV_EXPONENT = 0.9056; // canonical Lorenz-63 lambda_max, for the step -> Lyapunov-time conversion

    // ---- Exposure-bias remedies (README Issue 2) ----
    // Both act ONLY during Train(), ONLY on the future channels (3-5) — the sole
    // train/free-run input mismatch; the teacher target stays the real S[f], and
    // channels 6-7 are past-derived so they remain consistent under both. The
    // FreeRun washout is left clean. Enable ONE at a time to preserve the
    // campaign's single-delta discipline; both default to OFF (baseline).
    // 2a — noise injection: zero-mean Gaussian std added to the future channels.
    //      0 disables. Recommended starting bracket: 1e-3 .. a few 1e-2.
    constexpr float TRAIN_FUTURE_NOISE = 0.005f;
    // 2b — scheduled sampling: probability ceiling of feeding the ensemble's own
    //      fresh prediction on the future channels instead of the real sample,
    //      linearly ramped 0 -> ceiling across epochs. 0 disables.
    //      Recommended starting ceiling: ~0.25 .. 0.5.
    constexpr float SCHEDULED_SAMPLING_CEILING = 0.4f;
    // RNG stream for the 2a noise draws and 2b Bernoulli decisions — kept distinct
    // from the reservoir SEED so toggling these never perturbs the reservoir.
    constexpr uint64_t TRAIN_EXPOSURE_RNG_SEED = 0x5EED5EEDULL;
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
/// FreeRun() then rides the same cursors past the window edge: each member's
/// future channels switch to that member's OWN prediction (per-member closed
/// loop, ExtractInputs_FreeRun per member) while the past cursor stays anchored
/// to real history; the consensus is reported and scored step-for-step against
/// the held-out orbit tail.
class Lorenz
{
public:
    /// Builds the ensemble and the datastream from the config:: constants.
    Lorenz();

    /// Runs config::EPOCHS teacher-forced training passes over the cursor
    /// window, printing one line per epoch: kappa and the prequential train
    /// RMSE (normalized units, over all 3 channels x all steps of the sweep).
    /// Optionally applies the Issue-2 exposure-bias remedies on the future
    /// channels (config::TRAIN_FUTURE_NOISE for 2a, config::SCHEDULED_SAMPLING_CEILING
    /// for 2b); both are off by default and the loop is teacher-forced when they are.
    void Train();

    /// Generative rollout over the held-out tail. Self-contained: resets the
    /// cursors, holds kappa at the config::KAPPA ceiling, re-sweeps the whole
    /// training window teacher-forced but inference-only (anchored washout),
    /// then goes generative for config::FREE_RUN_WINDOW_SIZE steps in the
    /// PER-MEMBER closed loop — each member is fed its own fresh prediction on
    /// the future input channels (so the consensus coupling acts on genuinely
    /// divergent trajectories) while the consensus is scored against the true
    /// orbit, printing an error trace, the VPT_THRESHOLD crossing, and the
    /// free-run RMSE.
    void FreeRun();

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

    /// Per-epoch learning-rate schedule: cosine-anneal (CosineLR) from lr_max at
    /// epoch 0 to lr_min at the final epoch — lowers the single-sample
    /// gradient-noise floor late in training.
    static float LrProfile(float lr_max, float lr_min, size_t epochs, size_t current_epoch);

    /// Per-epoch scheduled-sampling probability (README Issue 2b): linear ramp
    /// from 0 at epoch 0 to @p ceiling at the final epoch. Returns @p ceiling
    /// when epochs <= 1. ceiling == 0 keeps the ramp flat at 0 (2b disabled).
    static float ScheduledSamplingProfile(float ceiling, size_t epochs, size_t current_epoch);
};
