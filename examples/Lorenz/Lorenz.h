#pragma once

#include "ESN.h"
#include "LorenzAttractor.h"
#include "LorenzDatastream.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

// ============================================================================
//  CONFIGURATION — consolidation point for primary variables of interest
// ============================================================================
namespace config
{
    // ---- Diagnostics ----
    // Master gate for all live per-run diagnostic printf's (config banner, per-epoch
    // train lines, free-run header / step trace / runway notes). Leave true for a
    // normal single-seed run to watch progress; set false for a concurrent seed
    // survey, whose per-seed result table (VPT + RMSE) prints regardless — main
    // collects it from FreeRun()'s return value, not from these gated prints.
    inline bool ENABLE_PRINTF = true;

    // **** seed 13649419    VPT 347 steps ( 6.28 lt)  free-run RMSE 0.428023  (2000 steps)
    // ---- Reservoir / model ----
    constexpr size_t DIM = 11; // hypercube dimension
    constexpr uint64_t SEED = 13649419;//13649188; // reservoir seed
    constexpr float SPECTRAL_RADIUS = 0.99f; // A(x): ~0.90,  tanh(x): ~0.95 (tune per arm)
    constexpr float INPUT_SCALING = 0.005; // shared across all input channels
    constexpr float FEEDBACK_SCALING = 0.04f; // future-block gain on the dedicated feedback port
    constexpr float LEAK_RATE = 1.0;
    constexpr size_t HISTORY_DEPTH = 24; // delay-line depth

    // ---- Readout (HCNN), trained ONLINE (single-sample, multi-epoch) ----
    constexpr float LEARNING_RATE = 0.00004f; // peak per-step online learning rate (Adam); annealed by LrProfile
    constexpr float LEARNING_RATE_MIN = 0.000002f;//0.000005f; // anneal floor reached at the final epoch
    constexpr size_t EPOCHS = 100;

    // ---- Readout input: block-structured (delay-line slices + optional aux block) ----
    // The readout consumes B = READOUT_SLICES + (AUX_INPUT_DIM > 0) blocks of N, laid
    // out on a (DIM + log2 B) hypercube: one block per reservoir delay-line slice
    // (newest first), plus an optional block holding the auxiliary input. B must be a
    // power of two, and READOUT_SLICES must not exceed HISTORY_DEPTH. The reservoir is
    // untouched by all of this — the slices are the memory it already computes for its
    // recurrent gather, and which the readout has never been shown.
    constexpr size_t READOUT_SLICES = 1;//HISTORY_DEPTH;
    constexpr size_t AUX_INPUT_DIM = 0; // 3 = normalized past (x,y,z); 0 = no aux block
    constexpr bool USE_POOLING = true;

    // ---- Data stream (Lorenz-63 integration + Janus cursor window) ----
    constexpr int32_t TRAINING_WINDOW_SIZE = 20000;
    constexpr size_t FREE_RUN_WINDOW_SIZE = 2000;
    constexpr int32_t CURSOR_CENTER_INDEX = FREE_RUN_WINDOW_SIZE + TRAINING_WINDOW_SIZE / 2;
    constexpr size_t STREAM_LENGTH = 2*FREE_RUN_WINDOW_SIZE + TRAINING_WINDOW_SIZE;

    constexpr LorenzAttractor::State INITIAL_LORENZ_STATE = {-0.836584,-0.109998,0.615358};//{0.91, 0.275, 0.19};
    constexpr double DT = 0.02; // RK4 integration step (canonical Lorenz-63)

    // ---- Stage control ----
    constexpr size_t RESERVOIR_WARMUP_STEPS = 1000;

    // ---- Free-run scoring ----
    constexpr float VPT_THRESHOLD = 0.3f; // channel-RMS error (normalized units) ending the valid-prediction time; provisional
    constexpr double LYAPUNOV_EXPONENT = 0.9056; // canonical Lorenz-63 lambda_max, for the step -> Lyapunov-time conversion

    // ---- Exposure-bias remedies (README Issue 2) ----
    // Both act ONLY during Train(), ONLY on the future block (feedback port) — the
    // sole train/free-run drive mismatch; the teacher target stays the real S[f].
    // They perturb the future LINEAR channels; the future x*z is then re-derived so
    // the block stays consistent with free-run (where it is the predicted x*z). The
    // FreeRun washout is left clean. Enable ONE at a time to preserve the
    // campaign's single-delta discipline; both default to OFF (baseline).
    // 2a — noise injection: zero-mean Gaussian std added to the future channels.
    //      0 disables. Recommended starting bracket: 1e-3 .. a few 1e-2.
    constexpr float TRAIN_FUTURE_NOISE = 0.0f;

    // 2b — scheduled sampling: probability ceiling of feeding the model's own
    //      fresh prediction on the future channels instead of the real sample,
    //      linearly ramped 0 -> ceiling across epochs. 0 disables.
    //      Recommended starting ceiling: ~0.25 .. 0.5.
    constexpr float SCHEDULED_SAMPLING_CEILING = 0.0f;

    // RNG stream for the 2a noise draws and 2b Bernoulli decisions — kept distinct
    // from the reservoir SEED so toggling these never perturbs the reservoir.
    constexpr uint64_t TRAIN_EXPOSURE_RNG_SEED = 0x5EED5EEDULL;
}

/// One seed's free-run outcome: the numeric metrics the survey aggregates, plus a
/// pre-formatted display row. @ref Lorenz::FreeRun fills it; main() prints the rows
/// and computes min/max/mean/median/std over the fields.
struct FreeRunResult
{
    bool valid = false; ///< false if the rollout scored 0 steps (excluded from stats)
    uint64_t seed = 0; ///< reservoir seed of this run
    size_t vpt_steps = 0; ///< step of first VPT_THRESHOLD crossing; 0 = never crossed
    bool crossed = false; ///< whether the error ever crossed VPT_THRESHOLD (vpt_steps > 0)
    double vpt_lt = 0.0; ///< valid-prediction time in Lyapunov times (window floor if never crossed)
    double rmse = 0.0; ///< free-run RMSE over the scored steps (normalized units)
    size_t steps = 0; ///< number of generative steps actually scored
    std::string row; ///< human-readable table line for this seed
};

/// @brief Experiment driver: online Janus-cursor training of an ESN on the
/// Lorenz-63 attractor.
///
/// Owns the full pipeline and wires it from the config:: constants above:
///
///   LorenzDatastream (normalized float stream + dual Janus cursors)
///       |  input port:    past block   [x, y, z, x*z]  (real history)
///       |  feedback port: future block [x, y, z, x*z]  (real in train, prediction in free-run)
///       v
///   ESN (fixed reservoir + online-trained CNN readout)
///
/// Each Train() epoch resets the cursors, warms the reservoir up teacher-forced,
/// then sweeps the cursor window once teacher-forced at horizon 1, reporting the
/// prequential RMSE. FreeRun() then rides the same cursors past the window edge:
/// the future block on the feedback port switches to the model's OWN prediction
/// (single-ESN self-feedback closed loop, ExtractFuturePredicted) while the past
/// cursor stays anchored to real history; the prediction is scored step-for-step
/// against the held-out orbit tail.
class Lorenz
{
public:
    /// Builds the ESN from the config:: constants. The datastream is allocated later
    /// by Train() / FreeRun() via RebuildDatastream (fresh orbit each call).
    Lorenz(uint64_t seed, uint64_t orbit_seed);

    /// Runs config::EPOCHS teacher-forced training passes over the cursor
    /// window, printing one line per epoch: the learning rate and the prequential
    /// train RMSE (normalized units, over all 3 channels x all steps of the sweep).
    /// Optionally applies the Issue-2 exposure-bias remedies on the future
    /// channels (config::TRAIN_FUTURE_NOISE for 2a, config::SCHEDULED_SAMPLING_CEILING
    /// for 2b); both are off by default and the loop is teacher-forced when they are.
    void Train();

    /// Generative rollout over the held-out tail. Self-contained: resets the
    /// cursors, re-sweeps the whole training window teacher-forced but
    /// inference-only (anchored washout), then goes generative for
    /// config::FREE_RUN_WINDOW_SIZE steps in the self-feedback closed loop — the
    /// model's own prediction is fed on the future input channels while the past
    /// cursor stays anchored to real history, and the prediction is scored against
    /// the true orbit. The live error trace / VPT crossing / RMSE lines are gated
    /// on config::ENABLE_PRINTF; the numeric outcome and its formatted table row
    /// are always returned in a FreeRunResult (valid == false if 0 steps scored).
    FreeRunResult FreeRun(bool verbose);

    /// Human-readable HCNN stack + param counts (shared across survey seeds).
    [[nodiscard]] std::string ReadoutArchSummary() const {
        return esn_.ReadoutArchSummary();
    }

private:
    uint64_t seed_, orbit_seed_;

    ESNConfig esn_config_;
    ESN esn_;
    std::unique_ptr<LorenzDatastream> data_stream_; // rebuilt each epoch / free-run; owns the current stream

    void RebuildDatastream(bool verbose);

    /// Packs the 4-wide past block [x, y, z, x*z] (input port) from real history.
    static void ExtractPast(float past[4], const LorenzDatastreamResult& past_future_states);

    /// Packs the 4-wide future block [x, y, z, x*z] (feedback port) from the real
    /// future sample — teacher-forced drive (warmups + training).
    static void ExtractFutureReal(float future[4], const LorenzDatastreamResult& past_future_states);

    /// Packs the 4-wide future block [x, y, z, x*z] (feedback port) from the model's
    /// own prediction — the generative self-feedback drive.
    static void ExtractFuturePredicted(float future[4], const float* prediction);

    /// Copies the current future sample — the horizon-1 target — into targets.
    static void ExtractTargets(float targets[3], const NormalizedState& future_state);

    /// Packs the auxiliary readout input u_raw = normalized past (x,y,z) — fed onto
    /// the readout's aux block (see config::AUX_INPUT_DIM). Unused when the readout
    /// has no aux block.
    static void ExtractAuxPast(float u_raw[3], const LorenzDatastreamResult& past_future_states);

    /// Assembles the ESN config from the config:: constants.
    static ESNConfig MakeESNConfig(uint64_t seed);

    /// Assembles the LorenzDatastream config from the config:: constants and a
    /// by-value orbit seed state (owned by the caller for this call only).
    static LorenzDatastreamConfig MakeDatastreamConfig(LorenzAttractor::State orbit);

    /// Per-epoch learning-rate schedule: cosine-anneal (CosineLR) from lr_max at
    /// epoch 0 down to lr_min at 75% of the run, then held flat at lr_min for the
    /// final 25% — lowers the single-sample gradient-noise floor late in training
    /// and lets the readout settle at the floor rather than still cooling at the end.
    static float LrProfile(float lr_max, float lr_min, size_t epochs, size_t current_epoch);

    /// Per-epoch scheduled-sampling probability (README Issue 2b): linear ramp
    /// from 0 at epoch 0 to @p ceiling at the final epoch. Returns @p ceiling
    /// when epochs <= 1. ceiling == 0 keeps the ramp flat at 0 (2b disabled).
    static float ScheduledSamplingProfile(float ceiling, size_t epochs, size_t current_epoch);
};
