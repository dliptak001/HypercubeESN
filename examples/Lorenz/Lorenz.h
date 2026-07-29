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
    // Master gate for live per-run diagnostic printf's (config banner, per-epoch
    // train lines, free-run traces). Leave true for a single-seed interactive run;
    // set false for concurrent seed surveys (main collects FreeRunResult rows).
    inline bool ENABLE_PRINTF = true;

    // ---- Reservoir / model ----
    constexpr size_t DIM = 11;
    constexpr uint64_t SEED = 13649419;
    constexpr float SPECTRAL_RADIUS = 0.99f;
    // Sole drive gain: teacher / self-prediction on the input bank (4 channels).
    // (Former Janus used a weak past on input and ~0.04 on the future ext-fb path;
    // the forward signal was the 0.04 arm — start there for free-run.)
    constexpr float INPUT_SCALING = 0.04f;
    constexpr float LEAK_RATE = 1.0f;
    constexpr size_t HISTORY_DEPTH = 24;

    // ---- Readout (HCNN), trained ONLINE (single-sample, multi-epoch) ----
    constexpr float LEARNING_RATE = 0.00004f;
    constexpr float LEARNING_RATE_MIN = 0.000002f;
    constexpr size_t EPOCHS = 100;
    constexpr size_t READOUT_SLICES = 2;
    constexpr size_t CONV_CHANNELS = 1;
    constexpr int NUM_LAYERS = 1;
    constexpr bool USE_POOLING = true;

    // ---- Data stream (Lorenz-63 + forward cursor window) ----
    // Layout: train/washout [0, TRAINING_WINDOW_SIZE] inclusive; free-run after span.
    constexpr int32_t TRAINING_WINDOW_SIZE = 20000;
    constexpr size_t FREE_RUN_WINDOW_SIZE = 1000;
    constexpr size_t STREAM_LENGTH =
        static_cast<size_t>(TRAINING_WINDOW_SIZE) + FREE_RUN_WINDOW_SIZE;

    constexpr LorenzAttractor::State INITIAL_LORENZ_STATE = {-0.836584, -0.109998, 0.615358};
    constexpr double DT = 0.02;

    // ---- Stage control ----
    constexpr size_t RESERVOIR_WARMUP_STEPS = 1000;

    // ---- Free-run scoring ----
    constexpr float VPT_THRESHOLD = 0.3f;
    constexpr double LYAPUNOV_EXPONENT = 0.9056;
}

/// One free-run outcome: metrics the survey aggregates, plus a display row.
struct FreeRunResult
{
    bool valid = false;
    uint64_t seed = 0;
    uint64_t orbit_seed = 0;
    size_t vpt_steps = 0;
    bool crossed = false;
    double vpt_lt = 0.0;
    double rmse = 0.0;
    size_t steps = 0;
    /// GS / re-lock proxies (θ = VPT_THRESHOLD; channel-RMS):
    /// duty = fraction of steps with err ≤ θ; n_relock / n_unlock / mean locked sojourn.
    double duty = 0.0;
    size_t n_relock = 0;
    size_t n_unlock = 0;
    double mean_locked_sojourn = 0.0;
    std::string row;
};

/// @brief Online free-run experiment on Lorenz-63.
///
///   LorenzDatastream (normalized float stream + forward Cursor)
///       |  input port: [x, y, z, x*z]  real in train/washout; prediction in free-run
///       v
///   ESN (fixed reservoir + online HCNN readout; external feedback off)
///
/// Train(): teacher-forced one-step sweeps. FreeRun(): washout then generative
/// self-feedback on the input bank; score vs held-out orbit tail.
class Lorenz
{
public:
    Lorenz(uint64_t seed, uint64_t orbit_seed);

    void Train();

    /// Washout on the training window (teacher-forced), then generative free-run
    /// for FREE_RUN_WINDOW_SIZE steps. Optional per-step CSV via @p csv_path.
    FreeRunResult FreeRun(bool verbose, const char* csv_path = nullptr);

    [[nodiscard]] std::string ReadoutArchSummary() const {
        return esn_.ReadoutArchSummary();
    }

private:
    uint64_t seed_, orbit_seed_;

    ESNConfig esn_config_;
    ESN esn_;
    std::unique_ptr<LorenzDatastream> data_stream_;

    void RebuildDatastream(bool verbose);

    /// Pack input drive [x, y, z, x*z] from a real stream sample (teacher force).
    static void ExtractDriveReal(float drive[4], const NormalizedState& state);

    /// Pack input drive [x, y, z, x*z] from the model's prediction (free-run).
    static void ExtractDrivePredicted(float drive[4], const float* prediction);

    /// Horizon-1 targets: current sample's (x, y, z).
    static void ExtractTargets(float targets[3], const NormalizedState& state);

    static ESNConfig MakeESNConfig(uint64_t seed);
    static LorenzDatastreamConfig MakeDatastreamConfig(LorenzAttractor::State orbit);
    static float LrProfile(float lr_max, float lr_min, size_t epochs, size_t current_epoch);
};
