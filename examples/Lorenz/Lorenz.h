#pragma once

#include "ESN.h"
#include "LorenzAttractor.h"
#include "LorenzDatastream.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ============================================================================
//  CONFIGURATION — consolidation point for primary variables of interest
// ============================================================================

/// Free-run protocol (orbit choice + where generative scoring starts).
enum class FreeRunProtocol
{
    /// Challenge (default): new IC each free-run (not a train replay);
    /// edge warmup then free-run past span on that orbit.
    Unseen = 0,
    /// Easy / in-sample: replay a training orbit; warmup from start of train;
    /// free-run scores only while still inside the train window (index <= span).
    TrainInSample = 1,
    /// Temporal holdout on a train orbit: edge warmup on last W of train;
    /// free-run starts at span+1 (first point past the train section).
    TrainHoldout = 2,
};

/// Reservoir input-drive layout (num_inputs must divide N = 2^DIM).
/// Free-run rebuilds products from predicted (x,y,z) -- no denorm/renorm.
enum class DriveLayout
{
    /// 4 channels: [x, y, z, x*z]  (ODE bilinear in y-dot). Baseline.
    XyzXz = 0,
    /// 8 channels: [x, y, z, x*y, x*z, x*x, y*y, z*z]
    /// Both ODE bilinears + pure quadratic pad (drops y*z; full deg-2 is 9).
    Quadratic8 = 1,
};

/// Max channels among DriveLayout variants (stack buffers / CSV).
inline constexpr size_t kMaxDriveChannels = 8;

namespace config
{
    // ---- Diagnostics ----
    // Verbose diagnostics: config banner, per-epoch train RMSE, free-run traces.
    // Survey campaigns force this false; Trace turns it on for train.
    inline bool ENABLE_PRINTF = true;
    // Coarse progress on stderr: train epoch heartbeats, free-run k/N, phase lines.
    // Set false for quiet overnight runs (final reports still go to stdout).
    // WARN / error / save-load notices always print.
    inline bool ENABLE_PROGRESS = false;

    // ---- Reservoir / model ----
    // Hypercube dim (N = 2^DIM). Not constexpr: campaigns pass DIM as an argument.
    // Reservoir requires 5 <= dim <= 16.
    inline size_t DIM = 11;
    constexpr uint64_t SEED = 13649419;
    constexpr float SPECTRAL_RADIUS = 0.99f;
    constexpr float INPUT_SCALING = 0.04f;
    constexpr float LEAK_RATE = 1.0f;
    // Delay-line depth M. Not constexpr: campaigns (e.g. M-sweep) may reassign.
    // Reservoir requires M in [1, 64].
    inline size_t HISTORY_DEPTH = 18;   // 18 is optimal for DIM11.
    // Input-drive feature map (see DriveLayout). Campaigns may reassign (A/B).
    // Load/save readout must match the layout used at train time.
    inline DriveLayout DRIVE_LAYOUT = DriveLayout::XyzXz;

    // ---- Readout (HCNN), trained ONLINE (single-sample, multi-epoch) ----
    constexpr float LEARNING_RATE = 0.00004f;
    constexpr float LEARNING_RATE_MIN = 0.000002f;
    constexpr size_t EPOCHS = 100;  // 50 for rapid comparison tests, 100 for reports
    constexpr size_t READOUT_SLICES = 2;
    constexpr size_t CONV_CHANNELS = 1;
    constexpr int NUM_LAYERS = 1;
    constexpr bool USE_POOLING = false;
    constexpr ReadoutActivation READOUT_ACTIVATION = ReadoutActivation::RELU;

    // ---- Data stream (Lorenz-63 + forward cursor window) ----
    // Layout: train [0, TRAINING_WINDOW_SIZE] inclusive; free-run runway after span.
    constexpr int32_t TRAINING_WINDOW_SIZE = 20000;
    constexpr size_t FREE_RUN_WINDOW_SIZE = 1000;
    constexpr size_t STREAM_LENGTH =
        static_cast<size_t>(TRAINING_WINDOW_SIZE) + FREE_RUN_WINDOW_SIZE;

    constexpr double DT = 0.02;

    // ---- Stage control ----
    // Teacher-forced open-loop reservoir drive before the useful phase:
    // train → before TrainStep; free-run → before generative scoring.
    // Free-run seating: Unseen/TrainHoldout = last W of train (edge);
    // TrainInSample = first W of train. Clamped to [1, span+1] at free-run use.
    constexpr size_t WARMUP_STEPS = 1000;

    // Default free-run arm (trainholdout).
    constexpr FreeRunProtocol FREE_RUN_PROTOCOL = FreeRunProtocol::TrainHoldout;

    // ---- Model I/O (readout HCNW + arch sidecar; reservoir is seed-reproducible) ----
    // Save: off by default. When true, Train() writes after the last epoch:
    //   {MODEL_SAVE_DIR}\lorenz_seed{SEED}_D{DIM}_M{M}_in{Nin}.hcnw + .arch.json
    // (DIM / M / num_inputs in the stem so layout and geometry do not collide.)
    constexpr bool SAVE_TRAINED_WEIGHTS = false;
    constexpr const char* MODEL_SAVE_DIR = R"(C:\\HypercubeESN\\models)";

    // Load: off by default. When true, skip Train() and load readout from stem
    // (no extension). ESN seed/arch/drive layout must match the run that produced
    // the file. Free-run after load is Unseen only (no train-orbit list).
    // Example stem (match save naming):
    //   C:\HypercubeESN\models\lorenz_seed21978990_D11_M24_in4
    constexpr bool LOAD_TRAINED_WEIGHTS = false;
    constexpr const char* LOAD_WEIGHTS_STEM =
        R"(C:\HypercubeESN\models\lorenz_seed21978990_D11_M24_in4)";

    // Campaign results (survey aggregates, M-sweep roll-ups). Created if missing.
    constexpr const char* RESULTS_DIR = R"(C:\HypercubeESN\results)";

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
    FreeRunProtocol protocol = FreeRunProtocol::Unseen;
    size_t vpt_steps = 0;
    bool crossed = false;
    double vpt_lt = 0.0;
    double rmse = 0.0;
    size_t steps = 0;
    double duty = 0.0;
    size_t n_relock = 0;
    size_t n_unlock = 0;
    double mean_locked_sojourn = 0.0;
    std::string row;
};

/// @brief Online free-run experiment on Lorenz-63.
///
///   LorenzDatastream (normalized float stream + forward Cursor)
///       |  input port: DriveLayout (4-in xz or 8-in quadratic)
///       |    real in train/warmup; prediction in free-run
///       v
///   ESN (fixed reservoir + online HCNN readout; external feedback off)
///
/// Train(): multi-orbit online epochs (new IC each epoch). FreeRun(): one of three
/// protocols (Unseen / TrainInSample / TrainHoldout) — see FreeRunProtocol.
class Lorenz
{
public:
    Lorenz(uint64_t seed, uint64_t orbit_seed);

    void Train();

    /// Load readout from config::LOAD_WEIGHTS_STEM (throws if stem empty / load fails).
    /// Clears train-orbit list (TrainInSample / TrainHoldout unavailable until Train()).
    void LoadTrainedWeights();

    /// Free-run under @p protocol (default @c config::FREE_RUN_PROTOCOL).
    /// @p warmup_steps 0 → config::WARMUP_STEPS.
    /// @p train_orbit_index for TrainInSample / TrainHoldout: which stored train
    /// orbit (SIZE_MAX = auto-cycle). Ignored for Unseen.
    FreeRunResult FreeRun(bool verbose, const char* csv_path = nullptr,
                          size_t warmup_steps = 0,
                          FreeRunProtocol protocol = config::FREE_RUN_PROTOCOL,
                          size_t train_orbit_index = static_cast<size_t>(-1));

    [[nodiscard]] std::string ReadoutArchSummary() const {
        return esn_.ReadoutArchSummary();
    }

    [[nodiscard]] size_t NumTrainOrbits() const { return train_orbit_seeds_.size(); }

    /// Config protocol, or Unseen when load-only (no train-orbit list).
    [[nodiscard]] FreeRunProtocol EffectiveFreeRunProtocol() const;

    static const char* ProtocolName(FreeRunProtocol p);
    static const char* DriveLayoutName(DriveLayout layout = config::DRIVE_LAYOUT);
    /// Channel count for @p layout (4 or 8); must divide N.
    static size_t NumDriveChannels(DriveLayout layout = config::DRIVE_LAYOUT);

private:
    uint64_t seed_, orbit_seed_;
    /// Post-mix seeds used for each training epoch (for TrainInSample / TrainHoldout).
    std::vector<uint64_t> train_orbit_seeds_;
    size_t next_train_orbit_pick_ = 0;
    bool weights_loaded_ = false; // true after LoadTrainedWeights(); skips train-orbit free-run

    ESNConfig esn_config_;
    ESN esn_;
    std::unique_ptr<LorenzDatastream> data_stream_;

    /// Advance remix chain and build a new stream (train + Unseen free-run).
    void RebuildDatastream(bool verbose);
    /// Build stream from a fixed orbit seed without advancing orbit_seed_.
    void BuildDatastreamFromSeed(uint64_t orbit_seed, bool verbose);

    /// If config::SAVE_TRAINED_WEIGHTS, write readout HCNW under MODEL_SAVE_DIR.
    void SaveTrainedWeightsIfEnabled() const;

    static LorenzAttractor::State IcFromOrbitSeed(uint64_t orbit_seed);

    /// Fill drive[0..NumDriveChannels) from normalized (x,y,z).
    static void FillDrive(float* drive, float x, float y, float z);
    static void ExtractDriveReal(float* drive, const NormalizedState& state);
    static void ExtractDrivePredicted(float* drive, const float* prediction);
    static void ExtractTargets(float targets[3], const NormalizedState& state);

    static ESNConfig MakeESNConfig(uint64_t seed);
    static LorenzDatastreamConfig MakeDatastreamConfig(LorenzAttractor::State orbit);
    static float LrProfile(float lr_max, float lr_min, size_t epochs, size_t current_epoch);
};
