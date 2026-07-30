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
    constexpr uint64_t SEED = 665127;//13649419;        //21978990 achieved 10.07 for oribit seed 9333312947715283458
    // Not constexpr: Campaign_SpectralRadiusAB reassigns for matched A/B.
    inline float SPECTRAL_RADIUS = 0.99f;
    constexpr float INPUT_SCALING = 0.04f;
    // Per-channel multipliers on top of global INPUT_SCALING (applied in FillDrive
    // after feature build, train + free-run). Index order matches DriveLayout:
    //   XyzXz:      [x, y, z, x*z]
    //   Quadratic8: [x, y, z, x*y, x*z, x*x, y*y, z*z]
    // Unused trailing slots ignored. Default unity = global scale only.
    // Reassignable for A/B; train and free-run/load must use the same gains.
    // See TODO_drive_scale_sr.md §4 for a first z / xz grid.
    inline float INPUT_SCALE_CH[kMaxDriveChannels] = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
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
    // Not constexpr: campaigns may reassign (surveys / heavy train).
    // Typical: 50–100 rapid A/B; 100–200 refine; 300–500 heavy train.
    inline size_t EPOCHS = 100;
    constexpr size_t READOUT_SLICES = 2;
    constexpr size_t CONV_CHANNELS = 1;
    constexpr int NUM_LAYERS = 1;
    constexpr bool USE_POOLING = true;
    constexpr ReadoutActivation READOUT_ACTIVATION = ReadoutActivation::TANH;

    // ---- Data stream (Lorenz-63 + forward cursor window) ----
    // Layout: train [0, TRAINING_WINDOW_SIZE] inclusive; free-run runway after span.
    constexpr int32_t TRAINING_WINDOW_SIZE = 20000;
    constexpr size_t FREE_RUN_WINDOW_SIZE = 2000;
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
    constexpr FreeRunProtocol FREE_RUN_PROTOCOL = FreeRunProtocol::Unseen;

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

    // Campaign artifacts (CWD-independent). Layout:
    //   RUNS_DIR/traces/     FreeRun + Campaign_Trace freerun CSVs
    //   RUNS_DIR/surveys/    FreeRunSurvey + SeedSweep leaderboards
    //   RUNS_DIR/campaigns/  SeedSurvey / M-sweep / DriveAB roll-ups (RESULTS_DIR)
    constexpr const char* RUNS_DIR = R"(C:\HypercubeESNRuns\results)";
    constexpr const char* RESULTS_DIR = R"(C:\HypercubeESNRuns\results\campaigns)";

    // ---- Free-run scoring ----
    constexpr float VPT_THRESHOLD = 0.25f;
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
    /// VPT (lt) * duty — first-hold length scaled by time-in-lock over the window.
    double vpt_x_duty = 0.0;
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

    /// Load readout from @p stem (no .hcnw extension), or config::LOAD_WEIGHTS_STEM
    /// when @p stem is null/empty. Throws if stem empty / load fails.
    /// Clears train-orbit list (TrainInSample / TrainHoldout list pick unavailable
    /// until Train(); FreeRun with fixed_ic / fixed_orbit_seed still works).
    void LoadTrainedWeights(const char* stem = nullptr);

    /// Save readout to @p stem (no .hcnw extension). If null/empty, uses the
    /// default under MODEL_SAVE_DIR: lorenz_seed{S}_D{DIM}_M{M}_in{Nin}.
    /// Always writes (unlike SaveTrainedWeightsIfEnabled, which is config-gated).
    void SaveTrainedWeights(const char* stem = nullptr) const;

    /// Free-run under @p protocol (default @c config::FREE_RUN_PROTOCOL).
    /// @p warmup_steps 0 → config::WARMUP_STEPS.
    /// @p train_orbit_index for TrainInSample / TrainHoldout: which stored train
    /// orbit (SIZE_MAX = auto-cycle). Ignored when a fixed stream is supplied.
    /// @p fixed_orbit_seed if non-zero, build the stream from this seed (no remix /
    /// train-list pick). Use for replaying a known orbit under the current protocol.
    /// @p fixed_ic if non-null, build the stream from this attractor IC (takes
    /// priority over @p fixed_orbit_seed). Same IC space as IcFromOrbitSeed.
    /// When @p verbose and ENABLE_PRINTF, prints every generative step (pred + true xyz).
    FreeRunResult FreeRun(bool verbose, const char* csv_path = nullptr,
                          size_t warmup_steps = 0,
                          FreeRunProtocol protocol = config::FREE_RUN_PROTOCOL,
                          size_t train_orbit_index = static_cast<size_t>(-1),
                          uint64_t fixed_orbit_seed = 0,
                          const LorenzAttractor::State* fixed_ic = nullptr);

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

    /// Map orbit seed → attractor IC (same map as freerun Unseen / train remix).
    /// Public so FreeRunSurvey can print ICs for FreeRun cherry-picks.
    static LorenzAttractor::State IcFromOrbitSeed(uint64_t orbit_seed);

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
    /// Build stream from an explicit attractor IC (no remix / seed map).
    void BuildDatastreamFromIC(LorenzAttractor::State ic, bool verbose);

    /// If config::SAVE_TRAINED_WEIGHTS, write default-stem readout under MODEL_SAVE_DIR.
    void SaveTrainedWeightsIfEnabled() const;

    /// Fill drive[0..NumDriveChannels) from normalized (x,y,z).
    static void FillDrive(float* drive, float x, float y, float z);
    static void ExtractDriveReal(float* drive, const NormalizedState& state);
    static void ExtractDrivePredicted(float* drive, const float* prediction);
    static void ExtractTargets(float targets[3], const NormalizedState& state);

    static ESNConfig MakeESNConfig(uint64_t seed);
    static LorenzDatastreamConfig MakeDatastreamConfig(LorenzAttractor::State orbit);
    static float LrProfile(float lr_max, float lr_min, size_t epochs, size_t current_epoch);
};
