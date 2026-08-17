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

/// Fixed reservoir input drive: always 4 channels [x, y, z, x*z]
/// (ODE bilinear in y-dot). Free-run rebuilds x*z from predicted (x,y,z).
/// Must divide N = 2^DIM (true for all legal DIM).
inline constexpr size_t kNumDriveChannels = 4;

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
    inline size_t DIM = 10;
    constexpr uint64_t SEED = 696634088797950509ull;
    // Not constexpr: Parallel* / FreeRun* may reassign for a run (RAII restore).
    inline float SPECTRAL_RADIUS = 0.999f;
    // Not constexpr: Parallel* / FreeRun* may reassign (restored on campaign exit).
    inline float INPUT_SCALING = 0.015f;
    // Per-channel multipliers on top of global INPUT_SCALING (applied in FillDrive
    // after feature build). Index order: [x, y, z, x*z]. Locked soft z/xz.
    // Train and free-run/load must use the same gains (edit here, rebuild).
    inline constexpr float INPUT_SCALE_CH[kNumDriveChannels] = {1.f, 1.f, 0.9f, 0.7f};
    constexpr float LEAK_RATE = 1.0f;
    // Delay-line depth M. Not constexpr: campaigns pass M as an argument.
    // Reservoir requires M in [1, 64].
    inline size_t HISTORY_DEPTH = 2;

    // ---- Readout (HCNN), trained ONLINE (single-sample, multi-epoch) ----
    constexpr float LEARNING_RATE = 0.00004f;
    constexpr float LEARNING_RATE_MIN = 0.000001f;
    // Not constexpr: campaigns may reassign (surveys / heavy train).
    // Typical: 50–100 rapid A/B; 100–200 refine; 300–500 heavy train.
    inline size_t EPOCHS = 100;
    constexpr uint64_t READOUT_SEED = 696634088797950509ull;
    constexpr size_t READOUT_SLICES = 1;
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
    // Free-run seating: last W of train (edge), then generative from span+1.
    // W clamped to [1, span+1] at free-run use.
    constexpr size_t WARMUP_STEPS = 1000;

    // ---- Model I/O (readout HCNW + arch sidecar; reservoir is seed-reproducible) ----
    // Save: off by default. When true, Train() writes after the last epoch:
    //   {MODEL_SAVE_DIR}\lorenz_seed{SEED}_D{DIM}_M{M}_in{Nin}.hcnw + .arch.json
    // (DIM / M / num_inputs in the stem so layout and geometry do not collide.)
    constexpr bool SAVE_TRAINED_WEIGHTS = false;
    constexpr const char* MODEL_SAVE_DIR = R"(C:\HypercubeESN\models)";

    // Load: off by default. When true, skip Train() and load readout from stem
    // (no extension). ESN seed/arch/drive layout must match the run that produced
    // the file. Free-run after load uses fixed IC / remix orbits (no train required).
    // Example stem (match save naming):
    //   C:\HypercubeESN\models\lorenz_seed21978990_D11_M24_in4
    constexpr bool LOAD_TRAINED_WEIGHTS = false;
    constexpr const char* LOAD_WEIGHTS_STEM =
        R"(C:\HypercubeESN\models\lorenz_seed21978990_D11_M24_in4)";

    // Campaign artifacts (CWD-independent), all under C:\HypercubeESN. Layout:
    //   RUNS_DIR/traces/   FreeRun plottable CSVs
    //   RUNS_DIR/surveys/  SeedSweep + OrbitSweep leaderboards
    //   RESULTS_DIR        legacy path alias under RUNS_DIR (unused by keepers)
    //   MODEL_SAVE_DIR     trained readout weights (sibling of results/)
    constexpr const char* RUNS_DIR = R"(C:\HypercubeESN\results)";
    constexpr const char* RESULTS_DIR = R"(C:\HypercubeESN\results\campaigns)";

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
///       |  input port: fixed 4-in [x, y, z, x*z]
///       |    real in train/warmup; prediction in free-run
///       v
///   ESN (fixed reservoir + online HCNN readout; external feedback off)
///
/// Train(): multi-orbit online epochs (new IC each epoch); full train+runway stream.
/// FreeRun(): multi-IC challenge — edge warmup then generative past span; stream is
/// slim (wash + free-run runway only, with discarded burn-in for orbit phase).
class Lorenz
{
public:
    Lorenz(uint64_t seed, uint64_t orbit_seed);

    void Train();

    /// Load readout from @p stem (no .hcnw extension), or config::LOAD_WEIGHTS_STEM
    /// when @p stem is null/empty. Throws if stem empty / load fails.
    /// FreeRun with fixed_ic / fixed_orbit_seed / remix still works after load.
    /// @p log_load  If true (default), print a stderr notice. Parallel campaigns
    ///              that load once per job should pass false and log a single summary.
    void LoadTrainedWeights(const char* stem = nullptr, bool log_load = true);

    /// Save readout to @p stem (no .hcnw extension). If null/empty, uses the
    /// default under MODEL_SAVE_DIR: lorenz_seed{S}_D{DIM}_M{M}_in{Nin}.
    /// Always writes (unlike SaveTrainedWeightsIfEnabled, which is config-gated).
    void SaveTrainedWeights(const char* stem = nullptr) const;

    /// Free-run: edge warmup (last W of train section) then generative past span.
    /// @p warmup_steps 0 → config::WARMUP_STEPS.
    /// @p fixed_orbit_seed if non-zero, build the stream from this seed (no remix).
    /// @p fixed_ic if non-null, build from this attractor IC (takes priority over
    /// @p fixed_orbit_seed). Same IC space as IcFromOrbitSeed.
    /// Otherwise remixed new orbit (multi-IC challenge).
    /// When @p verbose and ENABLE_PRINTF, prints every generative step (pred + true xyz).
    /// @p freerun_steps 0 → config::FREE_RUN_WINDOW_SIZE (generative runway length).
    FreeRunResult FreeRun(bool verbose, const char* csv_path = nullptr,
                          size_t warmup_steps = 0,
                          uint64_t fixed_orbit_seed = 0,
                          const LorenzAttractor::State* fixed_ic = nullptr,
                          size_t freerun_steps = 0);

    [[nodiscard]] std::string ReadoutArchSummary() const {
        return esn_.ReadoutArchSummary();
    }

    /// HCNN readout worker count used by MakeESNConfig. Must remain 1 so host
    /// campaigns (SeedSweep, OrbitSweep) can run many Lorenz instances
    /// without nested HCNN thread pools. Do not change to 0 (auto) or N>1
    /// without revisiting those campaigns.
    static constexpr size_t kReadoutNumThreads = 1;

    /// Map orbit seed → attractor IC (same map as freerun remix / train remix).
    /// Public so OrbitSweep can print ICs for FreeRun cherry-picks.
    static LorenzAttractor::State IcFromOrbitSeed(uint64_t orbit_seed);

private:
    uint64_t seed_, orbit_seed_;

    ESNConfig esn_config_;
    ESN esn_;
    std::unique_ptr<LorenzDatastream> data_stream_;

    /// Advance remix chain and build a full train-length stream (Train only).
    void RebuildDatastream(bool verbose);
    /// Build full train+runway stream from a fixed orbit seed (Train path).
    void BuildDatastreamFromSeed(uint64_t orbit_seed, bool verbose);
    /// Build full train+runway stream from an explicit attractor IC (Train path).
    void BuildDatastreamFromIC(LorenzAttractor::State ic, bool verbose);

    /// Free-run only: slim stream (wash + runway) with burn-in discard for phase.
    void BuildFreeRunDatastream(LorenzAttractor::State ic, size_t warmup_steps,
                                size_t freerun_steps);

    /// If config::SAVE_TRAINED_WEIGHTS, write default-stem readout under MODEL_SAVE_DIR.
    void SaveTrainedWeightsIfEnabled() const;

    /// Fill drive[0..kNumDriveChannels) = [x, y, z, x*z] with channel gains.
    static void FillDrive(float* drive, float x, float y, float z);
    static void ExtractDriveReal(float* drive, const NormalizedState& state);
    static void ExtractDrivePredicted(float* drive, const float* prediction);
    static void ExtractTargets(float targets[3], const NormalizedState& state);

    static ESNConfig MakeESNConfig(uint64_t seed);
    /// Full train window + free-run runway (Train).
    static LorenzDatastreamConfig MakeDatastreamConfig(LorenzAttractor::State orbit);
    /// Compact free-run stream: wash of @p warmup_steps + generative @p freerun_steps.
    /// @p freerun_steps 0 → config::FREE_RUN_WINDOW_SIZE.
    static LorenzDatastreamConfig MakeFreeRunDatastreamConfig(LorenzAttractor::State orbit,
                                                             size_t warmup_steps,
                                                             size_t freerun_steps = 0);
    static float LrProfile(float lr_max, float lr_min, size_t epochs, size_t current_epoch);
};
