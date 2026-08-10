#include "Lorenz.h"
#include "LorenzDatastream.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

ESNConfig Lorenz::MakeESNConfig(uint64_t seed)
{
    ESNConfig cfg;
    cfg.reservoir.dim = config::DIM;
    cfg.reservoir.seed = seed;

    const size_t n_in = kNumDriveChannels;
    const size_t N = size_t{1} << config::DIM;
    if (N % n_in != 0)
        throw std::invalid_argument(
            "MakeESNConfig: kNumDriveChannels must divide N = 2^DIM");
    cfg.reservoir.num_inputs = n_in;
    cfg.reservoir.num_external_feedback_channels = 0;
    cfg.reservoir.spectral_radius = config::SPECTRAL_RADIUS;
    cfg.reservoir.input_scaling = config::INPUT_SCALING;
    cfg.reservoir.leak_rate = config::LEAK_RATE;
    cfg.reservoir.history_depth = config::HISTORY_DEPTH;
    cfg.reservoir.bias_scaling = 0.003f;

    cfg.readout.num_outputs = 3; // [x, y, z]
    cfg.readout.seed = seed;//config::READOUT_SEED;
    cfg.readout_slices = config::READOUT_SLICES;
    cfg.readout.use_pooling = config::USE_POOLING;
    cfg.readout.num_layers = config::NUM_LAYERS;
    cfg.readout.momentum = 0.9f;
    cfg.readout.conv_channels = config::CONV_CHANNELS;
    cfg.readout.num_threads = kReadoutNumThreads;
    cfg.readout.task = ReadoutTask::Regression;
    cfg.readout.activation = config::READOUT_ACTIVATION;
    return cfg;
}

LorenzDatastreamConfig Lorenz::MakeDatastreamConfig(LorenzAttractor::State orbit)
{
    LorenzDatastreamConfig cfg;
    cfg.stream_length = config::STREAM_LENGTH;
    cfg.span = config::TRAINING_WINDOW_SIZE;
    cfg.discard_steps = 0;
    cfg.initial_lorenz_state = orbit;
    cfg.lorenz_dt = static_cast<float>(config::DT);
    return cfg;
}

LorenzDatastreamConfig Lorenz::MakeFreeRunDatastreamConfig(LorenzAttractor::State orbit,
                                                           size_t warmup_steps,
                                                           size_t freerun_steps)
{
    // Resolve W first so storage matches seating (edge wash of W, then FR steps).
    size_t W = (warmup_steps == 0) ? config::WARMUP_STEPS : warmup_steps;
    if (W < 1)
        W = 1;
    size_t FR = (freerun_steps == 0) ? config::FREE_RUN_WINDOW_SIZE : freerun_steps;
    if (FR < 1)
        FR = 1;

    // Local train edge: wash_start = span - W + 1. Prefer span = W-1 => wash at 0.
    // Cursor requires span > 0, so W==1 forces span=1 (wash_start=1).
    int32_t span = static_cast<int32_t>(W) - 1;
    if (span < 1)
        span = 1;

    LorenzDatastreamConfig cfg;
    cfg.span = span;
    // After wash, index = span+1; freerun needs span+1 .. span+FR inclusive end span+FR.
    // stream_length+1 samples => indices 0..stream_length; set stream_length = span+FR.
    cfg.stream_length = static_cast<size_t>(span) + FR;

    // Burn-in without storage so local wash lands at the same absolute orbit phase
    // as the old full-stream edge: local_wash + discard == TRAINING_WINDOW - W + 1
    //  => discard = TRAINING_WINDOW - span  (when span = W-1, discard = TW - W + 1).
    if (static_cast<size_t>(span) <= static_cast<size_t>(config::TRAINING_WINDOW_SIZE))
        cfg.discard_steps =
            static_cast<size_t>(config::TRAINING_WINDOW_SIZE) - static_cast<size_t>(span);
    else
        cfg.discard_steps = 0;

    // Lock min/max scale to the survey default runway (span + FREE_RUN_WINDOW_SIZE)
    // so freerun_steps > default only extends the tail — wash + early VPT stay
    // bit-identical to OrbitSweep / FR=config::FREE_RUN_WINDOW_SIZE.
    cfg.normalize_count =
        static_cast<size_t>(span) + config::FREE_RUN_WINDOW_SIZE + 1;

    cfg.initial_lorenz_state = orbit;
    cfg.lorenz_dt = static_cast<float>(config::DT);
    return cfg;
}

Lorenz::Lorenz(const uint64_t seed, uint64_t orbit_seed) : seed_(seed),
    orbit_seed_(orbit_seed),
    esn_config_(MakeESNConfig(seed_)),
    esn_(esn_config_)
{
    if (config::ENABLE_PRINTF)
    {
        std::printf("[Lorenz config] reservoir: DIM=%zu (N=%zu)  seed=%llu  SR=%.3f  input_scaling=%.4f  leak=%.2f"
                    "  history_depth=%zu\n",
                    config::DIM, size_t{1} << config::DIM, static_cast<unsigned long long>(seed_),
                    config::SPECTRAL_RADIUS, config::INPUT_SCALING, config::LEAK_RATE,
                    config::HISTORY_DEPTH);
        std::printf("[Lorenz config] ports:     input=%zu drive=[x,y,z,xz]  ext_feedback=0 (off)\n",
                    esn_config_.reservoir.num_inputs);
        {
            std::printf("[Lorenz config] drive_ch:  [");
            for (size_t i = 0; i < kNumDriveChannels; ++i)
                std::printf("%s%.3f", i ? ", " : "", config::INPUT_SCALE_CH[i]);
            std::printf("]  (x global INPUT_SCALING)\n");
        }
        std::printf("[Lorenz config] free-run:  edge warmup=%zu  window=%zu\n",
                    config::WARMUP_STEPS, config::FREE_RUN_WINDOW_SIZE);
        std::printf("[Lorenz config] model I/O: save=%s  load=%s\n",
                    config::SAVE_TRAINED_WEIGHTS ? "on" : "off",
                    config::LOAD_TRAINED_WEIGHTS ? "on" : "off");
        if (config::LOAD_TRAINED_WEIGHTS)
            std::printf("[Lorenz config] load stem: %s\n", config::LOAD_WEIGHTS_STEM);
        std::printf("[Lorenz config] readout:   lr %.6f -> %.6f   epochs=%zu\n",
                    config::LEARNING_RATE, config::LEARNING_RATE_MIN, config::EPOCHS);
        const char* act =
            config::READOUT_ACTIVATION == ReadoutActivation::TANH ? "TANH" :
            config::READOUT_ACTIVATION == ReadoutActivation::RELU ? "RELU" :
            config::READOUT_ACTIVATION == ReadoutActivation::LEAKY_RELU ? "LEAKY_RELU" : "NONE";
        std::printf("[Lorenz config] readout in: slices=%zu  pooling=%s  activation=%s\n",
                    config::READOUT_SLICES, config::USE_POOLING ? "on" : "off", act);
        std::printf("[Lorenz config] stream:    train_span=%d  stream_len=%zu  warmup=%zu\n",
                    config::TRAINING_WINDOW_SIZE, config::STREAM_LENGTH,
                    config::WARMUP_STEPS);
    }
}

static inline uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

LorenzAttractor::State Lorenz::IcFromOrbitSeed(const uint64_t orbit_seed)
{
    std::mt19937_64 rng(orbit_seed);
    std::uniform_real_distribution<double> dist(-0.999, 0.999);
    std::uniform_real_distribution<double> dist_uni(0.0, 0.999);
    return {dist(rng), dist(rng), dist_uni(rng)};
}

void Lorenz::BuildDatastreamFromSeed(const uint64_t orbit_seed, const bool verbose)
{
    data_stream_ = std::make_unique<LorenzDatastream>(
        MakeDatastreamConfig(IcFromOrbitSeed(orbit_seed)));
    if (verbose)
        data_stream_->PrintOrbit();
}

void Lorenz::BuildDatastreamFromIC(const LorenzAttractor::State ic, const bool verbose)
{
    data_stream_ = std::make_unique<LorenzDatastream>(MakeDatastreamConfig(ic));
    if (verbose)
        data_stream_->PrintOrbit();
}

void Lorenz::RebuildDatastream(const bool verbose)
{
    orbit_seed_ = mix64(orbit_seed_ ^ (0x100000001B3ULL));
    BuildDatastreamFromSeed(orbit_seed_, verbose);
}

void Lorenz::BuildFreeRunDatastream(const LorenzAttractor::State ic, const size_t warmup_steps,
                                    const size_t freerun_steps)
{
    data_stream_ = std::make_unique<LorenzDatastream>(
        MakeFreeRunDatastreamConfig(ic, warmup_steps, freerun_steps));
}

void Lorenz::FillDrive(float* drive, const float x, const float y, const float z)
{
    // Fixed [x, y, z, x*z]; product from same-step values (free-run safe).
    drive[0] = x;
    drive[1] = y;
    drive[2] = z;
    drive[3] = x * z;
    for (size_t i = 0; i < kNumDriveChannels; ++i)
        drive[i] *= config::INPUT_SCALE_CH[i];
}

void Lorenz::ExtractDriveReal(float* drive, const NormalizedState& state)
{
    FillDrive(drive, state.x, state.y, state.z);
}

void Lorenz::ExtractDrivePredicted(float* drive, const float* prediction)
{
    FillDrive(drive, prediction[0], prediction[1], prediction[2]);
}

void Lorenz::ExtractTargets(float targets[3], const NormalizedState& state)
{
    targets[0] = state.x;
    targets[1] = state.y;
    targets[2] = state.z;
}

float Lorenz::LrProfile(const float lr_max, const float lr_min, const size_t epochs, const size_t current_epoch)
{
    if (epochs <= 1)
        return lr_max;
    // Full-horizon cosine: lr hits lr_min only on the last epoch (no long
    // floor tail). Progress in [0, 1] over epochs-1 steps.
    const float progress = static_cast<float>(current_epoch) / static_cast<float>(epochs - 1);
    return CosineLR(progress, lr_max, lr_min);
}

void Lorenz::Train()
{
    float drive[kNumDriveChannels] = {};
    float targets[3] = {};
    float outputs[3] = {};

    for (size_t i = 0; i < config::EPOCHS; i++)
    {
        RebuildDatastream(config::ENABLE_PRINTF);

        data_stream_->Reset();
        const float lr = LrProfile(config::LEARNING_RATE, config::LEARNING_RATE_MIN, config::EPOCHS, i);
        LorenzDatastreamResult st = data_stream_->States();

        for (size_t j = 0; j < config::WARMUP_STEPS; j++)
        {
            if (data_stream_->OOB() || st.sample == nullptr)
                break;
            ExtractDriveReal(drive, *st.sample);
            esn_.ReservoirStep(drive, nullptr);
            st = data_stream_->Step();
        }

        double sq_err_sum = 0.0;
        size_t train_steps = 0;
        while (!data_stream_->OOB() && st.sample != nullptr)
        {
            ExtractTargets(targets, *st.sample);
            ExtractDriveReal(drive, *st.sample);

            esn_.Predict(outputs);
            esn_.TrainStep(targets, lr);
            esn_.ReservoirStep(drive, nullptr);

            for (size_t c = 0; c < 3; c++)
            {
                const double e = static_cast<double>(outputs[c]) - targets[c];
                sq_err_sum += e * e;
            }
            ++train_steps;

            st = data_stream_->Step();
        }

        if (config::ENABLE_PRINTF)
        {
            // Continues same line after State::print() IC prefix (fixed columns).
            if (train_steps > 0)
                std::printf("epoch %3zu  lr %.7f  train RMSE %.6f\n",
                            i, lr, std::sqrt(sq_err_sum / (3.0 * train_steps)));
            else
                std::printf("epoch %3zu  lr %.7f  train RMSE n/a  (0 steps - warmup consumed the window)\n",
                            i, lr);
        }
        else if (config::ENABLE_PROGRESS &&
                 ((i + 1) % 10 == 0 || i + 1 == config::EPOCHS))
        {
            std::fprintf(stderr, "[seed %llu] train epoch %zu/%zu\n",
                         static_cast<unsigned long long>(seed_),
                         i + 1, config::EPOCHS);
            std::fflush(stderr);
        }
    }

    SaveTrainedWeightsIfEnabled();
}

void Lorenz::SaveTrainedWeights(const char* stem) const
{
    namespace fs = std::filesystem;
    std::string path;
    if (stem && stem[0] != '\0')
    {
        path = stem;
        const fs::path parent = fs::path(path).parent_path();
        if (!parent.empty())
        {
            std::error_code ec;
            fs::create_directories(parent, ec);
            if (ec)
            {
                std::fprintf(stderr, "[Lorenz] SaveTrainedWeights: create_directories(%s) failed: %s\n",
                             parent.string().c_str(), ec.message().c_str());
                throw std::runtime_error("SaveTrainedWeights: could not create parent directory");
            }
        }
    }
    else
    {
        const fs::path dir = config::MODEL_SAVE_DIR;
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec)
        {
            std::fprintf(stderr, "[Lorenz] SaveTrainedWeights: create_directories(%s) failed: %s\n",
                         dir.string().c_str(), ec.message().c_str());
            throw std::runtime_error("SaveTrainedWeights: could not create MODEL_SAVE_DIR");
        }
        // Include DIM, M, num_inputs so geometry / drive-layout A/B do not collide.
        path = (dir / ("lorenz_seed" + std::to_string(seed_) +
                       "_D" + std::to_string(config::DIM) +
                       "_M" + std::to_string(config::HISTORY_DEPTH) +
                       "_in" + std::to_string(kNumDriveChannels))).string();
    }

    // Stem only -- SaveReadoutHcnnModel appends .hcnw and .arch.json.
    esn_.SaveReadoutHcnnModel(path);
    std::fprintf(stderr, "[Lorenz] saved trained readout: %s.hcnw (+ .arch.json)\n",
                 path.c_str());
    std::fflush(stderr);
}

void Lorenz::SaveTrainedWeightsIfEnabled() const
{
    if (!config::SAVE_TRAINED_WEIGHTS)
        return;
    try
    {
        SaveTrainedWeights(nullptr);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "[Lorenz] SAVE_TRAINED_WEIGHTS failed: %s\n", e.what());
        std::fflush(stderr);
    }
}

void Lorenz::LoadTrainedWeights(const char* stem, bool log_load)
{
    const char* path = (stem && stem[0] != '\0') ? stem : config::LOAD_WEIGHTS_STEM;
    if (path == nullptr || path[0] == '\0')
        throw std::invalid_argument(
            "LoadTrainedWeights: stem empty "
            "(pass a path or set config::LOAD_WEIGHTS_STEM without .hcnw)");

    esn_.LoadReadoutHcnnModel(path);
    if (log_load)
    {
        std::fprintf(stderr, "[Lorenz] loaded trained readout: %s.hcnw (+ .arch.json)  "
                             "seed=%llu\n",
                     path, static_cast<unsigned long long>(seed_));
        std::fflush(stderr);
    }
}

FreeRunResult Lorenz::FreeRun(bool verbose, const char* csv_path, size_t warmup_steps,
                              uint64_t fixed_orbit_seed,
                              const LorenzAttractor::State* fixed_ic,
                              size_t freerun_steps)
{
    float drive[kNumDriveChannels] = {};
    float targets[3] = {};
    float outputs[3] = {};
    const size_t n_drive = kNumDriveChannels;

    // Size the slim stream for the intended wash + generative runway.
    size_t W = (warmup_steps == 0) ? config::WARMUP_STEPS : warmup_steps;
    if (W < 1)
        W = 1;
    size_t FR = (freerun_steps == 0) ? config::FREE_RUN_WINDOW_SIZE : freerun_steps;
    if (FR < 1)
        FR = 1;

    uint64_t freerun_orbit_seed = 0;
    LorenzAttractor::State freerun_ic{};
    if (fixed_ic)
    {
        freerun_orbit_seed = 0;
        freerun_ic = *fixed_ic;
    }
    else if (fixed_orbit_seed != 0)
    {
        freerun_orbit_seed = fixed_orbit_seed;
        freerun_ic = IcFromOrbitSeed(freerun_orbit_seed);
    }
    else
    {
        // Multi-IC challenge: advance remix chain (do not build a train-length stream).
        orbit_seed_ = mix64(orbit_seed_ ^ (0x100000001B3ULL));
        freerun_orbit_seed = orbit_seed_;
        freerun_ic = IcFromOrbitSeed(orbit_seed_);
    }
    BuildFreeRunDatastream(freerun_ic, W, FR);

    std::ofstream csv;
    if (csv_path && csv_path[0])
    {
        csv.open(csv_path, std::ios::out | std::ios::trunc);
        if (csv)
        {
            csv << "step,lt,err,locked,pred_x,pred_y,pred_z,true_x,true_y,true_z"
                   ",drive_x,drive_y,drive_z,drive_xz\n";
        }
        else if (verbose || config::ENABLE_PRINTF)
            std::fprintf(stderr, "[FreeRun] failed to open CSV path: %s\n", csv_path);
    }

    const int32_t span = data_stream_->Span();
    const size_t max_w = static_cast<size_t>(span) + 1;
    if (W > max_w)
        W = max_w;

    // Stage 1: teacher-forced edge warmup (last W of local train edge) -> span+1.
    const int32_t wash_start = span - static_cast<int32_t>(W) + 1;
    data_stream_->Seek(wash_start);
    LorenzDatastreamResult st = data_stream_->States();
    for (size_t w = 0; w < W; ++w)
    {
        if (st.sample == nullptr)
            break;
        ExtractDriveReal(drive, *st.sample);
        esn_.ReservoirStep(drive, nullptr);
        st = data_stream_->Step();
    }

    // Stage 2: generative free-run on the input bank (eval runway past span).
    const std::vector<NormalizedState>& S = data_stream_->GetDataStream();
    const double steps_per_lt = 1.0 / (config::LYAPUNOV_EXPONENT * config::DT);

    double sq_err_sum = 0.0;
    size_t steps = 0;
    size_t vpt_steps = 0;
    size_t locked_steps = 0;

    for (size_t j = 0; j < FR; j++)
    {
        const int32_t t = data_stream_->Index();
        if (t < 0 || static_cast<size_t>(t) >= S.size())
        {
            if (config::ENABLE_PRINTF && verbose)
                std::printf("[FreeRun] runway exhausted after %zu steps\n", steps);
            break;
        }

        esn_.Predict(outputs);
        ExtractDrivePredicted(drive, outputs);
        esn_.ReservoirStep(drive, nullptr);

        ExtractTargets(targets, S[static_cast<size_t>(t)]);
        double step_sq = 0.0;
        for (size_t c = 0; c < 3; c++)
        {
            const double e = static_cast<double>(outputs[c]) - targets[c];
            step_sq += e * e;
        }
        sq_err_sum += step_sq;
        ++steps;

        const double step_err = std::sqrt(step_sq / 3.0);
        const bool locked = step_err <= static_cast<double>(config::VPT_THRESHOLD);
        if (locked)
            ++locked_steps;

        if (csv)
        {
            csv << steps << ','
                << (steps / steps_per_lt) << ','
                << step_err << ','
                << (locked ? 1 : 0) << ','
                << outputs[0] << ',' << outputs[1] << ',' << outputs[2] << ','
                << targets[0] << ',' << targets[1] << ',' << targets[2];
            for (size_t d = 0; d < n_drive; ++d)
                csv << ',' << drive[d];
            csv << '\n';
        }

        if (verbose && config::ENABLE_PRINTF)
        {
            if (steps == 1)
            {
                std::printf("%6s %8s %10s %6s  %10s %10s %10s  %10s %10s %10s\n",
                            "step", "lt", "err", "lock",
                            "pred_x", "pred_y", "pred_z",
                            "true_x", "true_y", "true_z");
            }
            std::printf("%6zu %8.4f %10.6f %6d  %10.6f %10.6f %10.6f  %10.6f %10.6f %10.6f\n",
                        steps, steps / steps_per_lt, step_err, locked ? 1 : 0,
                        outputs[0], outputs[1], outputs[2],
                        targets[0], targets[1], targets[2]);
        }

        if (vpt_steps == 0 && step_err > config::VPT_THRESHOLD)
            vpt_steps = steps;

        if (steps == FR)
            break;
        st = data_stream_->Step();
    }

    if (steps == 0)
        return {};

    const bool crossed = vpt_steps > 0;
    const double rmse = std::sqrt(sq_err_sum / (3.0 * steps));
    const double vpt_lt = (crossed ? vpt_steps : steps) / steps_per_lt;
    const double duty = static_cast<double>(locked_steps) / static_cast<double>(steps);
    const double vpt_x_duty = vpt_lt * duty;
    char buf[448];
    if (fixed_ic)
    {
        if (crossed)
            std::snprintf(buf, sizeof buf,
                          "seed %-10llu IC (%.6f,%.6f,%.6f) VPT %3zu steps (%5.2f lt)  "
                          "RMSE %.6f  duty %.3f  VPT*duty %.3f\n",
                          static_cast<unsigned long long>(seed_),
                          fixed_ic->x, fixed_ic->y, fixed_ic->z,
                          vpt_steps, vpt_lt, rmse, duty, vpt_x_duty);
        else
            std::snprintf(buf, sizeof buf,
                          "seed %-10llu IC (%.6f,%.6f,%.6f) VPT >=%3zu steps (%5.2f lt)  "
                          "RMSE %.6f  duty %.3f  VPT*duty %.3f  (never crossed %.2f)\n",
                          static_cast<unsigned long long>(seed_),
                          fixed_ic->x, fixed_ic->y, fixed_ic->z,
                          steps, vpt_lt, rmse, duty, vpt_x_duty, config::VPT_THRESHOLD);
    }
    else if (crossed)
        std::snprintf(buf, sizeof buf,
                      "seed %-10llu orbit_seed %-10llu VPT %3zu steps (%5.2f lt)  "
                      "RMSE %.6f  duty %.3f  VPT*duty %.3f\n",
                      static_cast<unsigned long long>(seed_),
                      static_cast<unsigned long long>(freerun_orbit_seed),
                      vpt_steps, vpt_lt, rmse, duty, vpt_x_duty);
    else
        std::snprintf(buf, sizeof buf,
                      "seed %-10llu orbit_seed %-10llu VPT >=%3zu steps (%5.2f lt)  "
                      "RMSE %.6f  duty %.3f  VPT*duty %.3f  (never crossed %.2f)\n",
                      static_cast<unsigned long long>(seed_),
                      static_cast<unsigned long long>(freerun_orbit_seed),
                      steps, vpt_lt, rmse, duty, vpt_x_duty, config::VPT_THRESHOLD);

    FreeRunResult r;
    r.valid = true;
    r.seed = seed_;
    r.orbit_seed = freerun_orbit_seed;
    r.vpt_steps = vpt_steps;
    r.crossed = crossed;
    r.vpt_lt = vpt_lt;
    r.rmse = rmse;
    r.steps = steps;
    r.duty = duty;
    r.vpt_x_duty = vpt_x_duty;
    r.row = buf;
    if (csv)
        csv.close();
    return r;
}
