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

size_t Lorenz::NumDriveChannels(const DriveLayout layout)
{
    switch (layout)
    {
    case DriveLayout::XyzXz:      return 4;
    case DriveLayout::Quadratic8: return 8;
    }
    return 4;
}

const char* Lorenz::DriveLayoutName(const DriveLayout layout)
{
    switch (layout)
    {
    case DriveLayout::XyzXz:      return "XyzXz";       // [x,y,z,xz]
    case DriveLayout::Quadratic8: return "Quadratic8";  // [x,y,z,xy,xz,xx,yy,zz]
    }
    return "Unknown";
}

ESNConfig Lorenz::MakeESNConfig(uint64_t seed)
{
    ESNConfig cfg;
    cfg.reservoir.dim = config::DIM;
    cfg.reservoir.seed = seed;

    const size_t n_in = NumDriveChannels(config::DRIVE_LAYOUT);
    const size_t N = size_t{1} << config::DIM;
    if (n_in == 0 || N % n_in != 0)
        throw std::invalid_argument(
            "MakeESNConfig: num_inputs must be >= 1 and divide N = 2^DIM");
    cfg.reservoir.num_inputs = n_in;
    cfg.reservoir.num_external_feedback_channels = 0;
    cfg.reservoir.spectral_radius = config::SPECTRAL_RADIUS;
    cfg.reservoir.input_scaling = config::INPUT_SCALING;
    cfg.reservoir.leak_rate = config::LEAK_RATE;
    cfg.reservoir.history_depth = config::HISTORY_DEPTH;
    cfg.reservoir.bias_scaling = 0.01f;

    cfg.readout.num_outputs = 3; // [x, y, z]
    cfg.readout.seed = static_cast<unsigned>(seed);
    cfg.readout_slices = config::READOUT_SLICES;
    cfg.readout.use_pooling = config::USE_POOLING;
    cfg.readout.num_layers = config::NUM_LAYERS;
    cfg.readout.momentum = 0.9f;
    cfg.readout.conv_channels = config::CONV_CHANNELS;
    cfg.readout.num_threads = 1;
    cfg.readout.task = ReadoutTask::Regression;
    cfg.readout.activation = config::READOUT_ACTIVATION;
    return cfg;
}

LorenzDatastreamConfig Lorenz::MakeDatastreamConfig(LorenzAttractor::State orbit)
{
    LorenzDatastreamConfig cfg;
    cfg.stream_length = config::STREAM_LENGTH;
    cfg.span = config::TRAINING_WINDOW_SIZE;
    cfg.initial_lorenz_state = orbit;
    cfg.lorenz_dt = static_cast<float>(config::DT);
    return cfg;
}

const char* Lorenz::ProtocolName(const FreeRunProtocol p)
{
    switch (p)
    {
    case FreeRunProtocol::Unseen:        return "Unseen";
    case FreeRunProtocol::TrainInSample: return "TrainInSample";
    case FreeRunProtocol::TrainHoldout:  return "TrainHoldout";
    }
    return "Unknown";
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
        std::printf("[Lorenz config] ports:     input=%zu drive=%s  ext_feedback=0 (off)\n",
                    esn_config_.reservoir.num_inputs, DriveLayoutName(config::DRIVE_LAYOUT));
        {
            const size_t n_in = esn_config_.reservoir.num_inputs;
            std::printf("[Lorenz config] drive_ch:  [");
            for (size_t i = 0; i < n_in; ++i)
                std::printf("%s%.3f", i ? ", " : "", config::INPUT_SCALE_CH[i]);
            std::printf("]  (× global INPUT_SCALING)\n");
        }
        std::printf("[Lorenz config] free-run:  protocol=%s  warmup=%zu  window=%zu\n",
                    ProtocolName(config::FREE_RUN_PROTOCOL),
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

void Lorenz::FillDrive(float* drive, const float x, const float y, const float z)
{
    // Always write state channels first; products from same-step values (free-run safe).
    drive[0] = x;
    drive[1] = y;
    drive[2] = z;
    switch (config::DRIVE_LAYOUT)
    {
    case DriveLayout::XyzXz:
        // [x, y, z, x*z]
        drive[3] = x * z;
        break;
    case DriveLayout::Quadratic8:
        // [x, y, z, x*y, x*z, x*x, y*y, z*z]
        drive[3] = x * y;
        drive[4] = x * z;
        drive[5] = x * x;
        drive[6] = y * y;
        drive[7] = z * z;
        break;
    }
    // Relative gains on top of reservoir input_scaling (global INPUT_SCALING).
    const size_t n = NumDriveChannels(config::DRIVE_LAYOUT);
    for (size_t i = 0; i < n; ++i)
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
    const float progress = static_cast<float>(current_epoch) / static_cast<float>(epochs - 1);
    constexpr float anneal_fraction = 0.75f;
    return CosineLR(progress / anneal_fraction, lr_max, lr_min);
}

void Lorenz::Train()
{
    float drive[kMaxDriveChannels] = {};
    float targets[3] = {};
    float outputs[3] = {};

    train_orbit_seeds_.clear();
    train_orbit_seeds_.reserve(config::EPOCHS);
    next_train_orbit_pick_ = 0;

    for (size_t i = 0; i < config::EPOCHS; i++)
    {
        RebuildDatastream(config::ENABLE_PRINTF);
        train_orbit_seeds_.push_back(orbit_seed_);

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
            if (train_steps > 0)
                std::printf("epoch %3zu lr %.7f  train RMSE %.6f\n",
                            i, lr, std::sqrt(sq_err_sum / (3.0 * train_steps)));
            else
                std::printf("epoch %3zu lr %.7f  train RMSE n/a  (0 steps - warmup consumed the window)\n",
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
                       "_in" + std::to_string(NumDriveChannels(config::DRIVE_LAYOUT)))).string();
    }

    // Stem only — SaveReadoutHcnnModel appends .hcnw and .arch.json.
    esn_.SaveReadoutHcnnModel(path);
    std::fprintf(stderr, "[Lorenz] saved trained readout: %s.hcnw (+ .arch.json)\n",
                 path.c_str());
    std::fflush(stderr);
}

void Lorenz::SaveTrainedWeightsIfEnabled() const
{
    if (!config::SAVE_TRAINED_WEIGHTS || weights_loaded_)
        return; // no-op after load-only path (never trained this session)
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

void Lorenz::LoadTrainedWeights(const char* stem)
{
    const char* path = (stem && stem[0] != '\0') ? stem : config::LOAD_WEIGHTS_STEM;
    if (path == nullptr || path[0] == '\0')
        throw std::invalid_argument(
            "LoadTrainedWeights: stem empty "
            "(pass a path or set config::LOAD_WEIGHTS_STEM without .hcnw)");

    esn_.LoadReadoutHcnnModel(path);
    train_orbit_seeds_.clear();
    next_train_orbit_pick_ = 0;
    weights_loaded_ = true;

    // Unseen (and fixed IC/orbit freeruns) do not need the train-orbit list.
    // Only warn when config still points at TrainInSample / TrainHoldout.
    if (config::FREE_RUN_PROTOCOL == FreeRunProtocol::Unseen)
    {
        std::fprintf(stderr, "[Lorenz] loaded trained readout: %s.hcnw (+ .arch.json)  "
                             "seed=%llu\n",
                     path, static_cast<unsigned long long>(seed_));
    }
    else
    {
        std::fprintf(stderr, "[Lorenz] loaded trained readout: %s.hcnw (+ .arch.json)  "
                             "seed=%llu  (train-list free-run unavailable until Train(); "
                             "fixed IC/orbit freerun OK; Unseen OK)\n",
                     path, static_cast<unsigned long long>(seed_));
    }
    std::fflush(stderr);
}

FreeRunProtocol Lorenz::EffectiveFreeRunProtocol() const
{
    if (weights_loaded_ && train_orbit_seeds_.empty() &&
        config::FREE_RUN_PROTOCOL != FreeRunProtocol::Unseen)
    {
        std::fprintf(stderr,
                     "[Lorenz] free-run protocol %s unavailable after load-only; using Unseen\n",
                     ProtocolName(config::FREE_RUN_PROTOCOL));
        std::fflush(stderr);
        return FreeRunProtocol::Unseen;
    }
    return config::FREE_RUN_PROTOCOL;
}

FreeRunResult Lorenz::FreeRun(bool verbose, const char* csv_path, size_t warmup_steps,
                              FreeRunProtocol protocol, size_t train_orbit_index,
                              uint64_t fixed_orbit_seed,
                              const LorenzAttractor::State* fixed_ic)
{
    float drive[kMaxDriveChannels] = {};
    float targets[3] = {};
    float outputs[3] = {};
    const size_t n_drive = NumDriveChannels(config::DRIVE_LAYOUT);

    const bool use_train_orbit =
        (protocol == FreeRunProtocol::TrainInSample ||
         protocol == FreeRunProtocol::TrainHoldout);

    uint64_t freerun_orbit_seed = 0;
    if (fixed_ic)
    {
        // Explicit attractor IC; protocol only seats warmup/score window.
        freerun_orbit_seed = 0;
        BuildDatastreamFromIC(*fixed_ic, false);
    }
    else if (fixed_orbit_seed != 0)
    {
        // Explicit orbit seed; protocol only seats warmup/score window.
        freerun_orbit_seed = fixed_orbit_seed;
        BuildDatastreamFromSeed(freerun_orbit_seed, false);
    }
    else if (use_train_orbit)
    {
        if (train_orbit_seeds_.empty())
            throw std::logic_error("FreeRun: TrainInSample/TrainHoldout require Train() first");
        size_t pick = train_orbit_index;
        if (pick == static_cast<size_t>(-1))
            pick = next_train_orbit_pick_++;
        freerun_orbit_seed = train_orbit_seeds_[pick % train_orbit_seeds_.size()];
        BuildDatastreamFromSeed(freerun_orbit_seed, false);
    }
    else
    {
        // Unseen: continue remix chain past all training seeds.
        RebuildDatastream(false);
        freerun_orbit_seed = orbit_seed_;
    }

    std::ofstream csv;
    if (csv_path && csv_path[0])
    {
        csv.open(csv_path, std::ios::out | std::ios::trunc);
        if (csv)
        {
            csv << "step,lt,err,locked,pred_x,pred_y,pred_z,true_x,true_y,true_z";
            if (config::DRIVE_LAYOUT == DriveLayout::Quadratic8)
                csv << ",drive_x,drive_y,drive_z,drive_xy,drive_xz,drive_xx,drive_yy,drive_zz\n";
            else
                csv << ",drive_x,drive_y,drive_z,drive_xz\n";
        }
        else if (verbose || config::ENABLE_PRINTF)
            std::fprintf(stderr, "[FreeRun] failed to open CSV path: %s\n", csv_path);
    }

    const int32_t span = data_stream_->Span();
    const size_t max_w = static_cast<size_t>(span) + 1;
    size_t W = (warmup_steps == 0) ? config::WARMUP_STEPS : warmup_steps;
    if (W < 1) W = 1;
    if (W > max_w) W = max_w;

    // Stage 1: teacher-forced warmup (open-loop drive; config::WARMUP_STEPS).
    // Unseen / TrainHoldout: last W of train (edge) -> leave cursor at span+1.
    // TrainInSample: first W of train -> free-run still inside [0, span].
    const int32_t wash_start = (protocol == FreeRunProtocol::TrainInSample)
                                   ? 0
                                   : (span - static_cast<int32_t>(W) + 1);
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

    // Stage 2: generative free-run on the input bank.
    // Unseen / TrainHoldout: score while index on stream (typically span+1 ...).
    // TrainInSample: score only while index <= span (in-train generative).
    const std::vector<NormalizedState>& S = data_stream_->GetDataStream();
    const double steps_per_lt = 1.0 / (config::LYAPUNOV_EXPONENT * config::DT);
    const bool in_sample = (protocol == FreeRunProtocol::TrainInSample);

    double sq_err_sum = 0.0;
    size_t steps = 0;
    size_t vpt_steps = 0;
    size_t locked_steps = 0;

    for (size_t j = 0; j < config::FREE_RUN_WINDOW_SIZE; j++)
    {
        const int32_t t = data_stream_->Index();
        if (t < 0 || static_cast<size_t>(t) >= S.size())
        {
            if (config::ENABLE_PRINTF && verbose)
                std::printf("[FreeRun] runway exhausted after %zu steps\n", steps);
            break;
        }
        if (in_sample && t > span)
        {
            if (config::ENABLE_PRINTF && verbose)
                std::printf("[FreeRun] left train window after %zu generative steps\n", steps);
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

        if (steps == config::FREE_RUN_WINDOW_SIZE)
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
                          "%-14s seed %-10llu IC (%.6f,%.6f,%.6f) VPT %3zu steps (%5.2f lt)  "
                          "RMSE %.6f  duty %.3f  VPT*duty %.3f\n",
                          ProtocolName(protocol),
                          static_cast<unsigned long long>(seed_),
                          fixed_ic->x, fixed_ic->y, fixed_ic->z,
                          vpt_steps, vpt_lt, rmse, duty, vpt_x_duty);
        else
            std::snprintf(buf, sizeof buf,
                          "%-14s seed %-10llu IC (%.6f,%.6f,%.6f) VPT >=%3zu steps (%5.2f lt)  "
                          "RMSE %.6f  duty %.3f  VPT*duty %.3f  (never crossed %.2f)\n",
                          ProtocolName(protocol),
                          static_cast<unsigned long long>(seed_),
                          fixed_ic->x, fixed_ic->y, fixed_ic->z,
                          steps, vpt_lt, rmse, duty, vpt_x_duty, config::VPT_THRESHOLD);
    }
    else if (crossed)
        std::snprintf(buf, sizeof buf,
                      "%-14s seed %-10llu orbit_seed %-10llu VPT %3zu steps (%5.2f lt)  "
                      "RMSE %.6f  duty %.3f  VPT*duty %.3f\n",
                      ProtocolName(protocol),
                      static_cast<unsigned long long>(seed_),
                      static_cast<unsigned long long>(freerun_orbit_seed),
                      vpt_steps, vpt_lt, rmse, duty, vpt_x_duty);
    else
        std::snprintf(buf, sizeof buf,
                      "%-14s seed %-10llu orbit_seed %-10llu VPT >=%3zu steps (%5.2f lt)  "
                      "RMSE %.6f  duty %.3f  VPT*duty %.3f  (never crossed %.2f)\n",
                      ProtocolName(protocol),
                      static_cast<unsigned long long>(seed_),
                      static_cast<unsigned long long>(freerun_orbit_seed),
                      steps, vpt_lt, rmse, duty, vpt_x_duty, config::VPT_THRESHOLD);

    FreeRunResult r;
    r.valid = true;
    r.seed = seed_;
    r.orbit_seed = freerun_orbit_seed;
    r.protocol = protocol;
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
