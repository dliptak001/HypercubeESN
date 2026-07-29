#include "Lorenz.h"
#include "LorenzDatastream.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <windows.h>


ESNConfig Lorenz::MakeESNConfig(uint64_t seed)
{
    ESNConfig cfg;
    cfg.reservoir.dim = config::DIM;
    cfg.reservoir.seed = seed;

    // Single forward drive on the input bank; external feedback off.
    cfg.reservoir.num_inputs = 4; // [x, y, z, x*z]
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
    // Survey owns outer jthreads — keep each HCNN single-threaded.
    cfg.readout.num_threads = 1;
    cfg.readout.task = ReadoutTask::Regression;
    cfg.readout.activation = ReadoutActivation::TANH;
    return cfg;
}

LorenzDatastreamConfig Lorenz::MakeDatastreamConfig(LorenzAttractor::State orbit)
{
    LorenzDatastreamConfig cfg;
    cfg.stream_length = config::STREAM_LENGTH;
    cfg.cursor_span = config::TRAINING_WINDOW_SIZE;
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
        std::printf("[Lorenz config] ports:     input=%zu [x,y,z,xz]  ext_feedback=0 (off)\n",
                    esn_config_.reservoir.num_inputs);
        std::printf("[Lorenz config] free-run:  teacher force train/washout; self-feedback on input bank\n");
        std::printf("[Lorenz config] readout:   lr %.6f -> %.6f   epochs=%zu\n",
                    config::LEARNING_RATE, config::LEARNING_RATE_MIN, config::EPOCHS);
        std::printf("[Lorenz config] readout in: slices=%zu  pooling=%s\n",
                    config::READOUT_SLICES, config::USE_POOLING ? "on" : "off");
        std::printf("[Lorenz config] stream:    train_span=%d  freerun=%zu  stream_len=%zu  warmup=%zu\n",
                    config::TRAINING_WINDOW_SIZE, config::FREE_RUN_WINDOW_SIZE,
                    config::STREAM_LENGTH, config::RESERVOIR_WARMUP_STEPS);
    }
}

static inline uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

void Lorenz::RebuildDatastream(bool verbose)
{
    orbit_seed_ = mix64(orbit_seed_ ^ (0x100000001B3ULL));
    std::mt19937_64 rng(orbit_seed_);
    std::uniform_real_distribution<double> dist(-0.999, 0.999);
    std::uniform_real_distribution<double> dist_uni(0.0, 0.999);
    LorenzAttractor::State orbit{
        dist(rng), dist(rng), dist_uni(rng)
    };
    data_stream_ = std::make_unique<LorenzDatastream>(MakeDatastreamConfig(orbit));
    if (verbose)
        data_stream_->PrintOrbit();
}

void Lorenz::ExtractDriveReal(float drive[4], const NormalizedState& state)
{
    drive[0] = state.x;
    drive[1] = state.y;
    drive[2] = state.z;
    drive[3] = drive[0] * drive[2];
}

void Lorenz::ExtractDrivePredicted(float drive[4], const float* prediction)
{
    drive[0] = prediction[0];
    drive[1] = prediction[1];
    drive[2] = prediction[2];
    drive[3] = drive[0] * drive[2];
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
    float drive[4] = {};
    float targets[3] = {};
    float outputs[3] = {};

    for (size_t i = 0; i < config::EPOCHS; i++)
    {
        RebuildDatastream(config::ENABLE_PRINTF);

        data_stream_->Reset();
        const float lr = LrProfile(config::LEARNING_RATE, config::LEARNING_RATE_MIN, config::EPOCHS, i);
        LorenzDatastreamResult st = data_stream_->States();

        for (size_t j = 0; j < config::RESERVOIR_WARMUP_STEPS; j++)
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
            // Horizon-1: predict S[t] at x(t) before injecting this step's drive.
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
        else if ((i + 1) % 10 == 0 || i + 1 == config::EPOCHS)
        {
            std::fprintf(stderr, "[seed %llu] train epoch %zu/%zu\n",
                         static_cast<unsigned long long>(seed_),
                         i + 1, config::EPOCHS);
            std::fflush(stderr);
        }
    }
}

FreeRunResult Lorenz::FreeRun(bool verbose, const char* csv_path)
{
    float drive[4] = {};
    float targets[3] = {};
    float outputs[3] = {};

    RebuildDatastream(false);
    const uint64_t freerun_orbit_seed = orbit_seed_;

    std::ofstream csv;
    if (csv_path && csv_path[0])
    {
        csv.open(csv_path, std::ios::out | std::ios::trunc);
        if (csv)
        {
            csv << "step,lt,err,locked,pred_x,pred_y,pred_z,true_x,true_y,true_z,"
                   "drive_x,drive_y,drive_z,drive_xz\n";
        }
        else if (verbose || config::ENABLE_PRINTF)
            std::fprintf(stderr, "[FreeRun] failed to open CSV path: %s\n", csv_path);
    }

    // Stage 1: teacher-forced washout over the training window.
    data_stream_->Reset();
    LorenzDatastreamResult st = data_stream_->States();
    while (!data_stream_->OOB() && st.sample != nullptr)
    {
        ExtractDriveReal(drive, *st.sample);
        esn_.ReservoirStep(drive, nullptr);
        st = data_stream_->Step();
    }

    // Stage 2: generative free-run — prediction fed on the input bank.
    const std::vector<NormalizedState>& S = data_stream_->GetDataStream();
    const double steps_per_lt = 1.0 / (config::LYAPUNOV_EXPONENT * config::DT);

    double sq_err_sum = 0.0;
    size_t steps = 0;
    size_t vpt_steps = 0;
    size_t locked_steps = 0;
    size_t n_relock = 0;
    size_t n_unlock = 0;
    size_t locked_sojourn_sum = 0;
    size_t locked_run_count = 0;
    size_t cur_locked_len = 0;
    bool have_prev = false;
    bool prev_locked = false;
    bool slipped = false;

    for (size_t j = 0; j < config::FREE_RUN_WINDOW_SIZE; j++)
    {
        const int32_t t = data_stream_->Index();
        if (t < 0 || static_cast<size_t>(t) >= S.size())
        {
            if (config::ENABLE_PRINTF && verbose)
                std::printf("[FreeRun] runway exhausted after %zu steps - stream ends\n", steps);
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
                << targets[0] << ',' << targets[1] << ',' << targets[2] << ','
                << drive[0] << ',' << drive[1] << ',' << drive[2] << ',' << drive[3]
                << '\n';
        }

        if (!have_prev)
        {
            have_prev = true;
            prev_locked = locked;
            if (locked)
                cur_locked_len = 1;
        }
        else
        {
            if (locked && !prev_locked)
            {
                if (slipped)
                    ++n_relock;
                cur_locked_len = 1;
            }
            else if (!locked && prev_locked)
            {
                ++n_unlock;
                slipped = true;
                locked_sojourn_sum += cur_locked_len;
                ++locked_run_count;
                cur_locked_len = 0;
            }
            else if (locked)
            {
                ++cur_locked_len;
            }
            prev_locked = locked;
        }

        if (vpt_steps == 0 && step_err > config::VPT_THRESHOLD)
            vpt_steps = steps;
        if (verbose && config::ENABLE_PRINTF && steps % 25 == 0)
            std::printf("free-run %4zu  (%5.2f lt)  err %.6f\n", steps, steps / steps_per_lt, step_err);

        if (steps == config::FREE_RUN_WINDOW_SIZE)
            break;
        st = data_stream_->Step();
    }

    if (steps == 0)
        return {};

    if (prev_locked && cur_locked_len > 0)
    {
        locked_sojourn_sum += cur_locked_len;
        ++locked_run_count;
    }

    const bool crossed = vpt_steps > 0;
    const double rmse = std::sqrt(sq_err_sum / (3.0 * steps));
    const double vpt_lt = (crossed ? vpt_steps : steps) / steps_per_lt;
    const double duty = static_cast<double>(locked_steps) / static_cast<double>(steps);
    const double mean_locked = locked_run_count > 0
        ? static_cast<double>(locked_sojourn_sum) / static_cast<double>(locked_run_count)
        : 0.0;
    char buf[384];
    if (crossed)
        std::snprintf(buf, sizeof buf,
                      "seed %-10llu orbit_seed %-10llu VPT %3zu steps (%5.2f lt)  "
                      "RMSE %.6f  duty %.3f  relock %zu  unlock %zu  meanLock %.1f\n",
                      static_cast<unsigned long long>(seed_), static_cast<unsigned long long>(freerun_orbit_seed),
                      vpt_steps, vpt_lt, rmse, duty, n_relock, n_unlock, mean_locked);
    else
        std::snprintf(buf, sizeof buf,
                      "seed %-10llu orbit_seed %-10llu VPT >=%3zu steps (%5.2f lt)  "
                      "RMSE %.6f  duty %.3f  relock %zu  unlock %zu  meanLock %.1f  (never crossed %.2f)\n",
                      static_cast<unsigned long long>(seed_), static_cast<unsigned long long>(freerun_orbit_seed),
                      steps, vpt_lt, rmse, duty, n_relock, n_unlock, mean_locked, config::VPT_THRESHOLD);

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
    r.n_relock = n_relock;
    r.n_unlock = n_unlock;
    r.mean_locked_sojourn = mean_locked;
    r.row = buf;
    if (csv)
        csv.close();
    return r;
}

static std::string RunTrial(uint64_t esn_seed, uint64_t orbit_seed, int num_runs)
{
    auto progress = [esn_seed](const char* phase, int done, int total)
    {
        std::fprintf(stderr, "[seed %llu] %s %d/%d\n",
                     static_cast<unsigned long long>(esn_seed), phase, done, total);
        std::fflush(stderr);
    };

    progress("start train", 0, static_cast<int>(config::EPOCHS));
    Lorenz lorenz(esn_seed, orbit_seed);
    lorenz.Train();
    progress("train done; start free-runs", 0, num_runs);

    std::vector<FreeRunResult> results;
    results.reserve(num_runs);
    const int prog_every = (num_runs <= 20) ? 1 : std::max(10, num_runs / 20);
    for (int i = 0; i < num_runs; i++)
    {
        results.push_back(lorenz.FreeRun(false));
        if ((i + 1) % prog_every == 0 || i + 1 == num_runs)
            progress("free-run", i + 1, num_runs);
    }

    std::string out;
    char buf[384];
    auto emit = [&](const char* s) { out += s; };

    std::snprintf(buf, sizeof buf, "\n=== ESN seed %llu : %d free-runs (orbit seed %llu) ===\n",
                  static_cast<unsigned long long>(esn_seed), num_runs,
                  static_cast<unsigned long long>(orbit_seed));
    emit(buf);

    std::vector<double> vpt_lts, rmses, duties, relocks, unlocks, mean_locks;
    size_t censored = 0, invalid = 0;
    for (const auto& r : results)
    {
        if (!r.valid)
        {
            ++invalid;
            continue;
        }
        vpt_lts.push_back(r.vpt_lt);
        rmses.push_back(r.rmse);
        duties.push_back(r.duty);
        relocks.push_back(static_cast<double>(r.n_relock));
        unlocks.push_back(static_cast<double>(r.n_unlock));
        mean_locks.push_back(r.mean_locked_sojourn);
        if (!r.crossed) ++censored;
    }

    auto report = [&](const char* label, std::vector<double> v, int prec)
    {
        if (v.empty())
        {
            std::snprintf(buf, sizeof buf, "  %-20s (no valid runs)\n", label);
            emit(buf);
            return;
        }
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        double sum = 0.0;
        for (double x : v) sum += x;
        const double mean = sum / static_cast<double>(n);
        double var = 0.0;
        for (double x : v) var += (x - mean) * (x - mean);
        var = n > 1 ? var / static_cast<double>(n - 1) : 0.0;
        const double sd = std::sqrt(var);
        const double median = n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
        std::snprintf(buf, sizeof buf, "  %-20s n=%2zu  min=%.*f  max=%.*f  mean=%.*f  median=%.*f  std=%.*f\n",
                      label, n, prec, v.front(), prec, v.back(), prec, mean, prec, median, prec, sd);
        emit(buf);
    };

    std::snprintf(buf, sizeof buf, "\n=== Free-run stats (%d runs) ===\n", num_runs);
    emit(buf);
    report("VPT (lt)", vpt_lts, 2);
    report("free-run RMSE", rmses, 6);
    report("duty (<=theta)", duties, 3);
    report("n_relock", relocks, 1);
    report("n_unlock", unlocks, 1);
    report("meanLock (steps)", mean_locks, 1);
    std::snprintf(buf, sizeof buf,
                  "  note: duty/relock/unlock/meanLock use theta=VPT_THRESHOLD=%.2f "
                  "(re-lock proxies; VPT is first upcrossing only)\n",
                  config::VPT_THRESHOLD);
    emit(buf);
    if (censored)
    {
        std::snprintf(buf, sizeof buf,
                      "  note: %zu/%zu run(s) never crossed VPT_THRESHOLD=%.2f; "
                      "their VPT is counted at the window floor (a lower bound)\n",
                      censored, vpt_lts.size(), config::VPT_THRESHOLD);
        emit(buf);
    }
    if (invalid)
    {
        std::snprintf(buf, sizeof buf, "  note: %zu run(s) scored 0 steps and are excluded from the stats\n", invalid);
        emit(buf);
    }

    std::vector<const FreeRunResult*> valid;
    valid.reserve(results.size());
    for (const auto& r : results)
        if (r.valid) valid.push_back(&r);

    const size_t top_n = std::min<size_t>(10, valid.size());

    std::snprintf(buf, sizeof buf, "\n=== Top %zu lowest free-run RMSE ===\n", top_n);
    emit(buf);
    std::sort(valid.begin(), valid.end(),
              [](const FreeRunResult* a, const FreeRunResult* b) { return a->rmse < b->rmse; });
    for (size_t i = 0; i < top_n; i++)
        emit(valid[i]->row.c_str());

    std::snprintf(buf, sizeof buf, "\n=== Top %zu highest VPT (lt) ===\n", top_n);
    emit(buf);
    std::sort(valid.begin(), valid.end(),
              [](const FreeRunResult* a, const FreeRunResult* b) { return a->vpt_lt > b->vpt_lt; });
    for (size_t i = 0; i < top_n; i++)
        emit(valid[i]->row.c_str());

    std::snprintf(buf, sizeof buf, "\n=== Top %zu highest duty (<=theta) ===\n", top_n);
    emit(buf);
    std::sort(valid.begin(), valid.end(),
              [](const FreeRunResult* a, const FreeRunResult* b) { return a->duty > b->duty; });
    for (size_t i = 0; i < top_n; i++)
        emit(valid[i]->row.c_str());

    return out;
}

// Diagnostic: train one seed, free-run, write per-step CSV under examples/Lorenz/traces/.
//   Lorenz.exe --trace <esn_seed> [max_freeruns=30] [target_orbit_seed=0]
static int RunTraceMode(uint64_t esn_seed, int max_freeruns, uint64_t target_orbit)
{
    constexpr uint64_t kBaseOrbit = 72983498;
    std::printf("=== HypercubeESN: Lorenz --trace ===\n");
    std::printf("[trace] esn_seed=%llu  max_freeruns=%d  target_orbit=%llu\n",
                static_cast<unsigned long long>(esn_seed), max_freeruns,
                static_cast<unsigned long long>(target_orbit));
    std::printf("[trace] train %zu epochs then free-run; CSV under examples/Lorenz/traces/\n",
                config::EPOCHS);
    std::fflush(stdout);

    config::ENABLE_PRINTF = true;
    Lorenz lorenz(esn_seed, kBaseOrbit);
    std::cout << lorenz.ReadoutArchSummary();
    lorenz.Train();
    config::ENABLE_PRINTF = false;

    namespace fs = std::filesystem;
    const fs::path trace_dir = fs::path("examples") / "Lorenz" / "traces";
    std::error_code ec;
    fs::create_directories(trace_dir, ec);
    if (ec)
        std::fprintf(stderr, "[trace] create_directories(%s): %s\n",
                     trace_dir.string().c_str(), ec.message().c_str());

    int dumped = 0;
    for (int i = 0; i < max_freeruns; ++i)
    {
        const fs::path tmp_path = trace_dir / ("_tmp_" + std::to_string(esn_seed) + ".csv");
        FreeRunResult r = lorenz.FreeRun(false, tmp_path.string().c_str());
        if (!r.valid)
        {
            std::printf("[trace] free-run %d invalid — stop\n", i);
            break;
        }
        const bool want = (target_orbit == 0) || (r.orbit_seed == target_orbit);
        std::printf("[trace] freerun %d/%d  orbit=%llu  VPT=%.2f lt  duty=%.3f  "
                    "relock=%zu unlock=%zu meanLock=%.1f  %s\n",
                    i + 1, max_freeruns,
                    static_cast<unsigned long long>(r.orbit_seed),
                    r.vpt_lt, r.duty, r.n_relock, r.n_unlock, r.mean_locked_sojourn,
                    want ? "DUMP" : "skip");
        std::fflush(stdout);

        if (want)
        {
            const fs::path out_path = trace_dir /
                ("seed" + std::to_string(esn_seed) + "_orbit" +
                 std::to_string(r.orbit_seed) + ".csv");
            fs::remove(out_path, ec);
            fs::rename(tmp_path, out_path, ec);
            if (ec)
            {
                std::printf("[trace] rename failed (%s) — CSV left at %s\n",
                            ec.message().c_str(), tmp_path.string().c_str());
            }
            else
                std::printf("[trace] wrote %s\n", out_path.string().c_str());
            std::printf("%s", r.row.c_str());
            ++dumped;
            if (target_orbit != 0)
                break;
        }
        else
        {
            fs::remove(tmp_path, ec);
        }
    }
    std::printf("[trace] done — %d CSV file(s)\n", dumped);
    return dumped > 0 ? 0 : 1;
}

int main(int argc, char** argv)
{
    uint64_t seed = 21978990;
    uint64_t orbit_seed = 72983498;

    if (argc >= 3 && std::strcmp(argv[1], "--trace") == 0)
    {
        const uint64_t esn_seed = std::strtoull(argv[2], nullptr, 10);
        int max_fr = 30;
        uint64_t target_orbit = 0;
        if (argc >= 4) max_fr = std::max(1, std::atoi(argv[3]));
        if (argc >= 5) target_orbit = std::strtoull(argv[4], nullptr, 10);
        return RunTraceMode(esn_seed, max_fr, target_orbit);
    }

    std::cout << "=== HypercubeESN: Lorenz ===\n";

    config::ENABLE_PRINTF = false;

    {
        Lorenz probe(seed, orbit_seed);
        std::cout << probe.ReadoutArchSummary();
    }

    // argv[1] = NUM_THREADS (default hardware_concurrency)
    // argv[2] = NUM_RUNS    (default 50 free-runs per trial)
    const size_t hw = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 1;
    const size_t max_threads = 4 * hw;
    size_t num_threads = hw;
    int num_runs = 50;
    if (argc > 1)
    {
        const int arg = std::atoi(argv[1]);
        if (arg > 0) num_threads = std::min<size_t>(max_threads, static_cast<size_t>(arg));
    }
    if (argc > 2)
    {
        const int arg = std::atoi(argv[2]);
        if (arg > 1) num_runs = arg;
    }

    const long long train_steps_per_trial =
        static_cast<long long>(config::EPOCHS) *
        static_cast<long long>(config::TRAINING_WINDOW_SIZE);
    const long long freerun_steps_per_trial =
        static_cast<long long>(num_runs) *
        (static_cast<long long>(config::TRAINING_WINDOW_SIZE) +
         static_cast<long long>(config::FREE_RUN_WINDOW_SIZE));
    std::printf("[survey] free-run (input-bank self-feedback; ext-fb off)\n");
    std::printf("[survey] %zu trial(s) x %d free-run(s)  DIM=%zu N=%zu  epochs=%zu  "
                "train_window=%d  freerun_window=%zu\n",
                num_threads, num_runs, config::DIM, size_t{1} << config::DIM,
                config::EPOCHS, config::TRAINING_WINDOW_SIZE, config::FREE_RUN_WINDOW_SIZE);
    std::printf("[survey] ~%lld train reservoir-steps/trial + ~%lld free-run "
                "reservoir-steps/trial (washout+score); progress on stderr\n",
                train_steps_per_trial, freerun_steps_per_trial);
    if (num_runs >= 500)
        std::printf("[survey] NOTE: NUM_RUNS=%d is a heavy survey — expect multi-hour wall "
                    "clock at this DIM/epoch/window setting\n",
                    num_runs);
    std::fflush(stdout);

    std::vector<std::string> reports(num_threads);
    {
        std::vector<std::jthread> pool;
        pool.reserve(num_threads);
        for (size_t t = 0; t < num_threads; t++)
        {
            pool.emplace_back([&, t]
            {
                const uint64_t esn_seed = seed + t;
                try
                {
                    reports[t] = RunTrial(esn_seed, orbit_seed, num_runs);
                }
                catch (const std::exception& e)
                {
                    char buf[512];
                    std::snprintf(buf, sizeof buf,
                                  "\n=== ESN seed %llu FAILED ===\n  %s\n",
                                  static_cast<unsigned long long>(esn_seed), e.what());
                    reports[t] = buf;
                }
                catch (...)
                {
                    char buf[256];
                    std::snprintf(buf, sizeof buf,
                                  "\n=== ESN seed %llu FAILED ===\n  unknown exception\n",
                                  static_cast<unsigned long long>(esn_seed));
                    reports[t] = buf;
                }
            });
        }
    }

    for (const auto& rep : reports)
        std::fputs(rep.c_str(), stdout);

    Beep(2500, 3000);
    return 0;
}
