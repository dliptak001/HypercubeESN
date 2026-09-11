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

    // Past block on the input port; future block on the dedicated external-feedback
    // port — each [x, y, z, x*z] of its cursor (real past / real-or-predicted future).
    // At FEEDBACK_SCALING=0 the future contributes nothing (past block alone).
    cfg.reservoir.num_inputs = 4; // [x_past, y_past, z_past, x_past*z_past]
    cfg.reservoir.num_external_feedback_channels = 4; // [x_fut,  y_fut,  z_fut,  x_fut*z_fut ]
    cfg.reservoir.external_feedback_scaling = config::FEEDBACK_SCALING;
    cfg.reservoir.spectral_radius = config::SPECTRAL_RADIUS;
    cfg.reservoir.input_scaling = config::INPUT_SCALING;
    cfg.reservoir.leak_rate = config::LEAK_RATE;
    cfg.reservoir.history_depth = config::HISTORY_DEPTH;
    cfg.reservoir.bias_scaling = 0.01;

    cfg.readout.num_outputs = 3; //[x, y, z]
    cfg.readout.seed = seed;
    cfg.readout_slices = config::READOUT_SLICES;
    cfg.readout.use_pooling = config::USE_POOLING;
    cfg.readout.num_layers = config::NUM_LAYERS;
    cfg.readout.momentum = 0.9;
    cfg.readout.conv_channels = config::CONV_CHANNELS;
    // One level of parallelism: the seed survey owns outer jthreads. Keep each
    // HCNN single-threaded so we do not spawn (hw−1) idle workers per trial.
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
    cfg.cursor_center_index = config::CURSOR_CENTER_INDEX;
    cfg.initial_lorenz_state = orbit; // by value — no non-owning pointer into caller storage
    cfg.lorenz_dt = config::DT;
    return cfg;
}

Lorenz::Lorenz(const uint64_t seed, uint64_t orbit_seed) : seed_(seed),
    orbit_seed_(orbit_seed),
    esn_config_(MakeESNConfig(seed_)),
    esn_(esn_config_)
{
    if (config::ENABLE_PRINTF)
    {
        std::printf("[Lorenz config] reservoir: DIM=%zu (N=%zu)  seed=%llu  SR=%.3f  input_scaling=%.3f  leak=%.2f"
                    "  history_depth=%zu\n",
                    config::DIM, size_t{1} << config::DIM, static_cast<unsigned long long>(seed_),
                    config::SPECTRAL_RADIUS, config::INPUT_SCALING, config::LEAK_RATE,
                    config::HISTORY_DEPTH);
        std::printf("[Lorenz config] ports:     input=%zu [past x,y,z,xz]  ext_feedback=%zu [future x,y,z,xz]"
                    "  external_feedback_scaling=%.4f\n",
                    esn_config_.reservoir.num_inputs, esn_config_.reservoir.num_external_feedback_channels,
                    config::FEEDBACK_SCALING);
        std::printf("[Lorenz config] arm:       %s\n",
                    config::FORWARD_ONLY
                        ? "forward-only (past=0 every ReservoirStep; train+washout+free-run)"
                        : "Janus (real past on input + future on ext-fb)");
        std::printf("[Lorenz config] readout:   lr %.6f -> %.6f   epochs=%zu\n",
                    config::LEARNING_RATE, config::LEARNING_RATE_MIN, config::EPOCHS);
        std::printf("[Lorenz config] readout in: slices=%zu  pooling=%s\n",
                    config::READOUT_SLICES, config::USE_POOLING ? "on" : "off");
        std::printf("[Lorenz config] stream:    x0=(%.2f, %.2f, %.2f)  warmup=%zu\n",
                    config::INITIAL_LORENZ_STATE.x, config::INITIAL_LORENZ_STATE.y, config::INITIAL_LORENZ_STATE.z,
                    config::RESERVOIR_WARMUP_STEPS);
    }
    // data_stream_ is left null here: Train() and FreeRun() each RebuildDatastream()
    // (fresh orbit) before first use, so there is nothing to allocate at construction.
}

// SplitMix64 finalizer. Avalanches a 64-bit value so that substreams labelled
// off one master seed are statistically independent (one input bit flips ~half
// the output bits). Replaces the old folklore of forking mt19937 by small
// additive offsets (seed + 0x9E3779B9, seed + 12345), which gave no such
// guarantee, and the separate bias_seed config field it papered over.
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
    data_stream_ = std::make_unique<LorenzDatastream>(MakeDatastreamConfig(orbit)); // frees the previous stream
    if (verbose)
        data_stream_->PrintOrbit();
}

void Lorenz::ExtractPast(float past[4], const LorenzDatastreamResult& past_future_states)
{
    past[0] = past_future_states.past.x; // past x
    past[1] = past_future_states.past.y; // past y
    past[2] = past_future_states.past.z; // past z
    past[3] = past[0] * past[2]; // past x*z (nonlinear term)
}

void Lorenz::FillPast(float past[4], const LorenzDatastreamResult& past_future_states)
{
    if (config::FORWARD_ONLY)
    {
        past[0] = past[1] = past[2] = past[3] = 0.f;
        return;
    }
    ExtractPast(past, past_future_states);
}

void Lorenz::ExtractFutureReal(float future[4], const LorenzDatastreamResult& past_future_states)
{
    future[0] = past_future_states.future->x; // future x (teacher-forced)
    future[1] = past_future_states.future->y; // future y
    future[2] = past_future_states.future->z; // future z
    future[3] = future[0] * future[2]; // future x*z (nonlinear term)
}

void Lorenz::ExtractFuturePredicted(float future[4], const float* prediction)
{
    future[0] = prediction[0]; // future x = the model's own prediction
    future[1] = prediction[1]; // future y
    future[2] = prediction[2]; // future z
    future[3] = future[0] * future[2]; // future x*z, from the predicted x,z
}

void Lorenz::ExtractTargets(float targets[3], const NormalizedState& future_state)
{
    targets[0] = future_state.x;
    targets[1] = future_state.y;
    targets[2] = future_state.z;
}

float Lorenz::LrProfile(const float lr_max, const float lr_min, const size_t epochs, const size_t current_epoch)
{
    if (epochs <= 1)
        return lr_max;
    const float progress = static_cast<float>(current_epoch) / static_cast<float>(epochs - 1);
    // Reach lr_min at 75% of the run, then hold flat for the last 25%: stretch the
    // [0, 0.75] progress window onto the full cosine [0, 1]. CosineLR clamps its
    // argument at 1, so every epoch past the 75% mark stays pinned at lr_min.
    constexpr float anneal_fraction = 0.75f;
    return CosineLR(progress / anneal_fraction, lr_max, lr_min); // swap in ExponentialDecayLR (Readout.h) to compare
    //return ExponentialDecayLR(progress / anneal_fraction, lr_max, lr_min);
}

void Lorenz::Train()
{
    float past[4] = {}; // input-port drive: [past x, y, z, x*z]
    float future[4] = {}; // feedback-port drive: [future x, y, z, x*z] (teacher-forced real)
    float targets[3] = {};
    float outputs[3] = {}; // the model's pre-update prediction (prequential read)

    for (size_t i = 0; i < config::EPOCHS; i++)
    {
        RebuildDatastream(config::ENABLE_PRINTF); // orbit print is a per-epoch diagnostic — silenced in the concurrent survey

        // Step 1: warm up the reservoir open-loop, teacher-forced, no readout update.
        data_stream_->Reset();
        const float lr = LrProfile(config::LEARNING_RATE, config::LEARNING_RATE_MIN, config::EPOCHS, i);
        LorenzDatastreamResult past_future_states = data_stream_->States();
        for (size_t j = 0; j < config::RESERVOIR_WARMUP_STEPS; j++)
        {
            // Stop if the future cursor has left the training window — ExtractFutureReal
            // requires a non-null teacher sample. Happens only if warmup >= remaining span.
            if (data_stream_->OOB())
                break;
            FillPast(past, past_future_states);
            ExtractFutureReal(future, past_future_states);
            esn_.ReservoirStep(past, future); // teacher-forced: past->input, real future->feedback
            past_future_states = data_stream_->Step();
        }

        // Step 2: Train - train towards future state targets
        double sq_err_sum = 0.0; // double accumulator: ~15K float-sized terms/epoch
        size_t train_steps = 0;
        while (!data_stream_->OOB())
        {
            // Horizon-1 alignment: predict at x(t) BEFORE injecting this call's
            // inputs, so x(t) has seen the future channel only through S[f-1]. The
            // aligned one-step target is S[f] — the sample this call is about to
            // inject — not S[f+1], one sample further on.
            ExtractTargets(targets, *past_future_states.future);
            FillPast(past, past_future_states);
            ExtractFutureReal(future, past_future_states);

            // Prequential (test-then-train) read: the pre-update prediction of this
            // call's own target, read at the unchanged state x(t) before TrainStep.
            esn_.Predict(outputs);
            esn_.TrainStep(targets, lr); // fit the readout on x(t)
            esn_.ReservoirStep(past, future); // step to x(t+1)

            // Prequential (test-then-train) error: `outputs` is the pre-update
            // prediction of this call's own target — the pairing is exact.
            for (size_t c = 0; c < 3; c++)
            {
                const double e = static_cast<double>(outputs[c]) - targets[c];
                sq_err_sum += e * e;
            }
            ++train_steps;

            past_future_states = data_stream_->Step();
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
            // Survey mode: coarse heartbeat on stderr (no per-step spam).
            std::fprintf(stderr, "[seed %llu] train epoch %zu/%zu\n",
                         static_cast<unsigned long long>(seed_),
                         i + 1, config::EPOCHS);
            std::fflush(stderr);
        }
    }
}

FreeRunResult Lorenz::FreeRun(bool verbose, const char* csv_path)
{
    float past[4] = {}; // input-port drive: [past x, y, z, x*z]
    float future[4] = {}; // feedback-port drive: [x, y, z, x*z] of the model's prediction
    float targets[3] = {};
    float outputs[3] = {}; // the model's generative prediction each step

    RebuildDatastream(false);
    const uint64_t freerun_orbit_seed = orbit_seed_; // post-mix value used for this free-run

    std::ofstream csv;
    if (csv_path && csv_path[0])
    {
        csv.open(csv_path, std::ios::out | std::ios::trunc);
        if (csv)
        {
            csv << "step,lt,err,locked,pred_x,pred_y,pred_z,true_x,true_y,true_z,"
                   "past_x,past_y,past_z,past_xz\n";
        }
        else if (verbose || config::ENABLE_PRINTF)
            std::fprintf(stderr, "[FreeRun] failed to open CSV path: %s\n", csv_path);
    }

    // Stage 1: anchored washout. Re-sweep the training window once, teacher-forced
    // but inference-only (no readout updates), so the reservoir crosses into the
    // generative region warm and in-distribution.
    data_stream_->Reset();
    LorenzDatastreamResult past_future_states = data_stream_->States();
    while (!data_stream_->OOB())
    {
        FillPast(past, past_future_states);
        ExtractFutureReal(future, past_future_states);
        esn_.ReservoirStep(past, future); // teacher-forced anchored washout
        past_future_states = data_stream_->Step();
    }

    // Stage 2: generative rollout, single-ESN self-feedback closed loop. The future
    // cursor is now one step past the window edge: the future block on the feedback
    // port switches to the model's OWN prediction, so the reservoir free-runs while
    // the past cursor stays anchored to real history. Predict() reads the prediction
    // at the current state, ExtractFuturePredicted feeds it onto the feedback port,
    // then ReservoirStep absorbs it; the prediction is scored against the true orbit.
    const std::vector<NormalizedState>& S = data_stream_->GetDataStream();
    const double steps_per_lt = 1.0 / (config::LYAPUNOV_EXPONENT * config::DT);
    /*if (config::ENABLE_PRINTF)
    {
        std::printf("[FreeRun] generative: %zu steps (%.1f Lyapunov times)  vpt_threshold %.2f\n",
                    config::FREE_RUN_WINDOW_SIZE, config::FREE_RUN_WINDOW_SIZE / steps_per_lt,
                    config::VPT_THRESHOLD);
        std::printf("[FreeRun] drive: self-feedback (own prediction fed on the future channels)\n");
    }*/

    double sq_err_sum = 0.0;
    size_t steps = 0;
    size_t vpt_steps = 0; // first step whose error exceeded VPT_THRESHOLD (0 = never)
    size_t locked_steps = 0; // steps with channel-RMS ≤ θ (GS duty numerator; on-track)
    size_t n_relock = 0; // unlocked → locked *after* at least one unlock (true re-locks)
    size_t n_unlock = 0; // locked → unlocked transitions
    size_t locked_sojourn_sum = 0; // sum of completed + trailing locked run lengths
    size_t locked_run_count = 0; // number of locked sojourns (for mean)
    size_t cur_locked_len = 0;
    bool have_prev = false;
    bool prev_locked = false;
    bool slipped = false; // true after first unlock; next lock is a re-lock
    for (size_t j = 0; j < config::FREE_RUN_WINDOW_SIZE; j++)
    {
        const int32_t f = data_stream_->Indices().second; // this step's held-out truth index
        if (f < 0 || static_cast<size_t>(f) >= S.size())
        {
            if (config::ENABLE_PRINTF && verbose)
                std::printf("[FreeRun] runway exhausted after %zu steps - stream ends\n", steps);
            break;
        }

        esn_.Predict(outputs); // the prediction of S[f]
        FillPast(past, past_future_states); // Janus: anchored past; FORWARD_ONLY: zeros
        ExtractFuturePredicted(future, outputs); // own prediction -> feedback port
        esn_.ReservoirStep(past, future); // absorb the fed-back prediction

        // Score against the true held-out orbit (normalized units).
        ExtractTargets(targets, S[f]);
        double step_sq = 0.0;
        for (size_t c = 0; c < 3; c++)
        {
            const double e = static_cast<double>(outputs[c]) - targets[c];
            step_sq += e * e;
        }
        sq_err_sum += step_sq;
        ++steps;

        const double step_err = std::sqrt(step_sq / 3.0);
        // Match VPT's strict ">" upcrossing: locked ⇔ err ≤ θ  (VPT fires when err > θ).
        const bool locked = step_err <= static_cast<double>(config::VPT_THRESHOLD);
        if (locked)
            ++locked_steps;

        if (csv)
        {
            // past[] is the input-port drive this step (zeros under FORWARD_ONLY).
            csv << steps << ','
                << (steps / steps_per_lt) << ','
                << step_err << ','
                << (locked ? 1 : 0) << ','
                << outputs[0] << ',' << outputs[1] << ',' << outputs[2] << ','
                << targets[0] << ',' << targets[1] << ',' << targets[2] << ','
                << past[0] << ',' << past[1] << ',' << past[2] << ',' << past[3]
                << '\n';
        }

        // GS proxies: duty from locked fraction; re-lock only after a prior unlock.
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
                    ++n_relock; // unlocked → locked after a slip (GS recovery)
                cur_locked_len = 1;
            }
            else if (!locked && prev_locked)
            {
                ++n_unlock; // locked → unlocked
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
            break; // done - don't step the cursors past the last scored index
        if (data_stream_->Indices().first <= 0)
        {
            if (verbose && config::ENABLE_PRINTF)
                std::printf("[FreeRun] anchor runway exhausted after %zu steps - past cursor at the seed\n", steps);
            break;
        }
        past_future_states = data_stream_->Step();
    }

    if (steps == 0)
        return {}; // valid == false — excluded from the survey stats

    // Close a trailing locked sojourn so mean locked length includes the final run.
    if (prev_locked && cur_locked_len > 0)
    {
        locked_sojourn_sum += cur_locked_len;
        ++locked_run_count;
    }

    // Per-seed outcome + display row (always built, independent of ENABLE_PRINTF).
    // VPT = first upcrossing of θ. duty / n_relock capture GS-style re-lock that
    // VPT alone misses (error can recover after the first crossing).
    const bool crossed = vpt_steps > 0;
    const double rmse = std::sqrt(sq_err_sum / (3.0 * steps));
    const double vpt_lt = (crossed ? vpt_steps : steps) / steps_per_lt;
    const double duty = static_cast<double>(locked_steps) / static_cast<double>(steps);
    const double mean_locked = locked_run_count > 0
        ? static_cast<double>(locked_sojourn_sum) / static_cast<double>(locked_run_count)
        : 0.0;
    char buf[384];
    const char* arm = config::FORWARD_ONLY ? "fwd-only" : "Janus";
    if (crossed)
        std::snprintf(buf, sizeof buf,
                      "arm %-8s seed %-10llu orbit_seed %-10llu VPT %3zu steps (%5.2f lt)  "
                      "RMSE %.6f  duty %.3f  relock %zu  unlock %zu  meanLock %.1f\n",
                      arm, static_cast<unsigned long long>(seed_), static_cast<unsigned long long>(orbit_seed_),
                      vpt_steps, vpt_lt, rmse, duty, n_relock, n_unlock, mean_locked);
    else
        std::snprintf(buf, sizeof buf,
                      "arm %-8s seed %-10llu orbit_seed %-10llu VPT >=%3zu steps (%5.2f lt)  "
                      "RMSE %.6f  duty %.3f  relock %zu  unlock %zu  meanLock %.1f  (never crossed %.2f)\n",
                      arm, static_cast<unsigned long long>(seed_), static_cast<unsigned long long>(orbit_seed_),
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

// One trial = train a single ESN on esn_seed, then free-run it num_runs times.
// RebuildDatastream re-mixes orbit_seed_ on every FreeRun call, so two trials
// started from the same orbit_seed are scored against the SAME sequence of
// held-out orbits — the ESN seed is the only independent variable. The report
// (aggregate stats + top-10 leaderboards; individual runs are not listed) is built
// into a string and returned so trials running in parallel can be printed serially
// without their output interleaving.
static std::string RunTrial(uint64_t esn_seed, uint64_t orbit_seed, int num_runs)
{
    // Survey progress goes to stderr so it does not interleave with the final
    // report strings on stdout. ENABLE_PRINTF stays false for per-step noise.
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

    std::vector<FreeRunResult> results; // one outcome per free-run (RebuildDatastream re-mixes the orbit each call)
    results.reserve(num_runs);
    // Progress cadence: every free-run for small surveys, else every ~5% (min 10).
    const int prog_every = (num_runs <= 20) ? 1 : std::max(10, num_runs / 20);
    for (int i = 0; i < num_runs; i++)
    {
        results.push_back(lorenz.FreeRun(false));
        if ((i + 1) % prog_every == 0 || i + 1 == num_runs)
            progress("free-run", i + 1, num_runs);
    }

    std::string out;
    char buf[384]; // long enough for FreeRunResult::row + aggregate note lines
    auto emit = [&](const char* s) { out += s; };

    std::snprintf(buf, sizeof buf, "\n=== ESN seed %llu : %d free-runs (orbit seed %llu) ===\n",
                  static_cast<unsigned long long>(esn_seed), num_runs,
                  static_cast<unsigned long long>(orbit_seed));
    emit(buf);

    // Aggregate stats over the valid runs. VPT is in Lyapunov times; runs that
    // never crossed VPT_THRESHOLD contribute their window floor (a lower bound),
    // so the VPT stats are conservative when any run is censored (noted below).
    // duty / n_relock are GS re-lock proxies (same θ as VPT).
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
        var = n > 1 ? var / static_cast<double>(n - 1) : 0.0; // sample variance
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
    // ASCII labels only — Windows consoles often mis-decode UTF-8 Greek theta.
    report("duty (<=theta)", duties, 3);
    report("n_relock", relocks, 1);
    report("n_unlock", unlocks, 1);
    report("meanLock (steps)", mean_locks, 1);
    std::snprintf(buf, sizeof buf,
                  "  note: duty/relock/unlock/meanLock use theta=VPT_THRESHOLD=%.2f (GS re-lock proxies; "
                  "VPT is first upcrossing only)\n",
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

    // Summary tables. Rank the valid runs and print leaderboards; each row is the
    // run's own display line (seed / orbit_seed / VPT / RMSE / duty / relock).
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

// Diagnostic mode: train one ESN seed, free-run until max_runs (or until target
// orbit_seed is hit), write per-step CSV under traces/.
//   Lorenz.exe --trace <esn_seed> [max_freeruns=30] [target_orbit_seed=0]
// target_orbit_seed 0 = dump every free-run; else dump only matching orbits and stop.
static int RunTraceMode(uint64_t esn_seed, int max_freeruns, uint64_t target_orbit)
{
    constexpr uint64_t kBaseOrbit = 72983498;
    std::printf("=== HypercubeESN: Lorenz --trace ===\n");
    std::printf("[trace] arm=%s  esn_seed=%llu  max_freeruns=%d  target_orbit=%llu\n",
                config::FORWARD_ONLY ? "fwd-only" : "Janus",
                static_cast<unsigned long long>(esn_seed), max_freeruns,
                static_cast<unsigned long long>(target_orbit));
    std::printf("[trace] train %zu epochs then free-run; CSV under "
                "\"Research Topics/Lorenz_JanusCursor/traces/\"\n",
                config::EPOCHS);
    std::fflush(stdout);

    config::ENABLE_PRINTF = true; // epoch lines while training
    Lorenz lorenz(esn_seed, kBaseOrbit);
    std::cout << lorenz.ReadoutArchSummary();
    lorenz.Train();
    config::ENABLE_PRINTF = false;

    namespace fs = std::filesystem;
    const fs::path trace_dir = fs::path("Research Topics") / "Lorenz_JanusCursor" / "traces";
    std::error_code ec;
    fs::create_directories(trace_dir, ec);
    if (ec)
        std::fprintf(stderr, "[trace] create_directories(%s): %s\n",
                     trace_dir.string().c_str(), ec.message().c_str());

    int dumped = 0;
    for (int i = 0; i < max_freeruns; ++i)
    {
        // Write to a temp path first; rename only if we keep the free-run.
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
                break; // found requested orbit
        }
        else
        {
            fs::remove(tmp_path, ec);
        }
    }
    std::printf("[trace] done — %d CSV file(s) under "
                "\"Research Topics/Lorenz_JanusCursor/traces/\"\n",
                dumped);
    return dumped > 0 ? 0 : 1;
}

int main(int argc, char** argv)
{
    uint64_t seed = 21978990;
    uint64_t orbit_seed = 72983498;

    // Diagnostic path (not the multi-thread survey).
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

    // One trial per thread: each trains its own ESN on a distinct seed and free-runs
    // it, but all share the same starting orbit_seed, so every trial is scored against
    // an identical sequence of held-out orbits (the ESN seed is the only independent
    // variable). Per-run diagnostics are silenced so the trial reports — collected as
    // strings and printed serially below — don't interleave.
    config::ENABLE_PRINTF = false;

    // Architecture is identical for every survey seed — print once before the pool.
    {
        Lorenz probe(seed, orbit_seed);
        std::cout << probe.ReadoutArchSummary();
    }

    // Optional positional CLI overrides (applied only when present and in range):
    //   argv[1] = NUM_THREADS — parallel trials, one ESN seed each (default = hardware_concurrency)
    //   argv[2] = NUM_RUNS    — free-runs accumulated per trial      (default = 50)
    // WARNING: each free-run rebuilds an orbit and re-washes the full training
    // window (~TRAINING_WINDOW_SIZE reservoir steps) before FREE_RUN_WINDOW_SIZE
    // generative steps. 4 x 2000 is many hours of wall clock at DIM 11 / 100 epochs.
    const size_t hw = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 1;
    const size_t max_threads = 4 * hw; // guard against a fat-fingered arg spawning a thread stampede
    size_t num_threads = hw;
    int num_runs = 50; // keep CLI default light; pass e.g. 2000 only for full surveys
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

    // Work estimate (order-of-magnitude). Free-run dominates when NUM_RUNS is large:
    // each FreeRun washes ~TRAINING_WINDOW_SIZE steps then scores FREE_RUN_WINDOW_SIZE.
    const long long train_steps_per_trial =
        static_cast<long long>(config::EPOCHS) *
        static_cast<long long>(config::TRAINING_WINDOW_SIZE);
    const long long freerun_steps_per_trial =
        static_cast<long long>(num_runs) *
        (static_cast<long long>(config::TRAINING_WINDOW_SIZE) +
         static_cast<long long>(config::FREE_RUN_WINDOW_SIZE));
    std::printf("[survey] arm=%s\n",
                config::FORWARD_ONLY
                    ? "forward-only (past=0 every step; train+washout+free-run)"
                    : "Janus (real past + future ext-fb)");
    std::printf("[survey] %zu trial(s) x %d free-run(s)  DIM=%zu N=%zu  epochs=%zu  "
                "train_window=%d  freerun_window=%zu\n",
                num_threads, num_runs, config::DIM, size_t{1} << config::DIM,
                config::EPOCHS, config::TRAINING_WINDOW_SIZE, config::FREE_RUN_WINDOW_SIZE);
    std::printf("[survey] ~%lld train reservoir-steps/trial + ~%lld free-run "
                "reservoir-steps/trial (washout+score); no stdout until all trials finish "
                "(progress on stderr)\n",
                train_steps_per_trial, freerun_steps_per_trial);
    if (num_runs >= 500)
        std::printf("[survey] NOTE: NUM_RUNS=%d is a heavy survey — expect multi-hour wall "
                    "clock at this DIM/epoch/window setting\n",
                    num_runs);
    std::fflush(stdout);

    // Each trial owns its own Lorenz/ESN/datastream (HCNN forced single-threaded
    // in MakeESNConfig). No shared Lorenz mutable state — each thread writes only
    // reports[t]. Catch exceptions so one failed trial does not std::terminate.
    std::vector<std::string> reports(num_threads); // one report per trial (= per thread)
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
    } // jthreads join on scope exit

    // Print the trial reports in seed order (independent of completion order).
    for (const auto& rep : reports)
        std::fputs(rep.c_str(), stdout);

    Beep(2500, 3000); // single completion beep for the whole survey
    return 0;
}
