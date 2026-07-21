#include "Lorenz.h"
#include "LorenzDatastream.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    cfg.reservoir.bias_scaling = 0.0;
    cfg.reservoir.history_floor = 1.0;

    // FSF A/B: set config::FULL_STATE_FEEDBACK in Lorenz.h (independent of external fb).
    cfg.reservoir.full_state_feedback = config::FULL_STATE_FEEDBACK;
    cfg.reservoir.fsf_seed = config::FSF_SEED;
    cfg.reservoir.fsf_scaling = config::FSF_SCALING;
    cfg.reservoir.fsf_v_scaling = config::FSF_V_SCALING;

    cfg.readout.num_outputs = 3; //[x, y, z]
    cfg.readout.seed = static_cast<unsigned>(seed);

    // Block-structured readout input: READOUT_SLICES delay-line slices plus an optional
    // aux block holding the normalized past (x,y,z). Both feed only the readout; the
    // reservoir is unchanged. The stream is already [-1,1], so u_raw needs no scaling.
    cfg.readout_slices = config::READOUT_SLICES;
    cfg.aux_input_dim = config::AUX_INPUT_DIM;
    cfg.readout.use_pooling = config::USE_POOLING;
    cfg.readout.num_layers = 1; // TODO num_layers = 2 with no pooling.
    cfg.readout.momentum = 0.9;
    cfg.readout.conv_channels = 8;
    // One level of parallelism: the seed survey owns outer jthreads. Keep each
    // HCNN single-threaded so we do not spawn (hw−1) idle workers per trial.
    cfg.readout.num_threads = 1;
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
        std::printf("[Lorenz config] FSF A/B:   %s  fsf_seed=%llu  fsf_scaling=%.3f  fsf_v_scaling=%.3f\n",
                    config::FULL_STATE_FEEDBACK ? "ON " : "OFF",
                    static_cast<unsigned long long>(config::FSF_SEED),
                    config::FSF_SCALING, config::FSF_V_SCALING);
        std::printf("[Lorenz config] readout:   lr %.6f -> %.6f   epochs=%zu\n",
                    config::LEARNING_RATE, config::LEARNING_RATE_MIN, config::EPOCHS);
        std::printf("[Lorenz config] readout in: slices=%zu  aux=%zu  blocks=%zu  pooling=%s\n",
                    config::READOUT_SLICES, config::AUX_INPUT_DIM,
                    config::READOUT_SLICES + (config::AUX_INPUT_DIM > 0 ? 1u : 0u),
                    config::USE_POOLING ? "on" : "off");
        std::printf("[Lorenz config] exposure:  2a future_noise=%.4f  2b ss_ceiling=%.3f\n",
                    config::TRAIN_FUTURE_NOISE, config::SCHEDULED_SAMPLING_CEILING);
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

static_assert(config::AUX_INPUT_DIM == 0 || config::AUX_INPUT_DIM == 3,
              "ExtractAuxPast fills exactly 3 components (past x,y,z)");

void Lorenz::ExtractAuxPast(float u_raw[3], const LorenzDatastreamResult& past_future_states)
{
    u_raw[0] = past_future_states.past.x; // normalized past x
    u_raw[1] = past_future_states.past.y; // normalized past y
    u_raw[2] = past_future_states.past.z; // normalized past z
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

float Lorenz::ScheduledSamplingProfile(const float ceiling, const size_t epochs, const size_t current_epoch)
{
    if (epochs <= 1)
        return ceiling;
    const float progress = static_cast<float>(current_epoch) / static_cast<float>(epochs - 1);
    return ceiling * progress; // linear ramp 0 -> ceiling
}

void Lorenz::Train()
{
    float past[4] = {}; // input-port drive: [past x, y, z, x*z]
    float future[4] = {}; // feedback-port drive: [future x, y, z, x*z] (real; prediction under 2b)
    float targets[3] = {};
    float outputs[3] = {}; // the model's pre-update prediction (prequential read)
    float u_past[3] = {}; // auxiliary readout input: normalized past (x,y,z)
    // nullptr when the readout has no aux block — ESN rejects a stray u_raw, and an aux
    // block is never silently zeroed. `aux` aliases u_past, so it tracks each refill.
    const float* const aux = (config::AUX_INPUT_DIM > 0) ? u_past : nullptr;

    // Exposure-bias remedies (README Issue 2) share one RNG stream. The guards on
    // the config scalars below keep the baseline path bit-identical (no draws) when
    // both remedies are off, so toggling them is a clean single-delta change.
    std::mt19937_64 exposure_rng(config::TRAIN_EXPOSURE_RNG_SEED);
    // stddev must be > 0 for the distribution (libstdc++ asserts it); when 2a is
    // off we pass a dummy 1.0 that is never sampled (the draw below is guarded).
    std::normal_distribution<float> future_noise(
        0.0f, config::TRAIN_FUTURE_NOISE > 0.0f ? config::TRAIN_FUTURE_NOISE : 1.0f);
    std::uniform_real_distribution<float> unit_uniform(0.0f, 1.0f);

    for (size_t i = 0; i < config::EPOCHS; i++)
    {
        RebuildDatastream(config::ENABLE_PRINTF); // orbit print is a per-epoch diagnostic — silenced in the concurrent survey

        // Step 1: warm up the reservoir open-loop, teacher-forced, no readout update.
        data_stream_->Reset();
        const float lr = LrProfile(config::LEARNING_RATE, config::LEARNING_RATE_MIN, config::EPOCHS, i);
        // 2b: this epoch's probability of substituting the model's own prediction
        // on the future channels (0 when the ceiling is 0, i.e. 2b disabled).
        const float ss_p = ScheduledSamplingProfile(config::SCHEDULED_SAMPLING_CEILING, config::EPOCHS, i);
        LorenzDatastreamResult past_future_states = data_stream_->States();
        for (size_t j = 0; j < config::RESERVOIR_WARMUP_STEPS; j++)
        {
            // Stop if the future cursor has left the training window — ExtractFutureReal
            // requires a non-null teacher sample. Happens only if warmup >= remaining span.
            if (data_stream_->OOB())
                break;
            ExtractPast(past, past_future_states);
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
            ExtractPast(past, past_future_states);
            ExtractFutureReal(future, past_future_states);
            ExtractAuxPast(u_past, past_future_states);

            // Prequential (test-then-train) read: the pre-update prediction of this
            // call's own target, read at the unchanged state x(t) before TrainStep.
            esn_.Predict(outputs, aux);

            // Exposure-bias remedies (README Issue 2), future block only; the teacher
            // target above stays the real S[f]. Both perturb the future LINEAR
            // channels; the future x*z is re-derived afterward so the whole block
            // stays consistent with free-run (where it is the predicted x*z).
            //
            // 2b scheduled sampling: with probability ss_p, overwrite the future
            // linear channels with the model's own fresh prediction of S[f] — which
            // is exactly `outputs`, read at the same unchanged state x(t).
            if (config::SCHEDULED_SAMPLING_CEILING > 0.0f && unit_uniform(exposure_rng) < ss_p)
            {
                future[0] = outputs[0];
                future[1] = outputs[1];
                future[2] = outputs[2];
            }
            // 2a noise injection: zero-mean Gaussian on the (possibly substituted)
            // future linear channels — trains the map to tolerate the perturbation
            // class the closed loop introduces.
            if (config::TRAIN_FUTURE_NOISE > 0.0f)
            {
                future[0] += future_noise(exposure_rng);
                future[1] += future_noise(exposure_rng);
                future[2] += future_noise(exposure_rng);
            }
            // Re-derive the future x*z from the (possibly perturbed) linear future,
            // mirroring free-run where the fed-back x*z is the product of predicted
            // x,z. A no-op in the teacher-forced baseline (both remedies off).
            future[3] = future[0] * future[2];

            esn_.TrainStep(targets, lr, 0.0f, aux); // fit the readout on x(t)
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
    }
}

FreeRunResult Lorenz::FreeRun(bool verbose)
{
    float past[4] = {}; // input-port drive: [past x, y, z, x*z]
    float future[4] = {}; // feedback-port drive: [x, y, z, x*z] of the model's prediction
    float targets[3] = {};
    float outputs[3] = {}; // the model's generative prediction each step
    float u_past[3] = {}; // auxiliary readout input: normalized past (x,y,z)
    // nullptr when the readout has no aux block (see Train). Aliases u_past.
    const float* const aux = (config::AUX_INPUT_DIM > 0) ? u_past : nullptr;

    RebuildDatastream(false);

    // Stage 1: anchored washout. Re-sweep the training window once, teacher-forced
    // but inference-only (no readout updates), so the reservoir crosses into the
    // generative region warm and in-distribution.
    data_stream_->Reset();
    LorenzDatastreamResult past_future_states = data_stream_->States();
    while (!data_stream_->OOB())
    {
        ExtractPast(past, past_future_states);
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
    for (size_t j = 0; j < config::FREE_RUN_WINDOW_SIZE; j++)
    {
        const int32_t f = data_stream_->Indices().second; // this step's held-out truth index
        if (f < 0 || static_cast<size_t>(f) >= S.size())
        {
            if (config::ENABLE_PRINTF && verbose)
                std::printf("[FreeRun] runway exhausted after %zu steps - stream ends\n", steps);
            break;
        }

        ExtractAuxPast(u_past, past_future_states); // u_raw for this Predict
        esn_.Predict(outputs, aux); // the prediction of S[f]
        ExtractPast(past, past_future_states); // anchored past -> input port
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

    // Per-seed outcome + display row (always built, independent of ENABLE_PRINTF).
    // VPT is the headline metric: the step at which the channel-RMS error first
    // crossed VPT_THRESHOLD, or ">= steps" (a lower bound) if it never did.
    const bool crossed = vpt_steps > 0;
    const double rmse = std::sqrt(sq_err_sum / (3.0 * steps));
    const double vpt_lt = (crossed ? vpt_steps : steps) / steps_per_lt;
    char buf[256];
    if (crossed)
        std::snprintf(buf, sizeof buf,
                      "seed %-10llu orbit_seed %-10llu VPT %3zu steps (%5.2f lt)  free-run RMSE %.6f\n",
                      static_cast<unsigned long long>(seed_), static_cast<unsigned long long>(orbit_seed_), vpt_steps, vpt_lt, rmse);
    else
        std::snprintf(buf, sizeof buf,
                      "seed %-10llu orbit_seed %-10llu VPT >=%3zu steps (%5.2f lt)  free-run RMSE %.6f  (never crossed %.2f)\n",
                      static_cast<unsigned long long>(seed_), static_cast<unsigned long long>(orbit_seed_), steps, vpt_lt, rmse, config::VPT_THRESHOLD);

    FreeRunResult r;
    r.valid = true;
    r.seed = seed_;
    r.vpt_steps = vpt_steps;
    r.crossed = crossed;
    r.vpt_lt = vpt_lt;
    r.rmse = rmse;
    r.steps = steps;
    r.row = buf;
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
    Lorenz lorenz(esn_seed, orbit_seed);
    lorenz.Train();

    std::vector<FreeRunResult> results; // one outcome per free-run (RebuildDatastream re-mixes the orbit each call)
    results.reserve(num_runs);
    for (int i = 0; i < num_runs; i++)
        results.push_back(lorenz.FreeRun(false));

    std::string out;
    char buf[256];
    auto emit = [&](const char* s) { out += s; };

    std::snprintf(buf, sizeof buf, "\n=== ESN seed %llu : %d free-runs (orbit seed %llu) ===\n",
                  static_cast<unsigned long long>(esn_seed), num_runs,
                  static_cast<unsigned long long>(orbit_seed));
    emit(buf);

    // Aggregate stats over the valid runs. VPT is in Lyapunov times; runs that
    // never crossed VPT_THRESHOLD contribute their window floor (a lower bound),
    // so the VPT stats are conservative when any run is censored (noted below).
    std::vector<double> vpt_lts, rmses;
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
        if (!r.crossed) ++censored;
    }

    auto report = [&](const char* label, std::vector<double> v, int prec)
    {
        if (v.empty())
        {
            std::snprintf(buf, sizeof buf, "  %-16s (no valid runs)\n", label);
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
        std::snprintf(buf, sizeof buf, "  %-16s n=%2zu  min=%.*f  max=%.*f  mean=%.*f  median=%.*f  std=%.*f\n",
                      label, n, prec, v.front(), prec, v.back(), prec, mean, prec, median, prec, sd);
        emit(buf);
    };

    std::snprintf(buf, sizeof buf, "\n=== Free-run stats (%d runs) ===\n", num_runs);
    emit(buf);
    report("VPT (lt)", vpt_lts, 2);
    report("free-run RMSE", rmses, 6);
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

    // Summary tables. Rank the valid runs and print the two leaderboards; each row
    // is the run's own display line (seed / orbit_seed / VPT / RMSE).
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

    return out;
}

int main(int argc, char** argv)
{
    uint64_t seed = 5941978990;
    uint64_t orbit_seed = 5859834983498;

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
    const size_t hw = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 1;
    const size_t max_threads = 4 * hw; // guard against a fat-fingered arg spawning a thread stampede
    size_t num_threads = hw;
    int num_runs = 2000;
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
