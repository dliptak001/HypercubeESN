#include "Lorenz.h"
#include "LorenzDatastream.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <windows.h>


EnsembleConfig Lorenz::MakeEnsembleConfig(uint64_t seed)
{
    EnsembleConfig cfg;
    cfg.SetDIM(config::DIM);
    cfg.SetSeed(seed);

    cfg.combine = config::COMBINE;
    cfg.learning_rate = config::LEARNING_RATE;

    // [x_past, y_past, z_past, x_future, y_future, z_future, distance, x_past*z_past]
    cfg.base.reservoir.num_inputs = 8;
    cfg.base.reservoir.num_feedback_channels = 3; // D = num_outputs
    cfg.base.reservoir.spectral_radius = config::SPECTRAL_RADIUS;
    cfg.base.reservoir.input_scaling = config::INPUT_SCALING;
    cfg.base.reservoir.leak_rate = config::LEAK_RATE;
    cfg.base.reservoir.history_depth = config::HISTORY_DEPTH;

    // The feedback channel count must equal num_outputs (= 3), or EnsembleESN's ctor rejects it.
    cfg.base.readout.num_outputs = 3; //[x, y, z]
    return cfg;
}

LorenzDatastreamConfig Lorenz::MakeDatastreamConfig()
{
    LorenzDatastreamConfig cfg;
    cfg.stream_length = config::STREAM_LENGTH;
    cfg.cursor_span = config::TRAINING_WINDOW_SIZE;
    cfg.cursor_center_index = config::CURSOR_CENTER_INDEX;
    cfg.initial_lorenz_state = config::INITIAL_LORENZ_STATE;
    cfg.lorenz_dt = config::DT;
    return cfg;
}

Lorenz::Lorenz(const uint64_t seed) : seed_(seed), esn_config_(MakeEnsembleConfig(seed_)), esn_(esn_config_),
                                      data_stream_(MakeDatastreamConfig())
{
    if (config::ENABLE_PRINTF)
    {
        std::printf("[Lorenz config] reservoir: DIM=%zu (N=%zu)  seed=%llu  SR=%.3f  input_scaling=%.3f  leak=%.2f"
                    "  history_depth=%zu\n",
                    config::DIM, size_t{1} << config::DIM, static_cast<unsigned long long>(config::SEED),
                    config::SPECTRAL_RADIUS, config::INPUT_SCALING, config::LEAK_RATE,
                    config::HISTORY_DEPTH);
        std::printf("[Lorenz config] ensemble:  M=%zu  kappa_max=%.3f  combine=%s\n",
                    esn_.NumMembers(), config::KAPPA, config::COMBINE == Combine::Mean ? "mean" : "median");
        std::printf("[Lorenz config] readout:   lr %.6f -> %.6f   epochs=%zu\n",
                    config::LEARNING_RATE, config::LEARNING_RATE_MIN, config::EPOCHS);
        std::printf("[Lorenz config] exposure:  2a future_noise=%.4f  2b ss_ceiling=%.3f\n",
                    config::TRAIN_FUTURE_NOISE, config::SCHEDULED_SAMPLING_CEILING);
        std::printf("[Lorenz config] stream:    x0=(%.2f, %.2f, %.2f)  warmup=%zu\n",
                    config::INITIAL_LORENZ_STATE.x, config::INITIAL_LORENZ_STATE.y, config::INITIAL_LORENZ_STATE.z,
                    config::RESERVOIR_WARMUP_STEPS);
    }
}

void Lorenz::ExtractInputs_Training(float inputs[8], const LorenzDatastreamResult& past_future_states)
{
    inputs[0] = std::get<1>(past_future_states).x; //past
    inputs[1] = std::get<1>(past_future_states).y; //past
    inputs[2] = std::get<1>(past_future_states).z; //past
    inputs[3] = std::get<2>(past_future_states)->x; //future
    inputs[4] = std::get<2>(past_future_states)->y; //future
    inputs[5] = std::get<2>(past_future_states)->z; //future
    inputs[6] = inputs[0] * inputs[2]; //past xz product
    inputs[7] = inputs[6];
    //inputs[3] * inputs[5];//std::get<0>(past_future_states); //distance between past and future indices
}

void Lorenz::ExtractInputs_FreeRun(float inputs[8], const LorenzDatastreamResult& past_future_states,
                                   const float* consensus)
{
    inputs[0] = std::get<1>(past_future_states).x; //past
    inputs[1] = std::get<1>(past_future_states).y; //past
    inputs[2] = std::get<1>(past_future_states).z; //past
    inputs[3] = consensus[0]; //future: the ensemble's last consensus output
    inputs[4] = consensus[1]; //future
    inputs[5] = consensus[2]; //future
    inputs[6] = inputs[0] * inputs[2]; //past xz product
    inputs[7] = inputs[6];
    //inputs[3] * inputs[5];//std::get<0>(past_future_states); //distance between past and future indices
}

void Lorenz::ExtractTargets(float targets[3], const NormalizedState& future_state)
{
    targets[0] = future_state.x;
    targets[1] = future_state.y;
    targets[2] = future_state.z;
}

double Lorenz::KappaProfile(double kappa_max, double k, size_t epochs, const size_t current_epoch)
{
    double x = static_cast<double>(current_epoch) / epochs;
    double c = k * x * x;
    return kappa_max * c / (1.0 + c);
}

float Lorenz::LrProfile(const float lr_max, const float lr_min, const size_t epochs, const size_t current_epoch)
{
    if (epochs <= 1)
        return lr_max;
    const float progress = static_cast<float>(current_epoch) / static_cast<float>(epochs - 1);
    return CosineLR(progress, lr_max, lr_min); // swap in ExponentialDecayLR (Readout.h) to compare schedules
    //return ExponentialDecayLR(progress, lr_max, lr_min);
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
    float inputs[8] = {};
    float targets[3] = {};
    float outputs[3] = {};
    float consensus[3] = {}; // 2b: the ensemble's own prediction, when scheduled sampling fires
    const size_t M = esn_.NumMembers();
    const size_t D = esn_.NumOutputs();
    std::vector<float> member_y(M * D); // per-step member outputs (diagnostic read)
    std::vector<double> dev_sq(M); // per-member sum of squared consensus deviations

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
        // Step 1: Warm up the reservoir. The epoch's kappa is set BEFORE the
        // warmup loop on purpose: warmup runs with the coupling live.
        data_stream_.Reset();
        esn_.SetKappa(KappaProfile(config::KAPPA, 25.0, config::EPOCHS, i));
        esn_.SetLr(LrProfile(config::LEARNING_RATE, config::LEARNING_RATE_MIN, config::EPOCHS, i));
        // 2b: this epoch's probability of substituting the model's own prediction
        // on the future channels (0 when the ceiling is 0, i.e. 2b disabled).
        const float ss_p = ScheduledSamplingProfile(config::SCHEDULED_SAMPLING_CEILING, config::EPOCHS, i);
        LorenzDatastreamResult past_future_states = data_stream_.States();
        for (size_t j = 0; j < config::RESERVOIR_WARMUP_STEPS; j++)
        {
            ExtractInputs_Training(inputs, past_future_states);
            esn_.Step(inputs, nullptr, outputs);
            past_future_states = data_stream_.Step(false);
        }

        // Step 2: Train - train towards future state targets
        double sq_err_sum = 0.0; // double accumulator: ~15K float-sized terms/epoch
        std::fill(dev_sq.begin(), dev_sq.end(), 0.0);
        size_t train_steps = 0;
        while (!data_stream_.OOB())
        {
            // Horizon-1 alignment: EnsembleESN::Step fits the readout on x(t) BEFORE
            // injecting this call's inputs, so x(t) has seen the future channel only
            // through S[f-1]. The aligned one-step target is S[f] — the sample this
            // call is about to inject — not NextFutureState() = S[f+1].
            ExtractTargets(targets, *std::get<2>(past_future_states));
            ExtractInputs_Training(inputs, past_future_states);

            // Exposure-bias remedies (README Issue 2), future channels 3-5 only;
            // the teacher target above stays the real S[f], and channels 6-7 are
            // past-derived so they remain consistent with free-run under both.
            //
            // 2b scheduled sampling: with probability ss_p, overwrite the future
            // channels with the ensemble's own fresh prediction of S[f]. Predict
            // reads x(t) WITHOUT advancing it (the closed-loop ordering FreeRun
            // uses); Step's own read below sees the same unchanged state, so the
            // prequential diagnostic still pairs outputs with this call's target.
            if (config::SCHEDULED_SAMPLING_CEILING > 0.0f && unit_uniform(exposure_rng) < ss_p)
            {
                esn_.Predict(consensus);
                inputs[3] = consensus[0];
                inputs[4] = consensus[1];
                inputs[5] = consensus[2];
            }
            // 2a noise injection: zero-mean Gaussian on the (possibly substituted)
            // future channels — trains the map to tolerate the perturbation class
            // the closed loop introduces.
            if (config::TRAIN_FUTURE_NOISE > 0.0f)
            {
                inputs[3] += future_noise(exposure_rng);
                inputs[4] += future_noise(exposure_rng);
                inputs[5] += future_noise(exposure_rng);
            }

            esn_.Step(inputs, targets, outputs);

            // Prequential (test-then-train) error: outputs is the consensus read at
            // x(t) before this call's TrainStep, i.e. the pre-update prediction of
            // this call's own target — the pairing is exact within one Step.
            for (size_t c = 0; c < 3; c++)
            {
                const double e = static_cast<double>(outputs[c]) - targets[c];
                sq_err_sum += e * e;
            }
            // Raw consensus deviations y_i - c: AllMemberOutputs returns the y_i
            // this Step used to form the consensus, so this is exactly the
            // pre-kappa error the Step fed back as phi_i = kappa*(y_i - c).
            esn_.AllMemberOutputs(member_y.data());
            for (size_t m = 0; m < M; m++)
            {
                for (size_t c = 0; c < D; c++)
                {
                    const double d = static_cast<double>(member_y[m * D + c]) - outputs[c];
                    dev_sq[m] += d * d;
                }
            }
            ++train_steps;

            past_future_states = data_stream_.Step(false);
        }

        if (train_steps > 0)
        {
            // Per-member epoch RMS of the raw (pre-kappa) consensus deviation,
            // and the spread (population std) of those M values.
            double dev_mean = 0.0;
            std::vector<double> dev_rms(M);
            for (size_t m = 0; m < M; m++)
            {
                dev_rms[m] = std::sqrt(dev_sq[m] / (static_cast<double>(D) * train_steps));
                dev_mean += dev_rms[m];
            }
            dev_mean /= static_cast<double>(M);
            double dev_var = 0.0;
            for (size_t m = 0; m < M; m++)
                dev_var += (dev_rms[m] - dev_mean) * (dev_rms[m] - dev_mean);
            dev_var /= static_cast<double>(M);
            if (config::ENABLE_PRINTF)
            {
                std::printf("epoch %3zu kappa %.4f lr %.7f  train RMSE %.6f  dev[",
                            i, esn_.Kappa(), esn_.Lr(), std::sqrt(sq_err_sum / (3.0 * train_steps)));
                for (size_t m = 0; m < M; m++)
                    std::printf(" %.6f", dev_rms[m]);
                std::printf(" ]  sd %.6f  (%zu steps)\n", std::sqrt(dev_var), train_steps);
            }
        }
        else if (config::ENABLE_PRINTF)
            std::printf("epoch %3zu kappa %.4f lr %.7f  train RMSE n/a  (0 steps - warmup consumed the window)\n",
                        i, esn_.Kappa(), esn_.Lr());
    }
}

std::string Lorenz::FreeRun()
{
    float inputs[8] = {};
    float targets[3] = {};
    float consensus[3] = {};
    float outputs[3] = {};
    float y_m[3] = {};
    std::vector<float> member_inputs(esn_.NumMembers() * 8); // one 8-channel row per member
    std::vector<float> member_y(esn_.NumMembers() * 3); // fresh member outputs (spread diagnostic)

    // Stage 1: anchored washout. Re-sweep the training window once, teacher-forced
    // but inference-only (no readout updates), so the reservoirs cross into the
    // generative region warm and in-distribution. Kappa holds at the ceiling.
    data_stream_.Reset();
    esn_.SetKappa(config::KAPPA);
    LorenzDatastreamResult past_future_states = data_stream_.States();
    while (!data_stream_.OOB())
    {
        ExtractInputs_Training(inputs, past_future_states);
        esn_.Step(inputs, nullptr, outputs);
        past_future_states = data_stream_.Step(false);
    }

    // Stage 2: generative rollout, PER-MEMBER closed loop. The future cursor is
    // now one step past the window edge: each member's channels 3-5 switch to
    // that member's OWN prediction (the consensus is only reported and scored),
    // so member trajectories genuinely diverge and the deviation coupling
    // phi_i = kappa*(y_i - c) becomes a live consensus-attraction force.
    // Predict() reads the fresh member outputs BEFORE StepPerMember consumes
    // them as inputs — the closed-loop ordering Step alone cannot express.

    const std::vector<NormalizedState>& S = data_stream_.GetDataStream();
    const double steps_per_lt = 1.0 / (config::LYAPUNOV_EXPONENT * config::DT);
    if (config::ENABLE_PRINTF)
    {
        std::printf("[FreeRun] generative: %zu steps (%.1f Lyapunov times)  kappa %.4f  vpt_threshold %.2f\n",
                    config::FREE_RUN_WINDOW_SIZE, config::FREE_RUN_WINDOW_SIZE / steps_per_lt,
                    esn_.Kappa(), config::VPT_THRESHOLD);
        std::printf("[FreeRun] drive: per-member (each member fed its own prediction on the future channels)\n");
    }

    double sq_err_sum = 0.0;
    size_t steps = 0;
    size_t vpt_steps = 0; // first step whose error exceeded VPT_THRESHOLD (0 = never)
    for (size_t j = 0; j < config::FREE_RUN_WINDOW_SIZE; j++)
    {
        const int32_t f = data_stream_.Indices().second; // this step's held-out truth index
        if (f < 0 || static_cast<size_t>(f) >= S.size())
        {
            if (config::ENABLE_PRINTF)
                std::printf("[FreeRun] runway exhausted after %zu steps - stream ends\n", steps);
            break;
        }

        esn_.Predict(consensus); // the prediction of S[f]; also refreshes the member outputs
        for (size_t m = 0; m < esn_.NumMembers(); m++)
        {
            esn_.MemberOutput(m, y_m); // member m's own prediction of S[f]
            ExtractInputs_FreeRun(&member_inputs[m * 8], past_future_states, y_m);
        }

        // Member spread: RMS of (y_m - c) over members x channels — how far the
        // independently-fed trajectories have drifted apart before phi pulls them
        // back (the coupling's working signal; the consensus-error counterpart of
        // Train's dev[] diagnostic).
        esn_.AllMemberOutputs(member_y.data());
        double spread_sq = 0.0;
        for (size_t m = 0; m < esn_.NumMembers(); m++)
        {
            for (size_t c = 0; c < 3; c++)
            {
                const double d = static_cast<double>(member_y[m * 3 + c]) - consensus[c];
                spread_sq += d * d;
            }
        }
        const double spread = std::sqrt(spread_sq / (3.0 * esn_.NumMembers()));
        esn_.StepPerMember(member_inputs.data(), nullptr, outputs); // absorb the fed-back predictions
        //ExtractInputs_FreeRun(inputs, past_future_states, consensus); // consensus-driven arm
        //esn_.Step(inputs, nullptr, outputs);

        // Score against the true held-out orbit (normalized units).
        ExtractTargets(targets, S[f]);
        double step_sq = 0.0;
        for (size_t c = 0; c < 3; c++)
        {
            const double e = static_cast<double>(consensus[c]) - targets[c];
            step_sq += e * e;
        }
        sq_err_sum += step_sq;
        ++steps;

        const double step_err = std::sqrt(step_sq / 3.0);
        if (vpt_steps == 0 && step_err > config::VPT_THRESHOLD)
            vpt_steps = steps;
        if (config::ENABLE_PRINTF && steps % 25 == 0)
            std::printf("free-run %4zu  (%5.2f lt)  err %.6f  spread %.6f\n",
                        steps, steps / steps_per_lt, step_err, spread);

        if (steps == config::FREE_RUN_WINDOW_SIZE)
            break; // done - don't step the cursors past the last scored index
        if (data_stream_.Indices().first <= 0)
        {
            if (config::ENABLE_PRINTF)
                std::printf("[FreeRun] anchor runway exhausted after %zu steps - past cursor at the seed\n", steps);
            break;
        }
        past_future_states = data_stream_.Step(true);
    }

    if (steps == 0)
        return "";

    // Per-seed result row for the survey table (always built, independent of
    // ENABLE_PRINTF). VPT is the headline metric: the step at which the channel-RMS
    // error first crossed VPT_THRESHOLD, or ">= steps" if it never did.
    const double rmse = std::sqrt(sq_err_sum / (3.0 * steps));
    char buf[256];
    if (vpt_steps > 0)
        std::snprintf(buf, sizeof buf,
                      "seed %-10llu  VPT %3zu steps (%5.2f lt)  free-run RMSE %.6f  (%zu steps)\n",
                      static_cast<unsigned long long>(seed_), vpt_steps, vpt_steps / steps_per_lt,
                      rmse, steps);
    else
        std::snprintf(buf, sizeof buf,
                      "seed %-10llu  VPT >=%3zu steps (%5.2f lt)  free-run RMSE %.6f  (never crossed %.2f)\n",
                      static_cast<unsigned long long>(seed_), steps, steps / steps_per_lt,
                      rmse, config::VPT_THRESHOLD);
    return buf;
}

int main()
{
    std::cout << "=== HypercubeESN: Lorenz ===\n";

    constexpr size_t NUM_SEEDS = 16;
    std::vector<std::string> results(NUM_SEEDS); // one result row per seed, filled in place

    // Each seed's run is fully independent (its own EnsembleESN + datastream, no
    // shared mutable state), so the survey parallelizes cleanly one-instance-per-
    // thread. Workers pull seed indices off a shared atomic counter; a bounded pool
    // (<= hardware_concurrency) avoids oversubscribing the cores.
    std::atomic<size_t> next_seed{0};
    const unsigned hw = std::thread::hardware_concurrency();
    const size_t num_threads = std::min<size_t>(NUM_SEEDS, hw ? hw : 1);

    {
        std::vector<std::jthread> pool;
        pool.reserve(num_threads);
        for (size_t t = 0; t < num_threads; t++)
            pool.emplace_back([&]
            {
                for (size_t i = next_seed.fetch_add(1); i < NUM_SEEDS; i = next_seed.fetch_add(1))
                {
                    Lorenz lorenz(3648759 + 33 * i);
                    lorenz.Train();
                    results[i] = lorenz.FreeRun(); // disjoint slot — no lock needed
                }
            });
    } // jthreads join on scope exit

    // Print the per-seed result table in seed order (independent of completion order).
    std::printf("\n=== Seed survey (%zu seeds) ===\n", NUM_SEEDS);
    for (size_t i = 0; i < NUM_SEEDS; i++)
        std::fputs(results[i].c_str(), stdout);

    // TODO Claude - report min, max, std... here

    Beep(2500, 3000); // single completion beep for the whole survey

    return 0;
}
