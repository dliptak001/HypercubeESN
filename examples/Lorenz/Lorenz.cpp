#include "Lorenz.h"
#include "LorenzDatastream.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>


EnsembleConfig Lorenz::MakeEnsembleConfig()
{
    EnsembleConfig cfg;
    cfg.SetDIM(config::DIM);
    cfg.SetSeed(config::SEED);

    cfg.learning_rate = config::LEARNING_RATE;
    cfg.weight_decay = config::WEIGHT_DECAY;

    // [x_past, y_past, z_past, x_future, y_future, z_future, distance, x_past*z_past]
    cfg.base.reservoir.num_inputs = 8;
    cfg.base.reservoir.num_feedback_channels = 3; // D = num_outputs
    cfg.base.reservoir.spectral_radius = config::SPECTRAL_RADIUS;
    cfg.base.reservoir.input_scaling = config::INPUT_SCALING;
    cfg.base.reservoir.leak_rate = config::LEAK_RATE;
    cfg.base.reservoir.history_depth = config::HISTORY_DEPTH;
    cfg.base.reservoir.lorentz_gamma = config::LORENTZ_GAMMA;

    // The feedback channel count must equal num_outputs (= 3), or EnsembleESN's ctor rejects it.
    cfg.base.readout.num_outputs = 3; //[x, y, z]
    return cfg;
}

LorenzDatastreamConfig Lorenz::MakeDatastreamConfig()
{
    LorenzDatastreamConfig cfg;
    cfg.stream_length = config::STREAM_LENGTH;
    cfg.cursor_span = config::CURSOR_SPAN;
    cfg.cursor_center_index = config::CURSOR_CENTER_INDEX;
    cfg.initial_lorenz_state = config::INITIAL_LORENZ_STATE;
    cfg.lorenz_dt = config::DT;
    return cfg;
}

Lorenz::Lorenz() : esn_config_(MakeEnsembleConfig()), esn_(esn_config_), data_stream_(MakeDatastreamConfig())
{
}

void Lorenz::ExtractInputs_Training(float inputs[8], const LorenzDatastreamResult& past_future_states)
{
    inputs[0] = std::get<1>(past_future_states).x; //past
    inputs[1] = std::get<1>(past_future_states).y; //past
    inputs[2] = std::get<1>(past_future_states).z; //past
    inputs[3] = std::get<2>(past_future_states)->x; //future
    inputs[4] = std::get<2>(past_future_states)->y; //future
    inputs[5] = std::get<2>(past_future_states)->z; //future
    inputs[6] = std::get<0>(past_future_states); //distance between past and future indices
    inputs[7] = inputs[0] * inputs[2]; //past xz product
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
    inputs[6] = std::get<0>(past_future_states); //distance between past and future indices
    inputs[7] = inputs[0] * inputs[2]; //past xz product
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
    double c = k*x*x;
    return kappa_max*c/(1.0 + c);
}

float Lorenz::LrProfile(const float lr_max, const float lr_min, const size_t hold_epochs,
                        const size_t epochs, const size_t current_epoch)
{
    if (current_epoch <= hold_epochs || epochs <= hold_epochs + 1)
        return lr_max;
    const float progress = static_cast<float>(current_epoch - hold_epochs) /
                           static_cast<float>(epochs - 1 - hold_epochs);
    return CosineLR(progress, lr_max, lr_min);
}

void Lorenz::Train()
{
    float inputs[8] = {};
    float targets[3] = {};
    float outputs[3] = {};
    const size_t M = esn_.NumMembers();
    const size_t D = esn_.NumOutputs();
    std::vector<float> member_y(M * D); // per-step member outputs (diagnostic read)
    std::vector<double> dev_sq(M); // per-member sum of squared consensus deviations
    for (size_t i = 0; i < config::EPOCHS; i++)
    {
        // Step 1: Warm up the reservoir. The epoch's kappa is set BEFORE the
        // warmup loop on purpose: warmup runs with the coupling live.
        data_stream_.Reset();
        esn_.SetKappa(KappaProfile(config::KAPPA, 25.0, config::EPOCHS, i));
        esn_.SetLr(LrProfile(config::LEARNING_RATE, config::LEARNING_RATE_MIN,
                             config::LR_HOLD_EPOCHS, config::EPOCHS, i));
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

            std::printf("epoch %3zu  kappa %.4f  lr %.5f  train RMSE %.6f  dev[",
                        i, esn_.Kappa(), esn_.Lr(), std::sqrt(sq_err_sum / (3.0 * train_steps)));
            for (size_t m = 0; m < M; m++)
                std::printf(" %.6f", dev_rms[m]);
            std::printf(" ]  sd %.6f  (%zu steps)\n", std::sqrt(dev_var), train_steps);
        }
        else
            std::printf("epoch %3zu  kappa %.4f  lr %.5f  train RMSE n/a  (0 steps - warmup consumed the window)\n",
                        i, esn_.Kappa(), esn_.Lr());
    }
}

int main()
{
    std::cout << "=== HypercubeESN: Lorenz ===\n";
    Lorenz lorenz;
    lorenz.Train();
    return 0;
}
