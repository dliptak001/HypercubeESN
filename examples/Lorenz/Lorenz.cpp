#include "Lorenz.h"
#include "LorenzDatastream.h"


EnsembleConfig Lorenz::MakeEnsembleConfig()
{
    EnsembleConfig cfg;
    cfg.SetDIM(config::DIM);
    cfg.SetSeed(config::SEED);

    // These are fixed by the design. One D, three roles: the feedback channel
    // count must equal num_outputs (= 3), or EnsembleESN's ctor rejects it.
    cfg.base.reservoir.num_inputs = 8; // 2 * [x, y, z, x*z]
    cfg.base.readout.num_outputs = 3; //[x, y, z]
    cfg.base.reservoir.num_feedback_channels = 3; // D = num_outputs

    cfg.base.reservoir.spectral_radius = config::SPECTRAL_RADIUS;
    cfg.base.reservoir.input_scaling = config::INPUT_SCALING;
    cfg.base.reservoir.leak_rate = config::LEAK_RATE;
    cfg.base.reservoir.history_depth = config::HISTORY_DEPTH;
    cfg.base.reservoir.lorentz_gamma = config::LORENTZ_GAMMA;
    cfg.base.reservoir.lorentz_inv_sigma2 = config::LORENTZ_INV_SIGMA2;

    cfg.base.readout.epochs = config::EPOCHS;
    return cfg;
}

LorenzDatastreamConfig Lorenz::MakeDatastreamConfig()
{
    LorenzDatastreamConfig cfg;
    cfg.stream_length = config::STREAM_LENGTH;
    cfg.cursor_span = config::CURSOR_SPAN;
    cfg.cursor_center_index = config::CURSOR_FOCUS_INDEX;
    cfg.initial_lorenz_state = config::INITIAL_LORENZ_STATE;
    cfg.lorenz_dt = config::DT;
    return cfg;
}

Lorenz::Lorenz() : esn_config_(MakeEnsembleConfig()), esn(esn_config_), data_stream_(MakeDatastreamConfig())
{
}

void Lorenz::Train()
{
    float inputs[8] = {};
    float targets[8] = {};
    const auto epochs = esn_config_.base.readout.epochs;
    for (size_t i = 0; i < epochs; i++)
    {
        data_stream_.Reset();
        LorenzDatastreamResult past_future_states = data_stream_.GetInitialStates();
        inputs[0] = std::get<1>(past_future_states).x;  //past
        inputs[1] = std::get<1>(past_future_states).y;  //past
        inputs[2] = std::get<1>(past_future_states).z;  //past
        inputs[3] = std::get<2>(past_future_states)->x; //future
        inputs[4] = std::get<2>(past_future_states)->y; //future
        inputs[5] = std::get<2>(past_future_states)->z; //future
        inputs[6] = std::get<0>(past_future_states);
        inputs[7] = inputs[0]*inputs[2];    //past xz product
        //esn.Step(inputs, targets, );


    }
}

int main()
{
    std::cout << "=== HypercubeESN: Lorenz ===\n";
    return 0;
}
