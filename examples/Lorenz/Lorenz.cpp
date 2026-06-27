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
    cfg.cursor_focus_index = config::CURSOR_FOCUS_INDEX;
    cfg.initial_lorenz_state = config::INITIAL_LORENZ_STATE;
    cfg.lorenz_dt = config::DT;
    return cfg;
}

Lorenz::Lorenz() : esn_config_(MakeEnsembleConfig()), esn(esn_config_), data_stream_(MakeDatastreamConfig())
{
}

int main()
{
    std::cout << "=== HypercubeESN: Lorenz ===\n";
    return 0;
}
