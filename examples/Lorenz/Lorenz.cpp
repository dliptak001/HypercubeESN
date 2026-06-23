#include "Lorenz.h"

// Build the populated config before any member is constructed: the member
// init list runs before the ctor body, so esn must be fed a config that is
// already filled in. A static factory keeps that ordering honest.
EnsembleConfig Lorenz::MakeConfig()
{
    EnsembleConfig cfg;
    cfg.SetDIM(config::DIM);
    cfg.SetSeed(config::SEED);

    // These are fixed by the design. One D, three roles: the feedback channel
    // count must equal num_outputs (= 3), or EnsembleESN's ctor rejects it.
    cfg.base.reservoir.num_inputs = 8; // 2 * [x, y, z, x*y*z]
    cfg.base.readout.num_outputs = 3; //[x, y, z]
    cfg.base.reservoir.num_feedback_channels = 3; // D = num_outputs

    cfg.base.reservoir.spectral_radius = config::SPECTRAL_RADIUS;
    cfg.base.reservoir.input_scaling = config::INPUT_SCALING;
    cfg.base.reservoir.leak_rate = config::LEAK_RATE;
    cfg.base.reservoir.history_depth = config::HISTORY_DEPTH;
    cfg.base.reservoir.lorentz_gamma = config::LORENTZ_GAMMA;
    cfg.base.reservoir.lorentz_inv_sigma2 = config::LORENTZ_INV_SIGMA2;

    cfg.base.readout.epochs = config::EPOCHS;

    cfg.washout = config::WASHOUT;
    cfg.lr = config::LR;
    cfg.weight_decay = config::WEIGHT_DECAY;
    return cfg;
}

Lorenz::Lorenz()
    : esn_config_(MakeConfig()),
      esn(esn_config_)
{
}

int main()
{
    std::cout << "=== HypercubeESN: Lorenz ===\n";
    return 0;
}
