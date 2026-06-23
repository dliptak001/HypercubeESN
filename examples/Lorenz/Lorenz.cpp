#include "Lorenz.h"

Lorenz::Lorenz()
{
    esn_config_.SetDIM(config::DIM);
    esn_config_.SetSeed(config::SEED);

    // These are fixed by the design
    esn_config_.base.reservoir.num_inputs = 8; // 2 * [x, y, z, x*y*z]
    esn_config_.base.readout.num_outputs = 3; //[x, y, z]

    esn_config_.base.reservoir.spectral_radius = config::SPECTRAL_RADIUS;
    esn_config_.base.reservoir.input_scaling = config::INPUT_SCALING;
    esn_config_.base.reservoir.leak_rate = config::LEAK_RATE;
    esn_config_.base.reservoir.history_depth = config::HISTORY_DEPTH;
    esn_config_.base.reservoir.lorentz_gamma = config::LORENTZ_GAMMA;
    esn_config_.base.reservoir.lorentz_inv_sigma2 = config::LORENTZ_INV_SIGMA2;

    esn_config_.washout = config::WASHOUT;
    esn_config_.lr = config::LR;
    esn_config_.weight_decay = config::WEIGHT_DECAY;
    esn_config_.base.readout.epochs = config::EPOCHS;
}


int main()
{
    std::cout << "=== HypercubeESN: Lorenz ===\n";
    return 0;
}
