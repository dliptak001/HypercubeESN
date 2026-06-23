#include "ESN.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

ReadoutConfig ESN::MakeReadoutConfig(const ESNConfig& cfg)
{
    // The readout sees all N = 2^dim reservoir vertices, so its hypercube
    // dimension is simply the reservoir's dim.
    ReadoutConfig rc = cfg.readout;
    rc.dim = cfg.reservoir.dim;
    return rc;
}

ESN::ESN(const ESNConfig& cfg)
    : reservoir_(Reservoir::Create(cfg.reservoir)),
      readout_(MakeReadoutConfig(cfg))
{
    n_                = reservoir_->Size();
    num_inputs_       = cfg.reservoir.num_inputs;
    esn_config_       = cfg;
    // The user leaves cfg.readout.dim at 0 ("do not set"); reflect the value the
    // readout was actually built with so GetConfig() doesn't report a stale 0.
    esn_config_.readout.dim = reservoir_->Dim();

    scratch_state_.resize(n_);
}

void ESN::ReservoirStep(const float* inputs, const float* feedback)
{
    // Closed-loop only when feedback is actually supplied. Stage it on the D
    // feedback channels (raw, no clamp) through the reservoir's dedicated port
    // (own weights + feedback_scaling, outside the SR rescale — a twin of the
    // input port). The ESN never generates feedback itself.
    if (feedback != nullptr)
    {
        // Guard the contract explicitly: with D == 0 the reservoir has no
        // feedback port, and InjectFeedback(ptr, 0) would no-op (0 == 0 passes
        // its count check) — silently degrading this to an open-loop step. Throw
        // instead, so a feedback caller on a non-feedback ESN fails loud.
        if (esn_config_.reservoir.num_feedback_channels == 0)
            throw std::invalid_argument(
                "ESN::ReservoirStep: feedback supplied but not configured "
                "(num_feedback_channels == 0); pass feedback=nullptr for open-loop drive");
        reservoir_->InjectFeedback(feedback, esn_config_.reservoir.num_feedback_channels);
    }

    for (size_t ch = 0; ch < num_inputs_; ++ch)
        reservoir_->InjectInput(ch, inputs[ch]);
    reservoir_->Step();
}

void ESN::ReservoirWarmup(const float* inputs, size_t num_steps)
{
    for (size_t s = 0; s < num_steps; ++s)
        ReservoirStep(inputs + s * num_inputs_);
}

void ESN::ReservoirRun(const float* inputs, size_t num_steps, bool clear_recorded)
{
    if (clear_recorded)
    {
        states_.clear();        // keep capacity; the resize below refills it
        num_collected_ = 0;
    }
    states_.resize((num_collected_ + num_steps) * n_);
    for (size_t s = 0; s < num_steps; ++s)
    {
        ReservoirStep(inputs + s * num_inputs_);
        CopyReservoirState(states_.data() + (num_collected_ + s) * n_);
    }
    num_collected_ += num_steps;
}

void ESN::ReservoirClear()
{
    reservoir_->Clear();
}

void ESN::Train(const float* targets, size_t train_size)
{
    if (train_size > num_collected_)
        throw std::out_of_range(
            "ESN::Train: train_size (" + std::to_string(train_size) +
            ") exceeds num_collected (" + std::to_string(num_collected_) + ")");
    readout_.Train(ReadoutInput(0), targets, train_size);
}

void ESN::TrainStep(const float* target, float lr, float weight_decay)
{
    CopyReservoirState(scratch_state_.data());
    readout_.TrainStep(scratch_state_.data(), target, lr, weight_decay);
}

void ESN::CopyReservoirState(float* out) const
{
    const float* src = reservoir_->Outputs();
    std::memcpy(out, src, n_ * sizeof(float));
}

void ESN::TrainStepBatch(const float* states, const float* targets,
                         size_t count, float lr, float weight_decay)
{
    readout_.TrainStepBatch(states, targets, count, lr, weight_decay);
}

std::vector<float> ESN::Predict() const
{
    std::vector<float> out(readout_.NumOutputs());
    Predict(out.data());
    return out;
}

void ESN::Predict(float* out) const
{
    CopyReservoirState(scratch_state_.data());
    readout_.PredictRaw(scratch_state_.data(), out);
}

std::vector<float> ESN::PredictFromRecorded(size_t timestep) const
{
    if (timestep >= num_collected_)
        throw std::out_of_range(
            "ESN::PredictFromRecorded: timestep (" + std::to_string(timestep) +
            ") >= num_collected (" + std::to_string(num_collected_) + ")");
    std::vector<float> out(readout_.NumOutputs());
    readout_.PredictRaw(ReadoutInput(timestep), out.data());
    return out;
}

std::vector<float> ESN::PredictFromState(const float* state) const
{
    std::vector<float> out(readout_.NumOutputs());
    readout_.PredictRaw(state, out.data());
    return out;
}

double ESN::R2(const float* targets, size_t start, size_t count) const
{
    if (start + count > num_collected_)
        throw std::out_of_range(
            "ESN::R2: start + count (" + std::to_string(start + count) +
            ") > num_collected (" + std::to_string(num_collected_) + ")");
    return readout_.R2(ReadoutInput(start), targets + start * readout_.NumOutputs(), count);
}

double ESN::NRMSE(const float* targets, size_t start, size_t count) const
{
    if (start + count > num_collected_)
        throw std::out_of_range(
            "ESN::NRMSE: start + count (" + std::to_string(start + count) +
            ") > num_collected (" + std::to_string(num_collected_) + ")");
    if (count == 0) return 0.0;

    const size_t K = readout_.NumOutputs();
    const float* tgt = targets + start * K;

    std::vector<float> preds(count * K);
    for (size_t s = 0; s < count; ++s)
        readout_.PredictRaw(ReadoutInput(start + s), preds.data() + s * K);

    double nrmse_sum = 0.0;
    for (size_t k = 0; k < K; ++k) {
        double mean = 0.0;
        for (size_t s = 0; s < count; ++s)
            mean += tgt[s * K + k];
        mean /= static_cast<double>(count);

        double var = 0.0, mse_k = 0.0;
        for (size_t s = 0; s < count; ++s) {
            double y  = tgt[s * K + k];
            double yh = preds[s * K + k];
            var += (y - mean) * (y - mean);
            mse_k += (y - yh) * (y - yh);
        }
        if (var < 1e-12)
            nrmse_sum += std::numeric_limits<double>::infinity();
        else
            nrmse_sum += std::sqrt(mse_k / count) / std::sqrt(var / count);
    }
    return nrmse_sum / static_cast<double>(K);
}

double ESN::Accuracy(const float* labels, size_t start, size_t count) const
{
    if (start + count > num_collected_)
        throw std::out_of_range(
            "ESN::Accuracy: start + count (" + std::to_string(start + count) +
            ") > num_collected (" + std::to_string(num_collected_) + ")");
    return readout_.Accuracy(ReadoutInput(start), labels + start, count);
}

ESNConfig ESN::GetConfig() const
{
    return esn_config_;
}

ESN::ReadoutState ESN::GetReadoutState() const
{
    ReadoutState s;
    // IsTrained() is set when the readout CNN is built (in the Readout ctor),
    // so it captures any readout that has weights worth persisting.
    s.is_trained = readout_.IsTrained();
    const auto& w = readout_.Weights();
    s.weights.assign(w.begin(), w.end());
    return s;
}

void ESN::SetReadoutState(const ReadoutState& state)
{
    if (!state.is_trained) return;
    readout_.SetState(state.weights);
}

// ---------------------------------------------------------------
//  Collected-state access helpers
// ---------------------------------------------------------------

const float* ESN::ReadoutInput(size_t timestep) const
{
    return states_.data() + timestep * n_;
}

std::vector<float> ESN::ReadoutStates(size_t start, size_t count) const
{
    std::vector<float> buf(count * n_);
    std::memcpy(buf.data(),
                states_.data() + start * n_,
                count * n_ * sizeof(float));
    return buf;
}

std::vector<float> ESN::CollectedStates() const
{
    return ReadoutStates(0, num_collected_);
}
