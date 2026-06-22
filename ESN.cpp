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

void ESN::StepLive(const float* inputs)
{
    // Input-only (open-loop) step. External feedback drive goes through
    // StepLiveExternalFeedback; the ESN never generates feedback itself.
    for (size_t ch = 0; ch < num_inputs_; ++ch)
        reservoir_->InjectInput(ch, inputs[ch]);
    reservoir_->Step();
}

void ESN::StepLiveExternalFeedback(const float* inputs, const float* feedback)
{
    // Guard the documented contract explicitly: with D == 0 the reservoir has no
    // feedback port, and InjectFeedback(ptr, 0) would no-op (0 == 0 passes its
    // count check) — silently degrading this to an open-loop Step. Throw instead,
    // so a feedback caller on a non-feedback ESN fails loud rather than running
    // input-only without notice.
    if (esn_config_.reservoir.num_feedback_channels == 0)
        throw std::invalid_argument(
            "ESN::StepLiveExternalFeedback: feedback is not configured "
            "(num_feedback_channels == 0); use StepLive for open-loop drive");

    // Stage the caller-supplied feedback on the D feedback channels (raw, no
    // clamp), then the task inputs, then Step. The reservoir routes feedback
    // through its dedicated port (own weights + feedback_scaling, outside the SR
    // rescale) — structurally a twin of the input port.
    reservoir_->InjectFeedback(feedback, esn_config_.reservoir.num_feedback_channels);
    for (size_t ch = 0; ch < num_inputs_; ++ch)
        reservoir_->InjectInput(ch, inputs[ch]);
    reservoir_->Step();
}

void ESN::Warmup(const float* inputs, size_t num_steps)
{
    for (size_t s = 0; s < num_steps; ++s)
        StepLive(inputs + s * num_inputs_);
}

void ESN::Run(const float* inputs, size_t num_steps)
{
    states_.resize((num_collected_ + num_steps) * n_);
    for (size_t s = 0; s < num_steps; ++s)
    {
        StepLive(inputs + s * num_inputs_);
        CopyLiveState(states_.data() + (num_collected_ + s) * n_);
    }
    num_collected_ += num_steps;
}

void ESN::ClearStates()
{
    states_.clear();
    states_.shrink_to_fit();
    num_collected_ = 0;
}

void ESN::ResetReservoirOnly()
{
    reservoir_->Reset();
}

void ESN::Train(const float* targets, size_t train_size)
{
    if (train_size > num_collected_)
        throw std::out_of_range(
            "ESN::Train: train_size (" + std::to_string(train_size) +
            ") exceeds num_collected (" + std::to_string(num_collected_) + ")");
    readout_.Train(ReadoutInput(0), targets, train_size);
}

void ESN::InitOnline(const float* warmup_inputs, size_t warmup_count)
{
    // The readout CNN is built eagerly in the ctor, so InitOnline now only runs
    // the reservoir warm-up to wash out the x(0) = 0 transient.
    Warmup(warmup_inputs, warmup_count);
}

void ESN::TrainLiveStep(float target_class, float lr, float weight_decay)
{
    CopyLiveState(scratch_state_.data());
    readout_.TrainOnlineStep(scratch_state_.data(),
                             static_cast<int>(target_class), lr, weight_decay);
}

void ESN::CopyLiveState(float* out) const
{
    const float* src = reservoir_->Outputs();
    std::memcpy(out, src, n_ * sizeof(float));
}

void ESN::TrainLiveBatch(const float* states, const int* targets,
                         size_t count, float lr)
{
    TrainLiveBatch(states, targets, count, lr, readout_.GetConfig().weight_decay);
}

void ESN::TrainLiveBatch(const float* states, const int* targets,
                         size_t count, float lr, float weight_decay)
{
    readout_.TrainOnlineBatch(states, targets, count, lr, weight_decay);
}

void ESN::TrainLiveStepRegression(const float* target, float lr,
                                  float weight_decay)
{
    CopyLiveState(scratch_state_.data());
    readout_.TrainOnlineStepRegression(scratch_state_.data(), target,
                                       lr, weight_decay);
}

void ESN::TrainLiveBatchRegression(const float* states, const float* targets,
                                   size_t count, float lr, float weight_decay)
{
    readout_.TrainOnlineBatchRegression(states, targets, count, lr, weight_decay);
}

float ESN::PredictRaw(size_t timestep) const
{
    if (timestep >= num_collected_)
        throw std::out_of_range(
            "ESN::PredictRaw: timestep (" + std::to_string(timestep) +
            ") >= num_collected (" + std::to_string(num_collected_) + ")");
    if (readout_.NumOutputs() != 1)
        throw std::invalid_argument(
            "ESN::PredictRaw(timestep): scalar prediction requires num_outputs == 1 "
            "(num_outputs=" + std::to_string(readout_.NumOutputs()) +
            "). Use PredictRaw(timestep, float*) for multi-output readouts.");
    return readout_.PredictRaw(ReadoutInput(timestep));
}

void ESN::PredictRaw(size_t timestep, float* output) const
{
    if (timestep >= num_collected_)
        throw std::out_of_range(
            "ESN::PredictRaw: timestep (" + std::to_string(timestep) +
            ") >= num_collected (" + std::to_string(num_collected_) + ")");
    readout_.PredictRaw(ReadoutInput(timestep), output);
}

float ESN::PredictLiveRaw() const
{
    if (readout_.NumOutputs() != 1)
        throw std::invalid_argument(
            "ESN::PredictLiveRaw(): scalar prediction requires num_outputs == 1 "
            "(num_outputs=" + std::to_string(readout_.NumOutputs()) +
            "). Use the PredictLiveRaw(float*) overload for multi-output readouts.");
    CopyLiveState(scratch_state_.data());
    return readout_.PredictRaw(scratch_state_.data());
}

void ESN::PredictLiveRaw(float* output) const
{
    CopyLiveState(scratch_state_.data());
    readout_.PredictRaw(scratch_state_.data(), output);
}

void ESN::PredictFromState(const float* state, float* output) const
{
    readout_.PredictRaw(state, output);
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

size_t ESN::NumOutputs() const
{
    return readout_.NumOutputs();
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
