#include "ESN.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

ESN::ReadoutGeometry
ESN::ComputeReadoutGeometry(size_t dim, float of)
{
    if (!(of > 0.0f && of <= 1.0f))
        throw std::invalid_argument("ESN: output_fraction must be in (0.0, 1.0]");

    // Guard the N = 2^dim shift below: dim >= 64 would be undefined behavior
    // (shift >= width of unsigned long long). Reservoir::Create enforces the
    // authoritative [5, 16] range; this only keeps the shift well-defined for a
    // config that hasn't reached the reservoir ctor yet.
    if (dim >= 64)
        throw std::invalid_argument(
            "ESN: reservoir.dim too large to form N = 2^dim (Reservoir requires 5-16)");

    const size_t n = 1ULL << dim;
    const size_t M = std::max<size_t>(1, static_cast<size_t>(std::round(n * of)));
    const size_t stride = std::max<size_t>(1, n / M);

    if ((stride & (stride - 1)) != 0)
        throw std::invalid_argument(
            "ESN: output_fraction must yield a power-of-2 stride "
            "(1, 2, 4, 8, 16, ...). Use output_fraction in "
            "{1.0, 0.5, 0.25, 0.125, 0.0625, ...}.");

    const size_t verts = (n + stride - 1) / stride;
    size_t d = 0;
    for (size_t k = verts; k > 1; k >>= 1) ++d;
    return {stride, verts, d};
}

ReadoutConfig ESN::MakeReadoutConfig(const ESNConfig& cfg, const ReadoutGeometry& geo)
{
    ReadoutConfig rc = cfg.readout;
    rc.dim = geo.dim;
    return rc;
}

ESN::ESN(const ESNConfig& cfg)
    // Compute geometry once here (this also validates output_fraction, before
    // any member is constructed), then hand it to the delegating-target ctor.
    : ESN(cfg, ComputeReadoutGeometry(cfg.reservoir.dim, cfg.output_fraction))
{
}

ESN::ESN(const ESNConfig& cfg, const ReadoutGeometry& geo)
    : reservoir_(Reservoir::Create(cfg.reservoir)),
      readout_(MakeReadoutConfig(cfg, geo))
{
    n_                = reservoir_->Size();
    num_inputs_       = cfg.reservoir.num_inputs;
    esn_config_       = cfg;

    // cfg.output_fraction is the user-requested value; the readout sees
    // num_output_verts_ stride-selected vertices. Query NumOutputVerts() for
    // the effective readout-side feature count.
    output_stride_    = geo.output_stride;
    num_output_verts_ = geo.num_output_verts;
    scratch_subsampled_.resize(num_output_verts_);
}

void ESN::Warmup(const float* inputs, size_t num_steps)
{
    const size_t K = num_inputs_;
    for (size_t s = 0; s < num_steps; ++s)
    {
        for (size_t ch = 0; ch < K; ++ch)
            reservoir_->InjectInput(ch, inputs[s * K + ch]);
        reservoir_->Step();
    }
}

void ESN::Run(const float* inputs, size_t num_steps)
{
    const size_t K = num_inputs_;
    const size_t M = num_output_verts_;
    states_.resize((num_collected_ + num_steps) * M);
    for (size_t s = 0; s < num_steps; ++s)
    {
        for (size_t ch = 0; ch < K; ++ch)
            reservoir_->InjectInput(ch, inputs[s * K + ch]);
        reservoir_->Step();

        CopyLiveState(states_.data() + (num_collected_ + s) * M);
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
    Warmup(warmup_inputs, warmup_count);
    readout_.InitOnline();
}

void ESN::TrainLiveStep(float target_class, float lr, float weight_decay)
{
    CopyLiveState(scratch_subsampled_.data());
    readout_.TrainOnlineStep(scratch_subsampled_.data(),
                             static_cast<int>(target_class), lr, weight_decay);
}

void ESN::CopyLiveState(float* out) const
{
    const float* src = reservoir_->Outputs();
    size_t j = 0;
    for (size_t v = 0; v < n_; v += output_stride_)
        out[j++] = src[v];
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
    CopyLiveState(scratch_subsampled_.data());
    readout_.TrainOnlineStepRegression(scratch_subsampled_.data(), target,
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
            "). Use PredictLiveRaw(float*) / predict_live_raw_multi instead.");
    CopyLiveState(scratch_subsampled_.data());
    return readout_.PredictRaw(scratch_subsampled_.data());
}

void ESN::PredictLiveRaw(float* output) const
{
    CopyLiveState(scratch_subsampled_.data());
    readout_.PredictRaw(scratch_subsampled_.data(), output);
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
    // IsTrained() is set by both batch Train() and InitOnline() (online),
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
//  HCNN sub-hypercube subsampling helpers
// ---------------------------------------------------------------

const float* ESN::ReadoutInput(size_t timestep) const
{
    return states_.data() + timestep * num_output_verts_;
}

std::vector<float> ESN::ReadoutStates(size_t start, size_t count) const
{
    std::vector<float> buf(count * num_output_verts_);
    std::memcpy(buf.data(),
                states_.data() + start * num_output_verts_,
                count * num_output_verts_ * sizeof(float));
    return buf;
}

std::vector<float> ESN::SelectedStates() const
{
    return ReadoutStates(0, num_collected_);
}
