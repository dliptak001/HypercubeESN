#include "ESN.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Seam: readout_slices → HCNN start dim + block packing
// ---------------------------------------------------------------------------

ReadoutConfig ESN::MakeReadoutConfig(const ESNConfig& cfg)
{
    ReadoutConfig rc = cfg.readout;

    if (cfg.readout_slices == 0)
        throw std::invalid_argument("ESN: readout_slices must be >= 1");
    if (cfg.readout_slices > cfg.reservoir.history_depth)
        throw std::invalid_argument(
            "ESN: readout_slices (" + std::to_string(cfg.readout_slices) +
            ") exceeds reservoir.history_depth (" +
            std::to_string(cfg.reservoir.history_depth) +
            "); the delay line does not hold that many slices");

    const size_t blocks = cfg.readout_slices;
    if (!std::has_single_bit(blocks))
        throw std::invalid_argument(
            "ESN: readout_slices must be a power of two (got " +
            std::to_string(blocks) +
            "); the readout input is a hypercube of B blocks of N");

    // B blocks of N vertices → start_DIM = reservoir_DIM + log2(B).
    rc.dim = cfg.reservoir.dim + static_cast<size_t>(std::countr_zero(blocks));
    return rc;
}

std::vector<size_t> ESN::MakeBlockMap(const size_t blocks)
{
    std::vector<size_t> map(blocks);

    // B ≤ 2: no 2-bit-apart pair exists; identity is fine.
    if (blocks <= 2)
    {
        for (size_t s = 0; s < blocks; ++s)
            map[s] = s;
        return map;
    }

    // HCNN conv: 1-bit neighbors only, no self term. Two blocks fall in one
    // filter's reach iff their indices differ in two bits. Map consecutive
    // logical slots onto (rep, rep ^ 0b11) so {age0, age1} share a filter, then
    // {age2, age3}, …  `rep` walks indices with low bits 00 or 01 → full permutation.
    for (size_t i = 0; i < blocks / 2; ++i)
    {
        const size_t rep = ((i >> 1) << 2) | (i & 1u);
        map[2 * i] = rep;
        map[2 * i + 1] = rep ^ 3u;
    }
    return map;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

ESN::ESN(const ESNConfig& cfg)
    : reservoir_(Reservoir::Create(cfg.reservoir)),
      readout_(MakeReadoutConfig(cfg))
{
    n_ = reservoir_->Size();
    num_inputs_ = cfg.reservoir.num_inputs;
    readout_slices_ = cfg.readout_slices;
    readout_blocks_ = readout_slices_;
    readout_width_ = n_ * readout_blocks_;
    block_of_ = MakeBlockMap(readout_blocks_);

    esn_config_ = cfg;
    // Caller leaves readout.dim at 0; store the derived HCNN start dim for GetConfig().
    esn_config_.readout.dim = readout_.GetConfig().dim;

    readout_input_.resize(readout_width_);
}

// ---------------------------------------------------------------------------
// Readout-input assembly
// ---------------------------------------------------------------------------

size_t ESN::ReadoutBlockOf(const size_t slot) const
{
    if (slot >= readout_blocks_)
        throw std::out_of_range(
            "ESN::ReadoutBlockOf: slot (" + std::to_string(slot) +
            ") >= block count (" + std::to_string(readout_blocks_) + ")");
    return block_of_[slot];
}

void ESN::AssembleReadoutInput() const
{
    float* const base = readout_input_.data();
    // Logical age k → SliceAt(k); physical placement via block_of_.
    for (size_t k = 0; k < readout_slices_; ++k)
        std::memcpy(base + block_of_[k] * n_, reservoir_->SliceAt(k),
                    n_ * sizeof(float));
}

// ---------------------------------------------------------------------------
// Reservoir driving
// ---------------------------------------------------------------------------

void ESN::ReservoirStep(const float* inputs, const float* external_feedback)
{
    // External feedback is caller-owned (policy lives outside ESN). Stage raw
    // values on D channels; reservoir scales via external_feedback_scaling (outside SR).
    if (external_feedback != nullptr)
    {
        // D==0 + non-null must not silently no-op.
        if (esn_config_.reservoir.num_external_feedback_channels == 0)
            throw std::invalid_argument(
                "ESN::ReservoirStep: external_feedback supplied but not configured "
                "(num_external_feedback_channels == 0); pass nullptr to skip");
        reservoir_->InjectExternalFeedback(
            external_feedback,
            esn_config_.reservoir.num_external_feedback_channels);
    }

    for (size_t ch = 0; ch < num_inputs_; ++ch)
        reservoir_->InjectInput(ch, inputs[ch]);
    reservoir_->Step();
}

void ESN::ReservoirStep(std::span<const float> inputs, std::span<const float> external_feedback)
{
    if (inputs.size() != num_inputs_)
        throw std::invalid_argument(
            "ESN::ReservoirStep: inputs.size() (" + std::to_string(inputs.size()) +
            ") must equal NumInputs() (" + std::to_string(num_inputs_) + ")");
    const size_t D = esn_config_.reservoir.num_external_feedback_channels;
    if (!external_feedback.empty())
    {
        if (D == 0)
            throw std::invalid_argument(
                "ESN::ReservoirStep: external_feedback provided but "
                "num_external_feedback_channels == 0");
        if (external_feedback.size() != D)
            throw std::invalid_argument(
                "ESN::ReservoirStep: external_feedback.size() (" +
                std::to_string(external_feedback.size()) +
                ") must equal NumExternalFeedbackChannels() (" + std::to_string(D) + ")");
        ReservoirStep(inputs.data(), external_feedback.data());
    }
    else
    {
        ReservoirStep(inputs.data(), nullptr);
    }
}

void ESN::ReservoirWarmup(const float* inputs, size_t num_steps)
{
    for (size_t s = 0; s < num_steps; ++s)
        ReservoirStep(inputs + s * num_inputs_);
}

void ESN::ReservoirWarmup(std::span<const float> inputs)
{
    if (num_inputs_ == 0 || inputs.size() % num_inputs_ != 0)
        throw std::invalid_argument(
            "ESN::ReservoirWarmup: inputs.size() (" + std::to_string(inputs.size()) +
            ") must be a multiple of NumInputs() (" + std::to_string(num_inputs_) + ")");
    ReservoirWarmup(inputs.data(), inputs.size() / num_inputs_);
}

void ESN::ReservoirRun(const float* inputs, size_t num_steps, bool clear_recorded)
{
    if (clear_recorded)
    {
        states_.clear(); // capacity retained; resize below refills
        num_collected_ = 0;
    }
    states_.resize((num_collected_ + num_steps) * readout_width_);
    for (size_t s = 0; s < num_steps; ++s)
    {
        ReservoirStep(inputs + s * num_inputs_);
        AssembleReadoutInput();
        std::memcpy(states_.data() + (num_collected_ + s) * readout_width_,
                    readout_input_.data(), readout_width_ * sizeof(float));
    }
    num_collected_ += num_steps;
}

void ESN::ReservoirRun(std::span<const float> inputs, bool clear_recorded)
{
    if (num_inputs_ == 0 || inputs.size() % num_inputs_ != 0)
        throw std::invalid_argument(
            "ESN::ReservoirRun: inputs.size() (" + std::to_string(inputs.size()) +
            ") must be a multiple of NumInputs() (" + std::to_string(num_inputs_) + ")");
    ReservoirRun(inputs.data(), inputs.size() / num_inputs_, clear_recorded);
}

void ESN::ReservoirClear()
{
    reservoir_->Clear();
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

void ESN::Train(const float* targets, size_t train_size)
{
    if (train_size > num_collected_)
        throw std::out_of_range(
            "ESN::Train: train_size (" + std::to_string(train_size) +
            ") exceeds num_collected (" + std::to_string(num_collected_) + ")");
    readout_.Train(ReadoutInput(0), targets, train_size);
}

void ESN::Train(std::span<const float> targets, size_t train_size)
{
    const bool cls = esn_config_.readout.task == ReadoutTask::Classification;
    const size_t expected = cls ? train_size : train_size * readout_.NumOutputs();
    if (targets.size() != expected)
        throw std::invalid_argument(
            "ESN::Train: targets.size() (" + std::to_string(targets.size()) +
            ") must equal " + std::to_string(expected) +
            (cls ? " (train_size class indices)" : " (train_size * NumOutputs())"));
    Train(targets.data(), train_size);
}

void ESN::TrainStep(const float* target, float lr, float weight_decay)
{
    AssembleReadoutInput();
    readout_.TrainStep(readout_input_.data(), target, lr, weight_decay);
}

void ESN::TrainStep(std::span<const float> target, float lr, float weight_decay)
{
    const bool cls = esn_config_.readout.task == ReadoutTask::Classification;
    const size_t expected = cls ? 1u : readout_.NumOutputs();
    if (target.size() != expected)
        throw std::invalid_argument(
            "ESN::TrainStep: target.size() (" + std::to_string(target.size()) +
            ") must equal " + std::to_string(expected));
    TrainStep(target.data(), lr, weight_decay);
}

void ESN::TrainStepBatch(const float* readout_inputs, const float* targets,
                         size_t count, float lr, float weight_decay)
{
    readout_.TrainStepBatch(readout_inputs, targets, count, lr, weight_decay);
}

void ESN::CopyReservoirState(float* out) const
{
    const float* src = reservoir_->Outputs();
    std::memcpy(out, src, n_ * sizeof(float));
}

void ESN::CopyReservoirState(std::span<float> out) const
{
    if (out.size() != n_)
        throw std::invalid_argument(
            "ESN::CopyReservoirState: out.size() (" + std::to_string(out.size()) +
            ") must equal ReservoirNeuronCount() (" + std::to_string(n_) + ")");
    CopyReservoirState(out.data());
}

void ESN::CopyReadoutInput(float* out) const
{
    AssembleReadoutInput();
    std::memcpy(out, readout_input_.data(), readout_width_ * sizeof(float));
}

void ESN::CopyReadoutInput(std::span<float> out) const
{
    if (out.size() != readout_width_)
        throw std::invalid_argument(
            "ESN::CopyReadoutInput: out.size() (" + std::to_string(out.size()) +
            ") must equal ReadoutInputWidth() (" + std::to_string(readout_width_) + ")");
    CopyReadoutInput(out.data());
}

// ---------------------------------------------------------------------------
// Prediction & metrics
// ---------------------------------------------------------------------------

std::vector<float> ESN::Predict() const
{
    std::vector<float> out(readout_.NumOutputs());
    Predict(out.data());
    return out;
}

void ESN::Predict(float* out) const
{
    AssembleReadoutInput();
    readout_.PredictRaw(readout_input_.data(), out);
}

void ESN::Predict(std::span<float> out) const
{
    if (out.size() != readout_.NumOutputs())
        throw std::invalid_argument(
            "ESN::Predict: out.size() (" + std::to_string(out.size()) +
            ") must equal NumOutputs() (" + std::to_string(readout_.NumOutputs()) + ")");
    Predict(out.data());
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

std::vector<float> ESN::PredictFromState(const float* readout_input) const
{
    std::vector<float> out(readout_.NumOutputs());
    PredictFromState(readout_input, out.data());
    return out;
}

void ESN::PredictFromState(const float* readout_input, float* out) const
{
    readout_.PredictRaw(readout_input, out);
}

std::vector<float> ESN::PredictFromReadoutInput(std::span<const float> readout_input) const
{
    if (readout_input.size() != readout_width_)
        throw std::invalid_argument(
            "ESN::PredictFromReadoutInput: size (" + std::to_string(readout_input.size()) +
            ") must equal ReadoutInputWidth() (" + std::to_string(readout_width_) + ")");
    return PredictFromState(readout_input.data());
}

void ESN::PredictFromReadoutInput(std::span<const float> readout_input, std::span<float> out) const
{
    if (readout_input.size() != readout_width_)
        throw std::invalid_argument(
            "ESN::PredictFromReadoutInput: input size (" + std::to_string(readout_input.size()) +
            ") must equal ReadoutInputWidth() (" + std::to_string(readout_width_) + ")");
    if (out.size() != readout_.NumOutputs())
        throw std::invalid_argument(
            "ESN::PredictFromReadoutInput: out.size() (" + std::to_string(out.size()) +
            ") must equal NumOutputs() (" + std::to_string(readout_.NumOutputs()) + ")");
    PredictFromState(readout_input.data(), out.data());
}

namespace
{
void RequireFullTargetBuffer(const char* api, std::span<const float> targets,
                             size_t start, size_t count, size_t stride)
{
    const size_t need = (start + count) * stride;
    if (targets.size() < need)
        throw std::invalid_argument(
            std::string(api) + ": targets.size() (" + std::to_string(targets.size()) +
            ") < (start+count)*stride (" + std::to_string(start) + "+" +
            std::to_string(count) + ")*" + std::to_string(stride) + " = " +
            std::to_string(need) +
            ". Pass a buffer covering [0, start+count), not a window slice "
            "(or use the FromWindow overload).");
}

void RequireWindowTargetBuffer(const char* api, std::span<const float> window,
                               size_t count, size_t stride)
{
    const size_t need = count * stride;
    if (window.size() != need)
        throw std::invalid_argument(
            std::string(api) + ": window.size() (" + std::to_string(window.size()) +
            ") must equal count*stride (" + std::to_string(count) + "*" +
            std::to_string(stride) + " = " + std::to_string(need) + ")");
}
} // namespace

double ESN::R2(const float* targets, size_t start, size_t count) const
{
    if (start + count > num_collected_)
        throw std::out_of_range(
            "ESN::R2: start + count (" + std::to_string(start + count) +
            ") > num_collected (" + std::to_string(num_collected_) + ")");
    return readout_.R2(ReadoutInput(start),
                       targets + start * readout_.NumOutputs(), count);
}

double ESN::R2(std::span<const float> targets, size_t start, size_t count) const
{
    if (start + count > num_collected_)
        throw std::out_of_range(
            "ESN::R2: start + count (" + std::to_string(start + count) +
            ") > num_collected (" + std::to_string(num_collected_) + ")");
    // R² is multi-output regression layout: stride = NumOutputs() per sample.
    RequireFullTargetBuffer("ESN::R2", targets, start, count, readout_.NumOutputs());
    return R2(targets.data(), start, count);
}

double ESN::R2FromWindow(std::span<const float> targets_window,
                         size_t start, size_t count) const
{
    if (start + count > num_collected_)
        throw std::out_of_range(
            "ESN::R2FromWindow: start + count (" + std::to_string(start + count) +
            ") > num_collected (" + std::to_string(num_collected_) + ")");
    RequireWindowTargetBuffer("ESN::R2FromWindow", targets_window, count, readout_.NumOutputs());
    return readout_.R2(ReadoutInput(start), targets_window.data(), count);
}

double ESN::NRMSE(const float* targets, size_t start, size_t count) const
{
    if (start + count > num_collected_)
        throw std::out_of_range(
            "ESN::NRMSE: start + count (" + std::to_string(start + count) +
            ") > num_collected (" + std::to_string(num_collected_) + ")");
    if (count == 0)
        return 0.0;

    const size_t K = readout_.NumOutputs();
    const float* tgt = targets + start * K;

    std::vector<float> preds(count * K);
    for (size_t s = 0; s < count; ++s)
        readout_.PredictRaw(ReadoutInput(start + s), preds.data() + s * K);

    // Mean over outputs of (RMSE_k / std_k).
    double nrmse_sum = 0.0;
    for (size_t k = 0; k < K; ++k)
    {
        double mean = 0.0;
        for (size_t s = 0; s < count; ++s)
            mean += tgt[s * K + k];
        mean /= static_cast<double>(count);

        double var = 0.0, mse_k = 0.0;
        for (size_t s = 0; s < count; ++s)
        {
            const double y = tgt[s * K + k];
            const double yh = preds[s * K + k];
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

double ESN::NRMSE(std::span<const float> targets, size_t start, size_t count) const
{
    if (start + count > num_collected_)
        throw std::out_of_range(
            "ESN::NRMSE: start + count (" + std::to_string(start + count) +
            ") > num_collected (" + std::to_string(num_collected_) + ")");
    RequireFullTargetBuffer("ESN::NRMSE", targets, start, count, readout_.NumOutputs());
    return NRMSE(targets.data(), start, count);
}

double ESN::NRMSEFromWindow(std::span<const float> targets_window,
                            size_t start, size_t count) const
{
    if (start + count > num_collected_)
        throw std::out_of_range(
            "ESN::NRMSEFromWindow: start + count (" + std::to_string(start + count) +
            ") > num_collected (" + std::to_string(num_collected_) + ")");
    if (count == 0)
        return 0.0;
    const size_t K = readout_.NumOutputs();
    // NRMSE is regression-oriented (uses K-wide targets). Match pointer NRMSE layout.
    RequireWindowTargetBuffer("ESN::NRMSEFromWindow", targets_window, count, K);
    // Reuse core path by presenting a fake "full" buffer that starts at index 0 of window:
    // pointer NRMSE indexes targets + start*K, so pass window.data() - start*K is UB.
    // Implement inline with window as tgt base (same loop as NRMSE).
    const float* tgt = targets_window.data();
    std::vector<float> preds(count * K);
    for (size_t s = 0; s < count; ++s)
        readout_.PredictRaw(ReadoutInput(start + s), preds.data() + s * K);
    double nrmse_sum = 0.0;
    for (size_t k = 0; k < K; ++k)
    {
        double mean = 0.0;
        for (size_t s = 0; s < count; ++s)
            mean += tgt[s * K + k];
        mean /= static_cast<double>(count);
        double var = 0.0, mse_k = 0.0;
        for (size_t s = 0; s < count; ++s)
        {
            const double y = tgt[s * K + k];
            const double yh = preds[s * K + k];
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

double ESN::Accuracy(std::span<const float> labels, size_t start, size_t count) const
{
    if (start + count > num_collected_)
        throw std::out_of_range(
            "ESN::Accuracy: start + count (" + std::to_string(start + count) +
            ") > num_collected (" + std::to_string(num_collected_) + ")");
    RequireFullTargetBuffer("ESN::Accuracy", labels, start, count, 1);
    return Accuracy(labels.data(), start, count);
}

double ESN::AccuracyFromWindow(std::span<const float> labels_window,
                               size_t start, size_t count) const
{
    if (start + count > num_collected_)
        throw std::out_of_range(
            "ESN::AccuracyFromWindow: start + count (" + std::to_string(start + count) +
            ") > num_collected (" + std::to_string(num_collected_) + ")");
    RequireWindowTargetBuffer("ESN::AccuracyFromWindow", labels_window, count, 1);
    return readout_.Accuracy(ReadoutInput(start), labels_window.data(), count);
}

// ---------------------------------------------------------------------------
// Config & persistence
// ---------------------------------------------------------------------------

ESNConfig ESN::GetConfig() const
{
    return esn_config_;
}

ESN::ReadoutState ESN::GetReadoutState() const
{
    ReadoutState s;
    // IsTrained() is true once the HCNN exists (construction), not “has trained.”
    s.is_trained = readout_.IsTrained();
    const auto& w = readout_.Weights();
    s.weights.assign(w.begin(), w.end());
    return s;
}

void ESN::SetReadoutState(const ReadoutState& state, ReadoutLoadMode mode)
{
    if (!state.is_trained)
        return;
    readout_.SetState(state.weights, mode);
}

void ESN::SaveReadoutHcnnModel(const std::string& path_stem) const
{
    readout_.SaveHcnnModel(path_stem);
}

void ESN::LoadReadoutHcnnModel(const std::string& path_stem, ReadoutLoadMode mode)
{
    readout_.LoadHcnnModel(path_stem, mode);
}

std::string ESN::ReadoutArchSummary() const
{
    // Fixed reservoir weights: N·DIM·(M + drive_blocks), drive_blocks = 1 + [ext].
    // HCNN size depends on start_DIM = dim + log2(B), not on M alone.
    const ReservoirConfig rc = reservoir_->GetConfig();
    const size_t n = reservoir_->Size();
    const size_t dim = reservoir_->Dim();
    const size_t M = rc.history_depth;
    const size_t drive_blocks =
        1u + (rc.num_external_feedback_channels > 0 ? 1u : 0u);
    const size_t res_weights = n * dim * (M + drive_blocks);

    const size_t B = readout_blocks_;
    const size_t hcnn_dim = readout_.GetConfig().dim;
    const size_t hcnn_N = size_t{1} << hcnn_dim;

    std::ostringstream os;
    os << "Reservoir: DIM=" << dim << "  N=" << n << "  M=" << M
       << "  weights(fixed)=" << res_weights
       << "  [=N*DIM*(M+drives)=" << n << "*" << dim << "*(" << M << "+"
       << drive_blocks << ")]\n";
    os << "HCNN input: readout_slices=" << readout_slices_ << "  B=" << B
       << " blocks x N=" << n << "  -> start_DIM=" << hcnn_dim
       << "  N_hcnn=" << hcnn_N << "\n";
    os << "  (start_DIM = reservoir_DIM + log2(B) = " << dim << " + "
       << (hcnn_dim - dim) << "; M does not change HCNN size)\n";
    os << readout_.ArchSummary();
    return os.str();
}

// ---------------------------------------------------------------------------
// Recorded-buffer helpers
// ---------------------------------------------------------------------------

const float* ESN::ReadoutInput(size_t timestep) const
{
    return states_.data() + timestep * readout_width_;
}

std::vector<float> ESN::ReadoutStates(size_t start, size_t count) const
{
    std::vector<float> buf(count * readout_width_);
    std::memcpy(buf.data(), states_.data() + start * readout_width_,
                count * readout_width_ * sizeof(float));
    return buf;
}

std::vector<float> ESN::CollectedStates() const
{
    return ReadoutStates(0, num_collected_);
}
