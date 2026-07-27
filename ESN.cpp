#include "ESN.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

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

    // The readout consumes B blocks of N vertices each, so its hypercube dimension is
    // the reservoir's dim plus log2(B).
    rc.dim = cfg.reservoir.dim + static_cast<size_t>(std::countr_zero(blocks));
    return rc;
}

std::vector<size_t> ESN::MakeBlockMap(const size_t blocks)
{
    std::vector<size_t> map(blocks);

    // With one or two blocks no pair of indices differs in two bits; order is moot.
    if (blocks <= 2)
    {
        for (size_t s = 0; s < blocks; ++s) map[s] = s;
        return map;
    }

    // A hypercube conv filter gathers only single-bit-flip neighbours and carries no
    // self term, so it sees the blocks one bit from its own but never its own centre.
    // Two blocks therefore land in a single filter's reach exactly when their indices
    // differ in TWO bits. Pair consecutive slots onto (rep, rep ^ 0b11) — always a
    // two-bit difference — so {age 0, age 1}, the velocity, is one filter away, then
    // {age 2, age 3}, and so on. `rep` walks the blocks whose low two bits are 00 or
    // 01, which makes the whole map a permutation of [0, blocks).
    for (size_t i = 0; i < blocks / 2; ++i)
    {
        const size_t rep = ((i >> 1) << 2) | (i & 1u);
        map[2 * i]     = rep;
        map[2 * i + 1] = rep ^ 3u;
    }
    return map;
}

ESN::ESN(const ESNConfig& cfg)
    : reservoir_(Reservoir::Create(cfg.reservoir)),
      readout_(MakeReadoutConfig(cfg))
{
    n_              = reservoir_->Size();
    num_inputs_     = cfg.reservoir.num_inputs;
    readout_slices_ = cfg.readout_slices;
    readout_blocks_ = readout_slices_;
    readout_width_  = n_ * readout_blocks_;
    block_of_       = MakeBlockMap(readout_blocks_);

    esn_config_ = cfg;
    // The user leaves cfg.readout.dim at 0 ("do not set"); reflect the value the
    // readout was actually built with so GetConfig() doesn't report a stale 0.
    esn_config_.readout.dim = readout_.GetConfig().dim;

    readout_input_.resize(readout_width_);
}

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

    // Slice k is the state k steps back. SliceAt indexes by LOGICAL AGE, so this is
    // immune to the delay line's ring rotation — never read the history buffer raw.
    for (size_t k = 0; k < readout_slices_; ++k)
        std::memcpy(base + block_of_[k] * n_, reservoir_->SliceAt(k), n_ * sizeof(float));
}

void ESN::ReservoirStep(const float* inputs, const float* external_feedback)
{
    // External feedback: caller-owned closed-loop drive. Stage on the D channels
    // (raw, no clamp) through the reservoir's external-feedback port (own weights
    // + external_feedback_scaling, outside the SR rescale). The ESN never invents
    // these values.
    if (external_feedback != nullptr)
    {
        // Guard: with D == 0 InjectExternalFeedback(ptr, 0) would no-op and silently
        // degrade to no external drive. Fail loud instead.
        if (esn_config_.reservoir.num_external_feedback_channels == 0)
            throw std::invalid_argument(
                "ESN::ReservoirStep: external_feedback supplied but not configured "
                "(num_external_feedback_channels == 0); pass nullptr to skip");
        reservoir_->InjectExternalFeedback(
            external_feedback, esn_config_.reservoir.num_external_feedback_channels);
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
    AssembleReadoutInput();
    readout_.TrainStep(readout_input_.data(), target, lr, weight_decay);
}

void ESN::CopyReservoirState(float* out) const
{
    const float* src = reservoir_->Outputs();
    std::memcpy(out, src, n_ * sizeof(float));
}

void ESN::CopyReadoutInput(float* out) const
{
    AssembleReadoutInput();
    std::memcpy(out, readout_input_.data(), readout_width_ * sizeof(float));
}

void ESN::TrainStepBatch(const float* readout_inputs, const float* targets,
                         size_t count, float lr, float weight_decay)
{
    readout_.TrainStepBatch(readout_inputs, targets, count, lr, weight_decay);
}

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

void ESN::SetReadoutState(const ReadoutState& state, ReadoutLoadMode mode)
{
    if (!state.is_trained) return;
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
    // Reservoir vtx_weight_ layout: N·DIM·(M + drive_blocks), drive_blocks =
    // 1 (input) + [ext-fb]. Matches Reservoir construction (docs/Reservoir.md).
    // These weights are fixed (not trained). M scales the recurrent bank only.
    // HCNN size is independent of M unless readout_slices grow start-DIM.
    const ReservoirConfig rc = reservoir_->GetConfig();
    const size_t n = reservoir_->Size();
    const size_t dim = reservoir_->Dim();
    const size_t M = rc.history_depth;
    const size_t drive_blocks = 1u
        + (rc.num_external_feedback_channels > 0 ? 1u : 0u);
    const size_t res_weights = n * dim * (M + drive_blocks);

    // HCNN start-DIM = reservoir DIM + log2(B), B = readout_slices.
    // Not the same number as reservoir DIM when B > 1 (e.g. slices=4 -> +2).
    const size_t B = readout_blocks_;
    const size_t hcnn_dim = readout_.GetConfig().dim;
    const size_t hcnn_N = size_t{1} << hcnn_dim;

    std::ostringstream os;
    os << "Reservoir: DIM=" << dim << "  N=" << n
       << "  M=" << M
       << "  weights(fixed)=" << res_weights
       << "  [=N*DIM*(M+drives)=" << n << "*" << dim << "*(" << M << "+"
       << drive_blocks << ")]\n";
    os << "HCNN input: readout_slices=" << readout_slices_
       << "  B=" << B << " blocks x N=" << n
       << "  -> start_DIM=" << hcnn_dim << "  N_hcnn=" << hcnn_N << "\n";
    os << "  (start_DIM = reservoir_DIM + log2(B) = " << dim << " + "
       << (hcnn_dim - dim) << "; M does not change HCNN size)\n";
    os << readout_.ArchSummary();
    return os.str();
}

// ---------------------------------------------------------------
//  Collected-state access helpers
// ---------------------------------------------------------------

const float* ESN::ReadoutInput(size_t timestep) const
{
    return states_.data() + timestep * readout_width_;
}

std::vector<float> ESN::ReadoutStates(size_t start, size_t count) const
{
    std::vector<float> buf(count * readout_width_);
    std::memcpy(buf.data(),
                states_.data() + start * readout_width_,
                count * readout_width_ * sizeof(float));
    return buf;
}

std::vector<float> ESN::CollectedStates() const
{
    return ReadoutStates(0, num_collected_);
}
