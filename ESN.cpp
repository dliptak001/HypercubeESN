#include "ESN.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

ReadoutConfig ESN::MakeReadoutConfig(const ESNConfig& cfg)
{
    ReadoutConfig rc = cfg.readout;

    if (rc.readout_slices == 0)
        throw std::invalid_argument("ESN: readout.readout_slices must be >= 1");
    if (rc.readout_slices > cfg.reservoir.history_depth)
        throw std::invalid_argument(
            "ESN: readout.readout_slices (" + std::to_string(rc.readout_slices) +
            ") exceeds reservoir.history_depth (" +
            std::to_string(cfg.reservoir.history_depth) +
            "); the delay line does not hold that many slices");

    const size_t blocks = rc.readout_slices + (rc.aux_input_dim > 0 ? 1u : 0u);
    if (!std::has_single_bit(blocks))
        throw std::invalid_argument(
            "ESN: readout_slices + (aux_input_dim > 0) must be a power of two (got " +
            std::to_string(blocks) +
            "); the readout input is a hypercube of B blocks of N");

    const size_t n = size_t{1} << cfg.reservoir.dim;
    if (rc.aux_input_dim > 0 && std::bit_ceil(rc.aux_input_dim) > n)
        throw std::invalid_argument(
            "ESN: aux_input_dim rounded up to a power of two exceeds N = 2^dim; the aux "
            "block cannot give each component its own subcube");

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
    readout_slices_ = cfg.readout.readout_slices;
    d_aux_          = cfg.readout.aux_input_dim;
    readout_blocks_ = readout_slices_ + (d_aux_ > 0 ? 1u : 0u);
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

void ESN::AssembleReadoutInput(const float* u_raw) const
{
    // Fail loud either way: an aux block must never be silently zeroed, and a stray
    // u_raw must never be silently ignored.
    if (d_aux_ > 0 && u_raw == nullptr)
        throw std::invalid_argument(
            "ESN: the readout has an auxiliary block (aux_input_dim > 0) but u_raw is "
            "null; supply the auxiliary input on every Predict/TrainStep");
    if (d_aux_ == 0 && u_raw != nullptr)
        throw std::invalid_argument(
            "ESN: u_raw supplied but the readout has no auxiliary block "
            "(aux_input_dim == 0)");

    float* const base = readout_input_.data();

    // Slice k is the state k steps back. SliceAt indexes by LOGICAL AGE, so this is
    // immune to the delay line's ring rotation — never read the history buffer raw.
    for (size_t k = 0; k < readout_slices_; ++k)
        std::memcpy(base + block_of_[k] * n_, reservoir_->SliceAt(k), n_ * sizeof(float));

    if (d_aux_ == 0) return;

    // Aux block: broadcast each component onto its own contiguous run of N/num_sub
    // vertices. A contiguous power-of-two run of a hypercube IS a subcube, so each
    // component occupies a locally constant region that a conv filter reads directly —
    // unlike a dense random projection, which scatters it incoherently.
    float* const blk = base + block_of_[readout_slices_] * n_;
    const size_t num_sub = std::bit_ceil(d_aux_);
    const size_t sub = n_ / num_sub;
    for (size_t j = 0; j < num_sub; ++j)
        std::fill(blk + j * sub, blk + (j + 1) * sub, j < d_aux_ ? u_raw[j] : 0.0f);
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
    if (d_aux_ > 0)
        throw std::logic_error(
            "ESN::ReservoirRun: the batch path has nowhere to take a per-step auxiliary "
            "input from (aux_input_dim > 0); drive this model online via "
            "TrainStep/Predict instead");

    if (clear_recorded)
    {
        states_.clear();        // keep capacity; the resize below refills it
        num_collected_ = 0;
    }
    states_.resize((num_collected_ + num_steps) * readout_width_);
    for (size_t s = 0; s < num_steps; ++s)
    {
        ReservoirStep(inputs + s * num_inputs_);
        AssembleReadoutInput(nullptr);
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

void ESN::TrainStep(const float* target, float lr, float weight_decay, const float* u_raw)
{
    AssembleReadoutInput(u_raw);
    readout_.TrainStep(readout_input_.data(), target, lr, weight_decay);
}

void ESN::CopyReservoirState(float* out) const
{
    const float* src = reservoir_->Outputs();
    std::memcpy(out, src, n_ * sizeof(float));
}

void ESN::CopyReadoutInput(float* out, const float* u_raw) const
{
    AssembleReadoutInput(u_raw);
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
    Predict(out.data(), nullptr);
    return out;
}

void ESN::Predict(float* out, const float* u_raw) const
{
    AssembleReadoutInput(u_raw);
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
    readout_.PredictRaw(readout_input, out.data());
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
