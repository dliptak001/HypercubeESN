#include "EnsembleESN.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace
{
    // SplitMix64 finalizer — identical to the reservoir's internal mixer, so the
    // §5 seed derivation `seed_i = mix64(ensemble_seed ^ (GOLDEN*(i+1)))` is the
    // very mix the reservoir then re-fans into its own labelled substreams. A
    // 4-line pure function; duplicated rather than lifting a file-local helper
    // out of Reservoir.cpp.
    inline uint64_t mix64(uint64_t x)
    {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    // Golden-ratio odd constant — decorrelates the per-member offset before the
    // mix so adjacent member indices yield independent reservoir realizations.
    constexpr uint64_t GOLDEN = 0x9E3779B97F4A7C15ULL;

    // Role tag for the auxiliary-projection (W_u) RNG stream. XOR'd into a member's
    // derived seed before mixing so W_u is provably independent of the reservoir's
    // own substreams (which label with 0x100000001B3 * role, see Reservoir.cpp) —
    // a distinct large constant that is not one of those small multiples.
    constexpr uint64_t AUX_STREAM_TAG = 0xD1B54A32D192ED03ULL;
}

EnsembleESN::EnsembleESN(const EnsembleConfig& cfg)
    : M_(cfg.num_members),
      num_inputs_(cfg.base.reservoir.num_inputs),
      combine_(cfg.combine),
      lr_(cfg.learning_rate),
      wd_(cfg.weight_decay)
{
    if (M_ < 1)
        throw std::invalid_argument("EnsembleESN: num_members must be >= 1");
    if (combine_ == Combine::Median && M_ < 3)
        throw std::invalid_argument(
            "EnsembleESN: Combine::Median requires num_members >= 3 "
            "(a median is not meaningful below 3 members)");

    // One D, three roles (§3): output count == feedback-channel count == D > 0.
    const size_t d_readout = static_cast<size_t>(cfg.base.readout.num_outputs);
    const size_t d_feedback = cfg.base.reservoir.num_feedback_channels;
    if (d_feedback == 0)
        throw std::invalid_argument(
            "EnsembleESN: feedback is the whole point — "
            "base.reservoir.num_feedback_channels must be > 0");
    if (d_readout != d_feedback)
        throw std::invalid_argument(
            "EnsembleESN: the 'One D, three roles' identity is violated — "
            "base.readout.num_outputs must equal "
            "base.reservoir.num_feedback_channels");
    D_ = d_feedback;

    n_     = static_cast<size_t>(1) << cfg.base.reservoir.dim; // N = 2^dim, shared
    d_aux_ = cfg.aux_input_dim;

    // Build the M members from the single shared base config (§5), overriding
    // only the seed (derived) and verbose (silenced). When the aux channel is on,
    // build each member's fixed W_u projection from an independent RNG stream.
    esn_.reserve(M_);
    Wu_.resize(M_);
    for (size_t i = 0; i < M_; ++i)
    {
        const uint64_t member_seed = mix64(cfg.ensemble_seed ^ (GOLDEN * (i + 1)));

        ESNConfig member = cfg.base;
        member.reservoir.verbose = false;
        member.reservoir.spectral_radius += i*0.01;
        member.reservoir.lorentz_gamma = i*0.01;
        member.reservoir.seed = member_seed;
        esn_.push_back(std::make_unique<ESN>(member));

        if (d_aux_ > 0)
        {
            // U(-1,1) * aux_scaling with a 1/sqrt(d_aux) fan-in normalization (u_i is
            // a direct sum over d_aux terms). Stream seeded off the member seed via a
            // distinct aux tag, so W_u_i is independent of that member's reservoir
            // weights yet reproducible from ensemble_seed.
            std::mt19937_64 aux_rng(mix64(member_seed ^ AUX_STREAM_TAG));
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            const float scale = cfg.aux_scaling / std::sqrt(static_cast<float>(d_aux_));
            auto& W = Wu_[i];
            W.resize(n_ * d_aux_);
            for (float& w : W)
                w = static_cast<float>(dist(aux_rng)) * scale;
        }
    }

    // Pre-allocated per-step scratch (decision #5 — no per-tick heap traffic).
    y_flat_.assign(M_ * D_, 0.0f);
    phi_.assign(D_, 0.0f);
    median_scratch_.assign(M_, 0.0f);
    if (d_aux_ > 0)
        f_.assign(n_, 0.0f); // blended readout input; only needed when aux is on
}

void EnsembleESN::Consensus(float* c_out) const
{
    if (combine_ == Combine::Mean)
    {
        for (size_t c = 0; c < D_; ++c)
        {
            float s = 0.0f;
            for (size_t i = 0; i < M_; ++i)
                s += y_flat_[i * D_ + c];
            c_out[c] = s / static_cast<float>(M_);
        }
        return;
    }

    // Median, per channel. M is small; gather + nth_element.
    for (size_t c = 0; c < D_; ++c)
    {
        for (size_t i = 0; i < M_; ++i)
            median_scratch_[i] = y_flat_[i * D_ + c];
        const size_t mid = M_ / 2;
        std::nth_element(median_scratch_.begin(),
                         median_scratch_.begin() + mid,
                         median_scratch_.end());
        if (M_ & 1u)
        {
            c_out[c] = median_scratch_[mid];
        }
        else
        {
            // even M: average the two central order statistics. nth_element put
            // the upper-middle at `mid`; the lower-middle is the max below it.
            const float hi = median_scratch_[mid];
            const float lo = *std::max_element(median_scratch_.begin(),
                                               median_scratch_.begin() + mid);
            c_out[c] = 0.5f * (lo + hi);
        }
    }
}

void EnsembleESN::BlendedState(size_t i, const float* u_raw, float* out) const
{
    // out = x_i (the member's live reservoir state)...
    esn_[i]->CopyReservoirState(out);

    // ...then blend in the per-member projection u_i = W_u_i . u_raw, elementwise:
    // out[v] = k*x_i[v] + (1-k)*u_i[v].
    const float k = k_;
    const float one_minus_k = 1.0f - k;
    const float* W = Wu_[i].data();
    for (size_t v = 0; v < n_; ++v)
    {
        const float* wv = W + v * d_aux_;
        float u = 0.0f;
        for (size_t j = 0; j < d_aux_; ++j)
            u += wv[j] * u_raw[j];
        out[v] = k * out[v] + one_minus_k * u;
    }
}

void EnsembleESN::Step(const float* input, const float* target, float* c_out,
                       const float* u_raw)
{
    StepImpl(input, 0, target, c_out, u_raw); // stride 0: one shared input row
}

void EnsembleESN::StepPerMember(const float* inputs_MxI, const float* target,
                                float* c_out, const float* u_raw)
{
    StepImpl(inputs_MxI, num_inputs_, target, c_out, u_raw); // row i drives member i
}

void EnsembleESN::StepImpl(const float* inputs, const size_t input_stride,
                           const float* target, float* c_out, const float* u_raw)
{
    if (u_raw != nullptr && d_aux_ == 0)
        throw std::invalid_argument(
            "EnsembleESN: u_raw supplied but aux input not configured "
            "(aux_input_dim == 0); pass u_raw=nullptr or build with aux_input_dim > 0");
    const bool aux = (u_raw != nullptr);

    // 1. read every member's output y_i at its current readout input — the raw
    //    state x_i(t), or the blended F_i when an aux vector is supplied — straight
    //    into the pre-allocated y_flat_ slice (no per-tick allocation, decision #5).
    for (size_t i = 0; i < M_; ++i)
    {
        if (aux)
        {
            BlendedState(i, u_raw, f_.data());
            esn_[i]->PredictFromState(f_.data(), y_flat_.data() + i * D_);
        }
        else
        {
            esn_[i]->Predict(y_flat_.data() + i * D_);
        }
    }

    // 2. consensus c(t) = combine_i y_i  (also the ensemble's output).
    Consensus(c_out);

    // 3. train flag: fit only when a target is given.
    const bool train = (target != nullptr);

    // 4. for each member: (train) update readout on the same readout input read in
    //    step 1 (x_i(t) is unchanged — no step between the loops), then inject the
    //    scaled deviation phi_i = kappa*(y_i - c) and step to x_i(t+1).
    for (size_t i = 0; i < M_; ++i)
    {
        if (train)
        {
            if (aux)
            {
                BlendedState(i, u_raw, f_.data());
                esn_[i]->TrainStepFromState(f_.data(), target, lr_, wd_);
            }
            else
            {
                esn_[i]->TrainStep(target, lr_, wd_);
            }
        }

        const float* y_i = y_flat_.data() + i * D_;
        for (size_t c = 0; c < D_; ++c)
            phi_[c] = kappa_ * (y_i[c] - c_out[c]);

        esn_[i]->ReservoirStep(inputs + i * input_stride, phi_.data());
    }
}

void EnsembleESN::Predict(float* c_out, const float* u_raw)
{
    if (u_raw != nullptr && d_aux_ == 0)
        throw std::invalid_argument(
            "EnsembleESN::Predict: u_raw supplied but aux input not configured "
            "(aux_input_dim == 0); pass u_raw=nullptr or build with aux_input_dim > 0");

    for (size_t i = 0; i < M_; ++i)
    {
        if (u_raw != nullptr)
        {
            BlendedState(i, u_raw, f_.data());
            esn_[i]->PredictFromState(f_.data(), y_flat_.data() + i * D_);
        }
        else
        {
            esn_[i]->Predict(y_flat_.data() + i * D_);
        }
    }
    Consensus(c_out);
}

void EnsembleESN::MemberOutput(size_t i, float* out) const
{
    if (i >= M_)
        throw std::out_of_range("EnsembleESN::MemberOutput: member index out of range");
    std::copy(y_flat_.begin() + i * D_, y_flat_.begin() + (i + 1) * D_, out);
}

void EnsembleESN::AllMemberOutputs(float* out_MxD) const
{
    std::copy(y_flat_.begin(), y_flat_.end(), out_MxD);
}

EnsembleESN::State EnsembleESN::GetState() const
{
    State s;
    s.member_weights.reserve(M_);
    for (const auto& e : esn_)
        s.member_weights.push_back(e->GetReadoutState().weights);
    s.kappa = kappa_;
    s.mix = k_;
    return s;
}

void EnsembleESN::SetState(const State& s)
{
    if (s.member_weights.size() != M_)
        throw std::invalid_argument(
            "EnsembleESN::SetState: member_weights count (" +
            std::to_string(s.member_weights.size()) +
            ") != num_members (" + std::to_string(M_) + ")");

    for (size_t i = 0; i < M_; ++i)
    {
        ESN::ReadoutState rs;
        rs.weights = s.member_weights[i];
        rs.is_trained = true; // SetReadoutState no-ops unless this is set
        esn_[i]->SetReadoutState(rs);
    }

    kappa_ = s.kappa;
    k_ = s.mix;
}
