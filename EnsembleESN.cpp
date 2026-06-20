#include "EnsembleESN.h"

#include <algorithm>
#include <cmath>
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
}

EnsembleESN::EnsembleESN(const EnsembleConfig& cfg)
    : M_(cfg.num_members),
      num_inputs_(cfg.base.reservoir.num_inputs),
      combine_(cfg.combine),
      lr_(cfg.lr),
      wd_(cfg.weight_decay),
      kappa_(cfg.kappa_start),
      kappa_target_(cfg.kappa_target),
      kappa_ramp_rate_(cfg.kappa_ramp_rate),
      gate_threshold_(cfg.gate_threshold),
      gate_err_ema_alpha_(cfg.gate_err_ema_alpha),
      resequence_washout_(cfg.resequence_washout)
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

    if (gate_err_ema_alpha_ <= 0.0f || gate_err_ema_alpha_ > 1.0f)
        throw std::invalid_argument(
            "EnsembleESN: gate_err_ema_alpha must be in (0, 1]");

    // Build the M members from the single shared base config (§5), overriding
    // only the seed (derived) and verbose (silenced).
    esn_.reserve(M_);
    for (size_t i = 0; i < M_; ++i)
    {
        ESNConfig member = cfg.base;
        member.reservoir.verbose = false;
        member.reservoir.seed = mix64(cfg.ensemble_seed ^ (GOLDEN * (i + 1)));
        esn_.push_back(std::make_unique<ESN>(member));
    }

    // The initial washout (§7.1): suppress the readout update for W steps.
    washout_remaining_ = cfg.washout;

    // Pre-allocated per-step scratch (decision #5 — no per-tick heap traffic).
    y_flat_.assign(M_ * D_, 0.0f);
    phi_.assign(D_, 0.0f);
    median_scratch_.assign(M_, 0.0f);
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

void EnsembleESN::Step(const float* input, const float* target, float* c_out)
{
    // 1. read every member's output y_i at its current state x_i(t).
    for (size_t i = 0; i < M_; ++i)
        esn_[i]->PredictLiveRaw(y_flat_.data() + i * D_);

    // 2. consensus c(t) = combine_i y_i  (also the ensemble's output).
    Consensus(c_out);

    // 3. train flag: fit only past the washout, and only when a target is given.
    const bool train = (target != nullptr) && (washout_remaining_ == 0);

    // 4. for each member: (train) update readout on x_i(t), then inject the
    //    scaled deviation phi_i = kappa*(y_i - c) and step to x_i(t+1).
    for (size_t i = 0; i < M_; ++i)
    {
        if (train)
            esn_[i]->TrainLiveStepRegression(target, lr_, wd_);

        const float* y_i = y_flat_.data() + i * D_;
        for (size_t c = 0; c < D_; ++c)
            phi_[c] = kappa_ * (y_i[c] - c_out[c]);

        esn_[i]->StepLiveExternalFeedback(input, phi_.data());
    }

    // 5. class drives kappa from the consensus error (§4.2) — not the caller.
    AdvanceKappa(c_out, target);

    if (washout_remaining_ > 0)
        --washout_remaining_;
    ++t_;
}

void EnsembleESN::AdvanceKappa(const float* c_out, const float* target)
{
    // Hold kappa at inference (no target) and through the washout: do not seed
    // the competence EMA with pre-training transient (decision #4).
    if (target == nullptr || washout_remaining_ > 0)
        return;

    // This step's consensus error: mean absolute error over the D channels.
    float e = 0.0f;
    for (size_t c = 0; c < D_; ++c)
        e += std::fabs(c_out[c] - target[c]);
    e /= static_cast<float>(D_);

    if (!err_init_)
    {
        consensus_err_ = e;     // seed directly rather than blending from 0
        err_init_ = true;
    }
    else
    {
        consensus_err_ = (1.0f - gate_err_ema_alpha_) * consensus_err_
                         + gate_err_ema_alpha_ * e;
    }

    if (!gate_open_ && consensus_err_ < gate_threshold_)
        gate_open_ = true;

    if (gate_open_)
    {
        if (kappa_ramp_rate_ <= 0.0f)
            kappa_ = kappa_target_;                       // snap on gate open
        else
            kappa_ = std::min(kappa_target_, kappa_ + kappa_ramp_rate_);
    }
}

void EnsembleESN::BeginSequence()
{
    for (auto& e : esn_)
        e->ResetReservoirOnly();
    // Re-impose a short washout; kappa schedule + gate state are preserved (§7.1).
    washout_remaining_ = resequence_washout_;
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
