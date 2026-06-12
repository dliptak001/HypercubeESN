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

ReadoutConfig ESN::MakeFeedbackReadoutConfig(const ESNConfig& cfg, const ReadoutGeometry& geo)
{
    // F sees the same stride-subsampled state as P, so it shares the
    // geometry-forced dim; its output shape is fixed by the scheme — one
    // scalar regression head, whatever P's task is (§6.14, §6.16).
    ReadoutConfig rc = cfg.feedback.readout;
    rc.dim = geo.dim;
    rc.num_outputs = 1;
    rc.task = ReadoutTask::Regression;
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

    if (cfg.reservoir.num_feedback_channels > 0)
    {
        if (cfg.reservoir.num_feedback_channels != 1)
            throw std::invalid_argument(
                "ESN: feedback training (v1) supports exactly 1 feedback channel "
                "(num_feedback_channels=" +
                std::to_string(cfg.reservoir.num_feedback_channels) + ")");
        feedback_readout_ = std::make_unique<Readout>(MakeFeedbackReadoutConfig(cfg, geo));
        // Eager CNN build (§6.15): F must exist before the first closed-loop
        // Step it feeds — warmup is already a consumer. The resulting random
        // F is the live policy from cycle 0 (§6.8) and is persist-worthy
        // immediately (IsTrained() is set here, by design).
        feedback_readout_->InitOnline();

        fb_decision_state_.resize(num_output_verts_);
        fb_pred_.resize(readout_.NumOutputs());
        fb_ring_.resize(kFeedbackTelemetryWindow);
    }
}

void ESN::InjectFeedbackClamped(float raw)
{
    // force_zero overrides the value at the clamp seam only — the caller has
    // still evaluated F, keeping the lesion arm compute-matched (§6.13).
    const float f = esn_config_.feedback.force_zero ? 0.0f : std::tanh(raw);
    reservoir_->InjectFeedback(0, f);
}

void ESN::StepLive(const float* inputs)
{
    if (feedback_readout_)
    {
        CopyLiveState(scratch_subsampled_.data());
        // Cached as telemetry's f_commit (the §7.4 on-trajectory value):
        // the commit's F-forward doubles as the measurement, no second pass.
        last_fb_raw_ = feedback_readout_->PredictRaw(scratch_subsampled_.data());
        InjectFeedbackClamped(last_fb_raw_);
    }
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
    const size_t M = num_output_verts_;
    states_.resize((num_collected_ + num_steps) * M);
    for (size_t s = 0; s < num_steps; ++s)
    {
        StepLive(inputs + s * num_inputs_);
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
    // Stored as the §6.17 validation washout W — same transient-killing job,
    // same magnitude, no new hyperparameter.
    warmup_count_ = warmup_count;
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

// ---------------------------------------------------------------
//  Feedback training orchestration
//  (docs/FeedbackTrainingMethodology.md §4, §6.9, §6.16, §6.17)
// ---------------------------------------------------------------

namespace
{
    // §6.16 probe losses. The accept test only needs a total order on the
    // three probes; both losses are computed in double so that bit-identical
    // probe states (the §6.13 kill-switch arms, ε = 0) yield exactly equal
    // values and every equality rejects.

    double MseLoss(const float* pred, const float* target, size_t k)
    {
        double s = 0.0;
        for (size_t i = 0; i < k; ++i)
        {
            const double d = static_cast<double>(pred[i]) - static_cast<double>(target[i]);
            s += d * d;
        }
        return s / static_cast<double>(k);
    }

    // Softmax cross-entropy of the target label from raw logits,
    // log-sum-exp stabilized: CE = log Σ exp(l_i − m) − (l_t − m).
    double CrossEntropyLoss(const float* logits, size_t k, int target_class)
    {
        double m = logits[0];
        for (size_t i = 1; i < k; ++i)
            m = std::max(m, static_cast<double>(logits[i]));
        double lse = 0.0;
        for (size_t i = 0; i < k; ++i)
            lse += std::exp(static_cast<double>(logits[i]) - m);
        return std::log(lse) - (static_cast<double>(logits[target_class]) - m);
    }
} // namespace

void ESN::RequireFeedbackTraining(const char* method) const
{
    if (!feedback_readout_)
        throw std::logic_error(std::string("ESN::") + method +
                               ": feedback not configured (num_feedback_channels == 0)");
    if (!readout_.IsTrained())
        throw std::logic_error(std::string("ESN::") + method +
                               ": call InitOnline first (P's network is not built)");
}

double ESN::ProbeLoss(const float* input, float raw_feedback,
                      const float* target, int target_class)
{
    InjectFeedbackClamped(raw_feedback);
    for (size_t ch = 0; ch < num_inputs_; ++ch)
        reservoir_->InjectInput(ch, input[ch]);
    reservoir_->Step();

    CopyLiveState(scratch_subsampled_.data());
    readout_.PredictRaw(scratch_subsampled_.data(), fb_pred_.data());

    const size_t k = readout_.NumOutputs();
    return (esn_config_.readout.task == ReadoutTask::Regression)
               ? MseLoss(fb_pred_.data(), target, k)
               : CrossEntropyLoss(fb_pred_.data(), k, target_class);
}

ESN::FeedbackCycleInfo ESN::TrainFeedbackCycleImpl(const float* input,
                                                   const float* target, int target_class)
{
    const FeedbackConfig& fb = esn_config_.feedback;

    FeedbackCycleInfo info;
    info.pretrain = fb_examples_ < fb.pretrain_steps;
    info.e0 = info.e_plus = info.e_minus = info.sf =
        std::numeric_limits<float>::quiet_NaN();

    if (!info.pretrain)
    {
        // ----- Pass 2 (§4 steps a–i): probe, maybe train F, restore -----
        const Reservoir::Snapshot snap = reservoir_->TakeSnapshot();
        CopyLiveState(fb_decision_state_.data()); // subsample(Sx), F's training input (§6.4)
        const float sf = feedback_readout_->PredictRaw(fb_decision_state_.data());
        info.sf = sf;

        const double e0 = ProbeLoss(input, sf, target, target_class);
        reservoir_->RestoreSnapshot(snap);
        const double ep = ProbeLoss(input, sf + fb.epsilon, target, target_class);
        reservoir_->RestoreSnapshot(snap);
        const double em = ProbeLoss(input, sf - fb.epsilon, target, target_class);
        reservoir_->RestoreSnapshot(snap);

        info.e0 = static_cast<float>(e0);
        info.e_plus = static_cast<float>(ep);
        info.e_minus = static_cast<float>(em);

        // Accept iff a direction beats baseline by the RELATIVE margin
        // (min < E0·(1−r), §6.6 as amended) AND the directions are
        // distinguishable — every exact equality rejects (§6.6; this
        // strictness is what lets the §6.13 kill-switch arms freeze F with
        // no extra machinery). Relative, not absolute: probe losses span
        // orders of magnitude across cycles, and the §6.11 saturation
        // ratchet lives exactly in the deltas that are real but a microscopic
        // fraction of E0 — an absolute threshold cannot sit between the two
        // regimes (margin sweep, first §9.4 campaign).
        if (std::min(ep, em) < e0 * (1.0 - static_cast<double>(fb.margin)) && ep != em)
        {
            info.accepted = true;
            info.sign = (ep < em) ? 1.0f : -1.0f;
            // Pre-clamp target (§6.11), boxed to the clamp's usable region:
            // the wall is what bounds the chance-accept random walk — when
            // drift pushes Sf toward the box, outward creeps clamp back to
            // the wall while inward creeps remain full-size.
            const float f_star = std::clamp(sf + info.sign * fb.epsilon,
                                            -fb.f_box, fb.f_box);
            feedback_readout_->TrainOnlineStepRegression(
                fb_decision_state_.data(), &f_star, fb.lr, fb.readout.weight_decay);
            (info.sign > 0.0f ? fb_accepts_pos_ : fb_accepts_neg_)++;
        }
    }

    // ----- Pass 1: commit for real and train P per-step (§4, §6.10) -----
    // StepLive re-evaluates F on the restored state: the committed feedback
    // is F's live post-update output F′(Sx), never f* or the cached Sf (§6.2).
    StepLive(input);

    // §6.9 lr policy: cosine over the declared pre-train budget, annealing
    // into the constant alternation lr — no discontinuity at the boundary.
    const float p_lr =
        info.pretrain
            ? CosineLR(static_cast<float>(fb_examples_) / static_cast<float>(fb.pretrain_steps),
                       esn_config_.readout.lr_max, fb.p_lr)
            : fb.p_lr;
    info.p_lr = p_lr;

    CopyLiveState(scratch_subsampled_.data());
    if (esn_config_.readout.task == ReadoutTask::Regression)
        readout_.TrainOnlineStepRegression(scratch_subsampled_.data(), target,
                                           p_lr, esn_config_.readout.weight_decay);
    else
        readout_.TrainOnlineStep(scratch_subsampled_.data(), target_class,
                                 p_lr, esn_config_.readout.weight_decay);

    if (!info.pretrain)
    {
        // One telemetry record per alternation cycle, written post-commit so
        // f_commit (StepLive's cached raw F′(Sx)) and the §7.6 state RMS
        // describe the committed trajectory.
        FeedbackCycleRecord rec;
        rec.e0 = info.e0;
        rec.e_plus = info.e_plus;
        rec.e_minus = info.e_minus;
        rec.sf = info.sf;
        rec.f_commit = last_fb_raw_;
        rec.realization = info.accepted
                              ? std::fabs(last_fb_raw_ - info.sf) / fb.epsilon
                              : std::numeric_limits<float>::quiet_NaN();
        const float* state = reservoir_->Outputs();
        double ss = 0.0;
        for (size_t v = 0; v < n_; ++v)
            ss += static_cast<double>(state[v]) * static_cast<double>(state[v]);
        rec.state_rms = static_cast<float>(std::sqrt(ss / static_cast<double>(n_)));
        rec.accepted = info.accepted;
        rec.sign = info.sign;

        const size_t cycle_idx = fb_examples_ - fb.pretrain_steps;
        fb_ring_[cycle_idx % kFeedbackTelemetryWindow] = rec;
    }

    ++fb_examples_;
    return info;
}

ESN::FeedbackTelemetry ESN::GetFeedbackTelemetry() const
{
    if (!feedback_readout_)
        throw std::logic_error(
            "ESN::GetFeedbackTelemetry: feedback not configured (num_feedback_channels == 0)");

    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    FeedbackTelemetry t;
    t.pretrain_examples = std::min(fb_examples_, esn_config_.feedback.pretrain_steps);
    t.cycles = fb_examples_ - t.pretrain_examples;
    t.accepts_pos = fb_accepts_pos_;
    t.accepts_neg = fb_accepts_neg_;
    t.accepts = fb_accepts_pos_ + fb_accepts_neg_;
    t.window = std::min(t.cycles, kFeedbackTelemetryWindow);

    t.accept_rate = t.sign_balance = t.mean_f = t.var_f =
        t.mean_tanh_f = t.var_tanh_f = t.mean_abs_f =
        t.saturation_frac = t.mean_lever = t.mean_realization = t.mean_e0 =
        t.state_rms_mean = t.state_rms_max = nan;
    if (t.window == 0) return t;

    const size_t w = t.window;
    size_t w_accepts = 0, w_pos = 0, w_sat = 0;
    double sum_f = 0.0, sum_f2 = 0.0, sum_th = 0.0, sum_th2 = 0.0;
    double sum_abs = 0.0, sum_lever = 0.0;
    double sum_e0 = 0.0, sum_real = 0.0, sum_rms = 0.0, max_rms = 0.0;
    for (size_t i = t.cycles - w; i < t.cycles; ++i)
    {
        const FeedbackCycleRecord& r = fb_ring_[i % kFeedbackTelemetryWindow];
        const double f = r.f_commit;
        sum_f += f;
        sum_f2 += f * f;
        sum_abs += std::fabs(f);
        const double th = std::tanh(f);
        sum_th += th;
        sum_th2 += th * th;
        sum_lever += 1.0 - th * th;
        if (std::fabs(f) > kFeedbackSaturationRawAbs) ++w_sat;
        sum_e0 += r.e0;
        sum_rms += r.state_rms;
        max_rms = std::max(max_rms, static_cast<double>(r.state_rms));
        if (r.accepted)
        {
            ++w_accepts;
            if (r.sign > 0.0f) ++w_pos;
            sum_real += r.realization;
        }
    }
    const double dw = static_cast<double>(w);
    t.accept_rate = static_cast<double>(w_accepts) / dw;
    t.mean_f = sum_f / dw;
    t.var_f = std::max(0.0, sum_f2 / dw - t.mean_f * t.mean_f);
    t.mean_tanh_f = sum_th / dw;
    t.var_tanh_f = std::max(0.0, sum_th2 / dw - t.mean_tanh_f * t.mean_tanh_f);
    t.mean_abs_f = sum_abs / dw;
    t.saturation_frac = static_cast<double>(w_sat) / dw;
    t.mean_lever = sum_lever / dw;
    t.mean_e0 = sum_e0 / dw;
    t.state_rms_mean = sum_rms / dw;
    t.state_rms_max = max_rms;
    if (w_accepts > 0)
    {
        t.sign_balance = (static_cast<double>(w_pos) -
                          static_cast<double>(w_accepts - w_pos)) / static_cast<double>(w_accepts);
        t.mean_realization = sum_real / static_cast<double>(w_accepts);
    }
    return t;
}

std::vector<ESN::FeedbackCycleRecord> ESN::GetFeedbackHistory() const
{
    if (!feedback_readout_)
        throw std::logic_error(
            "ESN::GetFeedbackHistory: feedback not configured (num_feedback_channels == 0)");

    const size_t cycles =
        fb_examples_ - std::min(fb_examples_, esn_config_.feedback.pretrain_steps);
    const size_t w = std::min(cycles, kFeedbackTelemetryWindow);
    std::vector<FeedbackCycleRecord> out;
    out.reserve(w);
    for (size_t i = cycles - w; i < cycles; ++i)
        out.push_back(fb_ring_[i % kFeedbackTelemetryWindow]);
    return out;
}

ESN::FeedbackCycleInfo ESN::TrainFeedbackCycle(const float* input, const float* target)
{
    RequireFeedbackTraining("TrainFeedbackCycle");
    if (esn_config_.readout.task != ReadoutTask::Regression)
        throw std::invalid_argument(
            "ESN::TrainFeedbackCycle(float* target): P's task is Classification — "
            "use the (input, int target_class) overload");
    return TrainFeedbackCycleImpl(input, target, /*target_class=*/-1);
}

ESN::FeedbackCycleInfo ESN::TrainFeedbackCycle(const float* input, int target_class)
{
    RequireFeedbackTraining("TrainFeedbackCycle");
    if (esn_config_.readout.task != ReadoutTask::Classification)
        throw std::invalid_argument(
            "ESN::TrainFeedbackCycle(int target_class): P's task is Regression — "
            "use the (input, float* target) overload");
    return TrainFeedbackCycleImpl(input, /*target=*/nullptr, target_class);
}

double ESN::ValidateClosedLoop(const float* inputs, const float* targets, size_t count)
{
    RequireFeedbackTraining("ValidateClosedLoop");
    const size_t W = warmup_count_;
    if (count <= W)
        throw std::invalid_argument(
            "ESN::ValidateClosedLoop: count (" + std::to_string(count) +
            ") must exceed the washout W (" + std::to_string(W) +
            ", the warmup count passed to InitOnline)");

    // §6.17 bracket: only the reservoir needs protecting — validation is
    // forward-only, so P's and F's weights and Adam moments are untouched by
    // construction. Zero-reset entry makes every evaluation start
    // bit-identically; consecutive scores differ only because F/P changed.
    const Reservoir::Snapshot snap = reservoir_->TakeSnapshot();
    reservoir_->Reset();

    for (size_t s = 0; s < W; ++s)
        StepLive(inputs + s * num_inputs_); // closed-loop washout, unscored

    const size_t k = readout_.NumOutputs();
    const bool regression = (esn_config_.readout.task == ReadoutTask::Regression);
    const size_t scored = count - W;

    // Regression: per-output running sums for NRMSE. Classification: CE sum.
    std::vector<double> sum_y(k, 0.0), sum_y2(k, 0.0), sum_se(k, 0.0);
    double ce_sum = 0.0;

    for (size_t s = W; s < count; ++s)
    {
        StepLive(inputs + s * num_inputs_);
        CopyLiveState(scratch_subsampled_.data());
        readout_.PredictRaw(scratch_subsampled_.data(), fb_pred_.data());

        if (regression)
        {
            const float* y = targets + s * k;
            for (size_t j = 0; j < k; ++j)
            {
                const double yj = y[j];
                const double d = yj - static_cast<double>(fb_pred_[j]);
                sum_y[j] += yj;
                sum_y2[j] += yj * yj;
                sum_se[j] += d * d;
            }
        }
        else
        {
            ce_sum += CrossEntropyLoss(fb_pred_.data(), k, static_cast<int>(targets[s]));
        }
    }

    reservoir_->RestoreSnapshot(snap); // training resumes gapless

    if (!regression)
        return ce_sum / static_cast<double>(scored);

    double nrmse_sum = 0.0;
    for (size_t j = 0; j < k; ++j)
    {
        const double mean = sum_y[j] / static_cast<double>(scored);
        const double var = sum_y2[j] / static_cast<double>(scored) - mean * mean;
        if (var < 1e-12)
            nrmse_sum += std::numeric_limits<double>::infinity();
        else
            nrmse_sum += std::sqrt((sum_se[j] / static_cast<double>(scored)) / var);
    }
    return nrmse_sum / static_cast<double>(k);
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

ESN::ReadoutState ESN::GetFeedbackState() const
{
    if (!feedback_readout_)
        throw std::logic_error(
            "ESN::GetFeedbackState: feedback not configured (num_feedback_channels == 0)");
    ReadoutState s;
    s.is_trained = feedback_readout_->IsTrained();
    const auto& w = feedback_readout_->Weights();
    s.weights.assign(w.begin(), w.end());
    return s;
}

void ESN::SetFeedbackState(const ReadoutState& state)
{
    if (!feedback_readout_)
        throw std::logic_error(
            "ESN::SetFeedbackState: feedback not configured (num_feedback_channels == 0)");
    if (!state.is_trained) return;
    feedback_readout_->SetState(state.weights);
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
