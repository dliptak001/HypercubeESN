#include "Reservoir.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Reservoir::Reservoir(const ReservoirConfig& cfg)
    : rng_seed_(cfg.seed),
      dim_(cfg.dim),
      num_inputs_(cfg.num_inputs),
      spectral_radius_(cfg.spectral_radius),
      leak_rate_(cfg.leak_rate),
      input_scaling_(cfg.input_scaling),
      verbose_(cfg.verbose),
      history_depth_(cfg.history_depth),
      num_ext_feedback_channels_(cfg.num_external_feedback_channels),
      ext_feedback_scaling_(cfg.external_feedback_scaling),
      bias_scaling_(cfg.bias_scaling)
{
    if (dim_ < 5 || dim_ > 16)
        throw std::invalid_argument("dim must be in 5 <= dim <= 16");

    n_ = 1ULL << dim_;
    num_input_weights_ = n_ * dim_;

    if (spectral_radius_ <= 0.0f)
        throw std::invalid_argument("spectral_radius must be positive");
    if (leak_rate_ <= 0.0f || leak_rate_ > 1.0f)
        throw std::invalid_argument("leak_rate must be in (0.0, 1.0]");
    if (num_inputs_ == 0)
        throw std::invalid_argument("num_inputs must be >= 1");
    if (n_ % num_inputs_ != 0)
        throw std::invalid_argument(
            "num_inputs must divide N = 2^dim evenly "
            "(otherwise InjectInput drops the remainder vertices)");
    if (history_depth_ < 1 || history_depth_ > 64)
        throw std::invalid_argument("history_depth must be in [1, 64]");
    if (num_ext_feedback_channels_ > n_)
        throw std::invalid_argument(
            "num_external_feedback_channels must not exceed N = 2^dim "
            "(each channel drives a block of floor(N/D) >= 1 vertices)");
    // D need not divide N: block = floor(N/D); the N mod D tail stays at zero as
    // *sources* but still receives drive via the neighbor gather.

    // Weight layout: [ input: N·DIM | ext-fb: N·DIM if D>0 | recurrent: N·M·DIM ]
    const size_t drive_blocks = 1u /*input*/
        + (num_ext_feedback_channels_ > 0 ? 1u : 0u);
    num_weights_ = n_ * dim_ * (history_depth_ + drive_blocks);

    vtx_input_.reset(AllocAligned(n_));
    vtx_state_.reset(AllocAligned(n_));
    vtx_output_history_.reset(AllocAligned(n_ * history_depth_));
    vtx_weight_.reset(AllocAligned(num_weights_));
    // Value-initialized (trailing ()): pointers filled in Clear() via Initialize().
    // Null until then so a premature read faults loudly.
    slice_ptrs_.reset(new float*[history_depth_]());

    vtx_bias_.reset(AllocAligned(n_));

    num_ext_feedback_weights_ = num_ext_feedback_channels_ > 0 ? n_ * dim_ : 0;
    if (num_ext_feedback_channels_ > 0)
        vtx_ext_feedback_.reset(AllocAligned(n_));

    Initialize();
}

// ---------------------------------------------------------------------------
// Seeding
// ---------------------------------------------------------------------------

// SplitMix64 finalizer: avalanches a 64-bit value so labeled substreams of one
// master seed are statistically independent. Prefer this over additive offsets
// on mt19937 seeds.
static inline uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Named substreams of the master seed (labeled, not sequential — adding a role
// later must not renumber these; values are part of the weight-draw ABI).
enum class SeedRole : uint64_t {
    Recurrent = 1,
    Input = 2,
    ExternalFeedback = 3, // historical name "Feedback"; keep value 3
    Bias = 4,
    SrProbe = 5
};

// ---------------------------------------------------------------------------
// Weight draw + spectral-radius rescale
// ---------------------------------------------------------------------------

void Reservoir::Initialize()
{
    auto seed_for = [this](SeedRole r) {
        return mix64(rng_seed_ ^ (0x100000001B3ULL * static_cast<uint64_t>(r)));
    };
    std::mt19937_64 rng(seed_for(SeedRole::Recurrent));
    std::mt19937_64 in_rng(seed_for(SeedRole::Input));
    std::mt19937_64 ext_fb_rng(seed_for(SeedRole::ExternalFeedback));
    std::mt19937_64 bias_rng(seed_for(SeedRole::Bias));
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Clear();

    for (size_t i = 0; i < n_; ++i)
        vtx_bias_[i] = static_cast<float>(dist(bias_rng)) * bias_scaling_;

    float* pW = vtx_weight_.get();

    // Drive ports: draw U(-1,1), then scale by (port_scaling / √dim). Scaling last
    // so fan-in of the dim-neighbor gather does not grow with degree. Local
    // construction only — not a claim that one scaling is optimal across DIM/task.
    float* const input_base = pW;
    for (size_t i = 0; i < num_input_weights_; ++i)
        (*pW++) = static_cast<float>(dist(in_rng));
    const float in_scaling = input_scaling_ / std::sqrt(static_cast<float>(dim_));
    for (size_t i = 0; i < num_input_weights_; ++i)
        input_base[i] *= in_scaling;

    float* const ext_base = pW;
    for (size_t i = 0; i < num_ext_feedback_weights_; ++i)
        (*pW++) = static_cast<float>(dist(ext_fb_rng));
    if (num_ext_feedback_weights_ > 0)
    {
        const float ext_scale =
            ext_feedback_scaling_ / std::sqrt(static_cast<float>(dim_));
        for (size_t i = 0; i < num_ext_feedback_weights_; ++i)
            ext_base[i] *= ext_scale;
    }

    // Recurrent: U(-1,1) / √(DIM·M), then global secant rescale to target SR.
    // Layout [vertex][slice][axis] matches UpdateState.
    const size_t rec_base = RecurrentWeightBase();
    const float w_scaling =
        1.0f / std::sqrt(static_cast<float>(dim_ * history_depth_));
    for (size_t i = rec_base; i < num_weights_; ++i)
        vtx_weight_[i] = static_cast<float>(dist(rng)) * w_scaling;

    const float target = spectral_radius_;
    const size_t MN = history_depth_ * n_;
    std::vector<float> sr_x(MN, 0.0f), sr_y(MN, 0.0f);
    {
        std::mt19937_64 sr_rng(seed_for(SeedRole::SrProbe));
        std::uniform_real_distribution<double> sr_dist(-1.0, 1.0);
        float norm = 0.0f;
        for (size_t v = 0; v < n_; ++v)
        {
            sr_x[v] = static_cast<float>(sr_dist(sr_rng));
            norm += sr_x[v] * sr_x[v];
        }
        norm = std::sqrt(norm);
        for (size_t v = 0; v < n_; ++v)
            sr_x[v] /= norm;
    }

    // eval_sr(s): multiply recurrent block by s/applied_scale and re-estimate ρ.
    float applied_scale = 1.0f;
    auto eval_sr = [&](float s) {
        const float rel = s / applied_scale;
        for (size_t i = rec_base; i < num_weights_; ++i)
            vtx_weight_[i] *= rel;
        applied_scale = s;
        return EstimateSpectralRadius(sr_x, sr_y);
    };

    const float pre_sr = EstimateSpectralRadius(sr_x, sr_y); // ρ at scale 1
    float post_sr = pre_sr;
    int sr_iters = 0;
    if (pre_sr > 1e-6f)
    {
        constexpr float kSrTolRel = 0.001f;
        constexpr int kMaxSrIters = 20;

        // Secant on h(s) = ρ(s) - target. s0 = 1; s1 = target/pre_sr (exact if M==1).
        float s0 = 1.0f, h0 = pre_sr - target;
        float s1 = target / pre_sr, h1 = eval_sr(s1) - target;
        ++sr_iters;
        post_sr = h1 + target;
        while (sr_iters < kMaxSrIters &&
               std::abs(post_sr - target) > target * kSrTolRel)
        {
            const float denom = h1 - h0;
            float s2 = (std::abs(denom) < 1e-12f)
                           ? s1 * (target / std::max(post_sr, 1e-6f))
                           : s1 - h1 * (s1 - s0) / denom;
            s2 = std::clamp(s2, 0.25f * s1, 4.0f * s1);
            post_sr = eval_sr(s2);
            ++sr_iters;
            s0 = s1;
            h0 = h1;
            s1 = s2;
            h1 = post_sr - target;
        }
    }
    realized_spectral_radius_ = post_sr;
    if (verbose_)
    {
        std::printf("[Reservoir DIM=%zu M=%zu seed=%llu leak=%.3g in_scale=%.3g "
                    "SR target=%.4f post=%.4f (secant iters=%d)]\n",
                    dim_, history_depth_,
                    static_cast<unsigned long long>(rng_seed_), leak_rate_,
                    input_scaling_, target, post_sr, sr_iters);
    }
}

// ---------------------------------------------------------------------------
// Dynamics
// ---------------------------------------------------------------------------

void Reservoir::Step()
{
    const float* p_vtx_prev = slice_ptrs_[0];
    for (size_t v = 0; v < n_; v++)
        UpdateState(v, p_vtx_prev[v]);

    // Age the delay line: rotate logical slice pointers, publish vtx_state_ as age 0.
    float* p0 = slice_ptrs_[history_depth_ - 1];
    for (size_t i = history_depth_ - 1; i > 0; --i)
        slice_ptrs_[i] = slice_ptrs_[i - 1];
    slice_ptrs_[0] = p0;

    std::memcpy(slice_ptrs_[0], vtx_state_.get(), n_ * sizeof(float));
    std::memset(vtx_input_.get(), 0, n_ * sizeof(float));

    if (num_ext_feedback_channels_ > 0)
        std::memset(vtx_ext_feedback_.get(), 0, n_ * sizeof(float));
}

void Reservoir::UpdateState(const size_t v, const float old_output_v)
{
    float s = 0.0f;
    const float* iw = vtx_weight_.get() + v * dim_; // input row for v
    const float* w =
        &vtx_weight_[RecurrentWeightBase()] + v * dim_ * history_depth_;

    // Input gather: dim Hamming neighbors of the staged input field.
    // num_inputs_==1 makes all neighbors identical (could collapse to a row-sum
    // multiply); we keep the general form so multi-input block boundaries still
    // mix correctly. Cost is negligible vs the recurrent block (dim·M FMAs).
    for (size_t i = 0; i < dim_; i++)
        s += vtx_input_[v ^ NearestMask(i)] * iw[i];

    if (num_ext_feedback_channels_ > 0)
    {
        const float* ew = &vtx_weight_[num_input_weights_] + v * dim_;
        for (size_t i = 0; i < dim_; i++)
            s += vtx_ext_feedback_[v ^ NearestMask(i)] * ew[i];
    }

    // Recurrent gather: M logical ages × dim axes.
    for (size_t i = 0; i < history_depth_; i++)
    {
        const float* pSlice = slice_ptrs_[i];
        for (size_t j = 0; j < dim_; j++)
            s += pSlice[v ^ NearestMask(j)] * (*w++);
    }

    const float activation = std::tanh(s) + vtx_bias_[v];
    vtx_state_[v] = (1.0f - leak_rate_) * old_output_v + leak_rate_ * activation;
}

// ---------------------------------------------------------------------------
// Drive injection
// ---------------------------------------------------------------------------

void Reservoir::InjectInput(const size_t channel, const float input)
{
    if (channel >= num_inputs_)
        throw std::invalid_argument(
            "InjectInput: channel out of range [0, num_inputs)");
    const size_t block = n_ / num_inputs_;
    const size_t v_end = (channel + 1) * block;
    for (size_t v = channel * block; v < v_end; ++v)
        vtx_input_[v] = input;
}

void Reservoir::InjectExternalFeedback(const size_t channel, const float value)
{
    if (channel >= num_ext_feedback_channels_)
        throw std::invalid_argument(
            "InjectExternalFeedback: channel out of range "
            "[0, num_external_feedback_channels)");

    const size_t block = n_ / num_ext_feedback_channels_;
    const size_t v_end = (channel + 1) * block;
    for (size_t v = channel * block; v < v_end; ++v)
        vtx_ext_feedback_[v] = value;
}

void Reservoir::InjectExternalFeedback(const float* values, const size_t count)
{
    if (count != num_ext_feedback_channels_)
        throw std::invalid_argument(
            "InjectExternalFeedback(vector): count must equal "
            "num_external_feedback_channels");
    if (count > 0 && values == nullptr)
        throw std::invalid_argument(
            "InjectExternalFeedback(vector): values is null but count > 0");
    for (size_t c = 0; c < count; ++c)
        InjectExternalFeedback(c, values[c]);
}

// ---------------------------------------------------------------------------
// Snapshot / config / clear
// ---------------------------------------------------------------------------

Reservoir::Snapshot Reservoir::TakeSnapshot() const
{
    Snapshot s;
    s.state.assign(vtx_state_.get(), vtx_state_.get() + n_);
    s.history.resize(n_ * history_depth_);
    // Logical age order via slice_ptrs_ (independent of physical ring phase).
    for (size_t i = 0; i < history_depth_; ++i)
        std::memcpy(s.history.data() + i * n_, slice_ptrs_[i], n_ * sizeof(float));
    return s;
}

void Reservoir::RestoreSnapshot(const Snapshot& snap)
{
    if (snap.state.size() != n_ || snap.history.size() != n_ * history_depth_)
        throw std::invalid_argument(
            "RestoreSnapshot: snapshot sizes do not match this reservoir "
            "(expected state=N, history=N*history_depth)");

    std::memcpy(vtx_state_.get(), snap.state.data(), n_ * sizeof(float));
    std::memcpy(vtx_output_history_.get(), snap.history.data(),
                n_ * history_depth_ * sizeof(float));

    // Snapshot is logical age order → re-home ring to canonical physical layout.
    for (size_t i = 0; i < history_depth_; ++i)
        slice_ptrs_[i] = &vtx_output_history_[i * n_];

    std::memset(vtx_input_.get(), 0, n_ * sizeof(float));
    if (num_ext_feedback_channels_ > 0)
        std::memset(vtx_ext_feedback_.get(), 0, n_ * sizeof(float));
}

ReservoirConfig Reservoir::GetConfig() const
{
    ReservoirConfig cfg;
    cfg.dim = dim_;
    cfg.seed = rng_seed_;
    cfg.spectral_radius = spectral_radius_; // target, not realized
    cfg.leak_rate = leak_rate_;
    cfg.input_scaling = input_scaling_;
    cfg.num_inputs = num_inputs_;
    cfg.history_depth = history_depth_;
    cfg.verbose = verbose_;
    cfg.num_external_feedback_channels = num_ext_feedback_channels_;
    cfg.external_feedback_scaling = ext_feedback_scaling_;
    cfg.bias_scaling = bias_scaling_;
    return cfg;
}

const float* Reservoir::SliceAt(const size_t age) const
{
    if (age >= history_depth_)
        throw std::out_of_range(
            "Reservoir::SliceAt: age (" + std::to_string(age) +
            ") >= history_depth (" + std::to_string(history_depth_) + ")");
    // slice_ptrs_ already in logical age order after Step()'s rotation.
    return slice_ptrs_[age];
}

void Reservoir::Clear()
{
    std::memset(vtx_state_.get(), 0, n_ * sizeof(float));
    std::memset(vtx_input_.get(), 0, n_ * sizeof(float));

    if (num_ext_feedback_channels_ > 0)
        std::memset(vtx_ext_feedback_.get(), 0, n_ * sizeof(float));

    std::memset(vtx_output_history_.get(), 0, n_ * history_depth_ * sizeof(float));

    for (size_t i = 0; i < history_depth_; i++)
        slice_ptrs_[i] = &vtx_output_history_[i * n_];
}

// ---------------------------------------------------------------------------
// Spectral radius (companion operator on MN-dimensional delay state)
// ---------------------------------------------------------------------------

float Reservoir::EstimateSpectralRadius(std::span<float> x, std::span<float> y) const
{
    const size_t MN = history_depth_ * n_;
    assert(x.size() >= MN && y.size() >= MN);

    // Power iteration with Gelfand (geometric-mean) growth rates.
    //
    // The augmented operator is a block-companion / delay-line matrix. Its
    // dominant eigenvalue is often a complex conjugate pair in a tight modulus
    // cluster as M grows. Instantaneous |A x| then oscillates and converges
    // slowly — too noisy for the secant SR solve at large M.
    //
    // With unit-normalized x each step, growth ratios telescope:
    //   prod |A x_k| = |A^n x_0|, geometric mean → |λ₁| (Gelfand).
    // Averaging log-ratios damps rotation and subdominant clustering. Stop when
    // the running geometric mean is stable over a spaced check.
    constexpr int kMaxIters = 1500;
    constexpr int kBurnIn = 32;
    constexpr int kCheckSpacing = 50;
    constexpr float kTolRel = 1e-4f;

    float rho_ring[kCheckSpacing] = {};
    double sum_log = 0.0;
    int n_acc = 0;
    float rho = 0.0f;

    for (int iter = 0; iter < kMaxIters; ++iter)
    {
        // Top block: y_0[v] = sum over slices j and axes i of W[v,j,i] * x_j[v^mask].
        for (size_t v = 0; v < n_; v++)
        {
            float s = 0.0f;
            const float* w =
                &vtx_weight_[RecurrentWeightBase()] + v * dim_ * history_depth_;
            for (size_t j = 0; j < history_depth_; j++)
            {
                const float* x_j = x.data() + j * n_;
                const float* wj = w + j * dim_;
                for (size_t i = 0; i < dim_; i++)
                    s += wj[i] * x_j[v ^ NearestMask(i)];
            }
            y[v] = s;
        }

        // Aging blocks: y_j = x_{j-1} for j >= 1.
        for (size_t j = 1; j < history_depth_; j++)
            std::memcpy(y.data() + j * n_, x.data() + (j - 1) * n_,
                        n_ * sizeof(float));

        float norm = 0.0f;
        for (size_t k = 0; k < MN; k++)
            norm += y[k] * y[k];
        norm = std::sqrt(norm);
        if (norm <= 1e-30f)
            return 0.0f; // nilpotent / zeroed operator

        const float inv = 1.0f / norm;
        for (size_t k = 0; k < MN; k++)
            x[k] = y[k] * inv;

        if (iter < kBurnIn)
            continue; // align to dominant subspace before accumulating

        sum_log += std::log(static_cast<double>(norm));
        ++n_acc;
        rho = static_cast<float>(std::exp(sum_log / static_cast<double>(n_acc)));

        const int slot = n_acc % kCheckSpacing;
        if (n_acc > kCheckSpacing &&
            std::abs(rho - rho_ring[slot]) < rho * kTolRel)
            break;
        rho_ring[slot] = rho;
    }

    return rho;
}
