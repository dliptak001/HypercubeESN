#include "Reservoir.h"

#include <random>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <vector>

Reservoir::Reservoir(const ReservoirConfig& cfg)
    : rng_seed_(cfg.seed),
      dim_(cfg.dim),
      num_inputs_(cfg.num_inputs),
      spectral_radius_(cfg.spectral_radius),
      leak_rate_(cfg.leak_rate),
      input_scaling_(cfg.input_scaling),
      verbose_(cfg.verbose),
      history_depth_(cfg.history_depth)
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
        throw std::invalid_argument("num_inputs must divide N = 2^dim evenly "
                                    "(otherwise InjectInput drops the remainder vertices)");
    if (history_depth_ < 1 || history_depth_ > 64)
        throw std::invalid_argument("history_depth must be in [1, 64]");

    num_weights_ = n_ * dim_ * (history_depth_ + 1 /*inputs*/);

    vtx_input_.reset(AllocAligned(n_));
    vtx_state_.reset(AllocAligned(n_));
    vtx_output_history_.reset(AllocAligned(n_ * history_depth_));
    vtx_weight_.reset(AllocAligned(num_weights_));
    slice_ptrs_.reset(new float*[history_depth_]);

    Initialize();
}

void Reservoir::Initialize()
{
    std::mt19937_64 rng(rng_seed_);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Reset();

    float* pW = vtx_weight_.get();
    // Input weights: 1/sqrt(dim) normalizes the dim-neighbor input fan-in so a given
    // input_scaling delivers DIM-invariant input drive. Unlike the recurrent block
    // (w_scaling, below) there is NO history factor: the input path has no delay
    // line — it sums dim neighbor inputs per vertex (UpdateState), independent of M.
    const float in_scaling = input_scaling_ / std::sqrt(static_cast<float>(dim_));
    for (size_t i = 0; i < num_input_weights_; ++i)
        (*pW++) = static_cast<float>(dist(rng)) * in_scaling;

    const float w_scaling = 1.0f / std::sqrt(static_cast<float>(dim_ * history_depth_));
    for (size_t i = 0; i < num_weights_ - num_input_weights_; i++)
        (*pW++) = static_cast<float>(dist(rng)) * w_scaling;

    const float target = spectral_radius_;
    const size_t MN = history_depth_ * n_;
    std::vector<float> sr_x(MN, 0.0f), sr_y(MN, 0.0f);
    {
        std::mt19937_64 sr_rng(rng_seed_ + 12345);
        std::uniform_real_distribution<double> sr_dist(-1.0, 1.0);
        float norm = 0.0f;
        for (size_t v = 0; v < n_; ++v)
        {
            sr_x[v] = static_cast<float>(sr_dist(sr_rng));
            norm += sr_x[v] * sr_x[v];
        }
        norm = std::sqrt(norm);
        for (size_t v = 0; v < n_; ++v) sr_x[v] /= norm;
    }

    float applied_scale = 1.0f;
    auto eval_sr = [&](float s)
    {
        const float rel = s / applied_scale;
        for (size_t i = num_input_weights_; i < num_weights_; ++i) vtx_weight_[i] *= rel;
        applied_scale = s;
        return EstimateSpectralRadius(sr_x, sr_y);
    };

    const float pre_sr = EstimateSpectralRadius(sr_x, sr_y); // rho at s = 1
    float post_sr = pre_sr;
    int sr_iters = 0;
    if (pre_sr > 1e-6f)
    {
        constexpr float kSrTolRel = 0.001f;
        constexpr int kMaxSrIters = 20;

        // Secant on h(s) = rho(s) - target. Seed s0 = 1 (rho = pre_sr) and the
        // linear guess s1 = target/pre_sr (which is exact for M==1).
        float s0 = 1.0f, h0 = pre_sr - target;
        float s1 = target / pre_sr, h1 = eval_sr(s1) - target;
        ++sr_iters;
        post_sr = h1 + target;
        while (sr_iters < kMaxSrIters &&
            std::abs(post_sr - target) > target * kSrTolRel)
        {
            const float denom = h1 - h0;
            float s2 = (std::abs(denom) < 1e-12f)
                           ? s1 * (target / std::max(post_sr, 1e-6f)) // fallback (guard /0)
                           : s1 - h1 * (s1 - s0) / denom; // secant step
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
        std::printf("[Reservoir DIM=%zu M=%zu seed=%llu leak=%.3g in_scale=%.3g "
                    "SR target=%.4f post=%.4f (secant iters=%d)]\n",
                    dim_, history_depth_,
                    static_cast<unsigned long long>(rng_seed_),
                    leak_rate_, input_scaling_,
                    target, post_sr, sr_iters);
}

void Reservoir::Step()
{
    const float* p_vtx_prev = slice_ptrs_[0];
    for (size_t v = 0; v < n_; v++)
        UpdateState(v, p_vtx_prev[v]);

    // Rotate slice pointers
    float* p0 = slice_ptrs_[history_depth_ - 1];
    for (size_t i = history_depth_ - 1; i > 0; --i)
        slice_ptrs_[i] = slice_ptrs_[i - 1];
    slice_ptrs_[0] = p0;

    std::memcpy(slice_ptrs_[0], vtx_state_.get(), n_ * sizeof(float));
    std::memset(vtx_input_.get(), 0, n_ * sizeof(float));
}

void Reservoir::UpdateState(size_t v, float old_output_v)
{
    float s = 0.0f;
    const float* iw = vtx_weight_.get() + v * dim_; // input block
    const float* w = &vtx_weight_[num_input_weights_] + v * dim_ * history_depth_; // recurrent block

    // Input fan-in: sum v's dim Hamming-neighbor inputs, each by its own weight.
    // For a SINGLE input (num_inputs_ == 1) InjectInput writes the same scalar to
    // every vertex, so all dim gathered values are identical and this collapses to
    // input * (sum of iw[0..dim)) — a single multiply against a per-vertex
    // precomputed weight-row-sum would suffice, making the dim-way gather here
    // wasted work. We keep the general form on purpose: with num_inputs_ > 1 the
    // neighbors of a vertex near a channel-block boundary straddle different
    // channels and carry DIFFERENT injected values, so the per-neighbor gather is
    // load-bearing and cannot be collapsed. The single-input waste is dim-1 extra
    // FMAs per vertex per step — negligible against the recurrent block below
    // (dim * history_depth) — so it isn't worth a second specialized code path.
    for (size_t i = 0; i < dim_; i++)
        s += vtx_input_[v ^ NearestMask(i)] * iw[i];

    for (size_t i = 0; i < history_depth_; i++)
    {
        const float* pSlice = slice_ptrs_[i];
        for (size_t j = 0; j < dim_; j++)
            s += pSlice[v ^ NearestMask(j)] * (*w++);
    }

    const float activation = std::tanh(s);

    vtx_state_[v] = (1.0f - leak_rate_) * old_output_v + leak_rate_ * activation;
}

void Reservoir::InjectInput(size_t channel, float input)
{
    if (channel >= num_inputs_)
        throw std::invalid_argument("InjectInput: channel out of range [0, num_inputs)");
    const size_t block = n_ / num_inputs_;
    const size_t v_end = (channel + 1) * block;
    for (size_t v = channel * block; v < v_end; ++v)
        vtx_input_[v] = input;
}

ReservoirConfig Reservoir::GetConfig() const
{
    ReservoirConfig cfg;
    cfg.dim             = dim_;
    cfg.seed            = rng_seed_;
    cfg.spectral_radius = spectral_radius_; // configured target, not realized
    cfg.leak_rate       = leak_rate_;
    cfg.input_scaling   = input_scaling_;
    cfg.num_inputs      = num_inputs_;
    cfg.history_depth   = history_depth_;
    cfg.verbose         = verbose_;
    return cfg;
}

void Reservoir::Reset()
{
    std::memset(vtx_state_.get(), 0, n_ * sizeof(float));
    std::memset(vtx_input_.get(), 0, n_ * sizeof(float));
    std::memset(vtx_output_history_.get(), 0, n_ * history_depth_ * sizeof(float));

    for (size_t i = 0; i < history_depth_; i++)
        slice_ptrs_[i] = &vtx_output_history_[i * n_];
}

float Reservoir::EstimateSpectralRadius(std::span<float> x, std::span<float> y) const
{
    const size_t MN = history_depth_ * n_;
    assert(x.size() >= MN && y.size() >= MN);

    // Power iteration with growth-rate (Gelfand) averaging.
    //
    // The augmented operator is a block-companion / delay-line matrix, whose
    // dominant eigenvalue is generically a COMPLEX conjugate pair sitting in a
    // cluster of near-modulus neighbours that worsens as history_depth grows.
    // That breaks naive power iteration two ways: (1) for a complex dominant
    // pair the iterate ROTATES, so the instantaneous norm |A x| oscillates and
    // never settles; (2) clustering pushes |lambda_2/lambda_1| -> 1, so
    // convergence crawls. Returning the instantaneous norm was therefore noisy
    // at the ~% level -- too noisy for the spectral-radius secant solve to hit
    // its tolerance at large M (it would cap out at kMaxSrIters off-target).
    //
    // Fix: x is renormalized to unit norm each step, so the per-step growth
    // ratio is exactly |A x|, and these ratios telescope:
    //   prod_{k=1..n} |A x_k| = |A^n x_0|.
    // Hence the GEOMETRIC MEAN of the ratios is (|A^n x_0|)^{1/n} -> |lambda_1|
    // (Gelfand's formula), whether or not lambda_1 is complex. Averaging the
    // log-ratios cancels the rotation oscillation and damps the subdominant
    // cluster. We track the running geometric mean and stop when it stops
    // moving across a spaced check -- the SMOOTHED mean, not the oscillating
    // instantaneous norm.
    constexpr int   kMaxIters     = 1500;  // hard cap (warm-started across secant evals)
    constexpr int   kBurnIn       = 32;    // align x with the dominant subspace first
    constexpr int   kCheckSpacing = 50;    // compare the running mean this many steps apart
    constexpr float kTolRel       = 1e-4f; // break when the running mean is this stable

    float  rho_ring[kCheckSpacing] = {};
    double sum_log = 0.0;
    int    n_acc   = 0;
    float  rho     = 0.0f;

    for (int iter = 0; iter < kMaxIters; ++iter)
    {
        // y = A x. Top block: y_0[v] = sum_j sum_i W[v,j,i] * x_j[v ^ mask(i)].
        for (size_t v = 0; v < n_; v++)
        {
            float s = 0.0f;
            const float* w = &vtx_weight_[num_input_weights_] + v * dim_ * history_depth_;
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
            std::memcpy(y.data() + j * n_, x.data() + (j - 1) * n_, n_ * sizeof(float));

        float norm = 0.0f;
        for (size_t k = 0; k < MN; k++) norm += y[k] * y[k];
        norm = std::sqrt(norm);
        if (norm <= 1e-30f) return 0.0f;   // nilpotent / zeroed operator

        const float inv = 1.0f / norm;
        for (size_t k = 0; k < MN; k++) x[k] = y[k] * inv;

        // Burn in before accumulating, so the early transient (before x aligns
        // with the dominant subspace) doesn't bias the geometric mean.
        if (iter < kBurnIn) continue;

        sum_log += std::log(static_cast<double>(norm));
        ++n_acc;
        rho = static_cast<float>(std::exp(sum_log / static_cast<double>(n_acc)));

        // rho_ring[slot] holds the running mean from kCheckSpacing steps ago;
        // break once the smoothed estimate has stopped moving over that span.
        const int slot = n_acc % kCheckSpacing;
        if (n_acc > kCheckSpacing &&
            std::abs(rho - rho_ring[slot]) < rho * kTolRel)
            break;
        rho_ring[slot] = rho;
    }

    return rho;
}
