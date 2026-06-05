#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ESN.h"

/// NARMA-N input/target generator + task builder for the HypercubeESN ESN.
///
/// History note (adaptation): this header was pulled forward from the
/// retired `diagnostics/NARMA_N.h` (deleted in the 2026-05 diagnostics
/// purge) into `examples/` and adapted to the current core API. The
/// pieces that survived unchanged are the parts that were always pure:
/// `NARMA_N_Generator`, the `NARMACoefficientsFor` schedule, and the
/// `NARMATask` plumbing.
///
/// `RunNARMATrial` is the place you configure the ESN: hand it an
/// `ESNConfig` (reservoir + readout knobs, incl. `reservoir.dim`) and a
/// `NARMATask`, and it constructs `ESN(cfg)`, drives it, trains, and reports
/// NRMSE/R2.
/// It is kept but adapted -- the original drove the readout through
/// `ESN::Train(targets, tr, hooks)` and a per-epoch `CNNTrainHooks`
/// callback (which `TrainAndTrimmedMeanNRMSE` used for a
/// trimmed-mean-over-epochs metric and a divergence abort). The current
/// core trains in a single `ESN::Train(targets, train_size)` call with no
/// hook surface, so that metric machinery is gone and `RunNARMATrial`
/// reports a single post-training NRMSE/R2 instead. `ESN::kReadoutCap` is
/// also gone, so task sizing (`collect`, `warmup`) is passed in explicitly
/// rather than derived from the readout cap.
///
/// Recurrence (order N):
///   raw    = alpha*y(t-1) + beta*y(t-1)*sum(y(t-1..t-N)) + gamma*u(t-N)*u(t) + delta
///   y(t)   = tanh_wrap ? tanh(raw) : raw
/// Inputs u(t) are drawn uniform in [u_low, u_high].
///
/// Two boundedness regimes, selected by `tanh_wrap` (A/B comparison):
///   - tanh_wrap == false (legacy): no squashing. Boundedness is the caller's
///     responsibility via the beta/delta schedule -- `NARMACoefficientsFor(N)`
///     (below) gives a real, stable fixed point for every supported order;
///     bare canonical coefficients (beta=0.05) diverge past N~=23. The cost is
///     that the schedule *weakens* the nonlinearity as N grows, so higher-order
///     NARMA is actually smoother/easier -- difficulty does NOT scale with order.
///   - tanh_wrap == true: the outer tanh bounds y(t) in (-1, 1) unconditionally,
///     so the coefficients stay FIXED at every order (MakeNARMATask uses the
///     canonical 0.3/0.05/1.5/0.1). The nonlinearity is preserved and the memory
///     demand grows with N, so difficulty scales *honestly* with order. This is
///     the standard literature form for NARMA-20/30.
///
/// Implementation is O(1) per step: an incremental running sum and
/// fixed-size vector ring buffers replace per-step window recomputation.
///
/// Target alignment -- bug fixed vs. the FractalHypercubeRC original.
/// This generator was ported from FractalHypercubeRC, whose version
/// paired inputs[t] = u(t) with targets[t] = y(t+1) -- a one-step-ahead
/// shift (it allocated an extra sample "+1 for target shift"). That is
/// wrong for NARMA: y(t+1) contains the term gamma*u(t+1)*u(t+1-N), so
/// it depends on the input u(t+1). A reservoir driven only through u(t)
/// has never seen u(t+1), making that term unlearnable -- NRMSE
/// collapses toward 1.0 (predict-the-mean). NARMA is system
/// identification, not forecasting: y(t) is produced from u(t) and
/// u(t-N). This version pairs inputs[t] = u(t) with targets[t] = y(t).
template <typename T = float>
class NARMA_N_Generator
{
public:
    NARMA_N_Generator(size_t N = 10,
                      uint64_t seed = 1939,
                      T alpha = T(0.3),
                      T beta = T(0.05),
                      T gamma = T(1.5),
                      T delta = T(0.1),
                      T u_low = T(0.0),
                      T u_high = T(0.5),
                      bool tanh_wrap = false)
        : N_(N), alpha_(alpha), beta_(beta), gamma_(gamma), delta_(delta),
          u_low_(u_low), u_high_(u_high), tanh_wrap_(tanh_wrap),
          rng_(seed), u_dist_(u_low_, u_high_)
    {
        if (N_ < 2)
            throw std::invalid_argument("NARMA_N_Generator: N must be >= 2");
    }

    /// Generate a NARMA-N series for a prediction task.
    /// Returns {inputs_u, targets_y} aligned at the same index:
    /// targets_y[t] = y(t) is the NARMA-N output for input u(t). y(t)
    /// depends only on u(t) and u(t-N) (plus y history), so a reservoir
    /// driven through u(t) has seen everything needed to reproduce it.
    std::pair<std::vector<T>, std::vector<T>>
    generate_prediction_task(size_t num_steps, size_t warmup_steps = 500)
    {
        if (num_steps == 0) return {{}, {}};

        const size_t total = num_steps + warmup_steps;

        std::vector<T> u_series(total);
        std::vector<T> y_series(total, T(0));

        // Fixed-size ring buffers for the last N_ y- and u-values. Head index
        // points to the oldest entry (the "front"); writing y_t at the head
        // then advancing the head naturally evicts the oldest. y_back tracks
        // the newest y (the previous step's y_t) so we avoid an extra buffer
        // read per step.
        //
        // u_hist is seeded with N_ independent draws so u(t-N) for the first
        // N_ steps comes from the stationary input distribution -- otherwise
        // the recurrence would see the same u_delayed value N_ times before
        // any genuine history arrives. Costs N_ extra RNG draws upfront.
        std::vector<T> y_hist(N_, T(0));
        std::vector<T> u_hist(N_);
        for (size_t i = 0; i < N_; ++i) u_hist[i] = u_dist_(rng_);
        size_t y_head = 0;
        size_t u_head = 0;
        T y_back = T(0);

        T running_sum_y = T(0);

        for (size_t t = 0; t < total; ++t)
        {
            T u_t = u_dist_(rng_);
            u_series[t] = u_t;

            T y_prev = y_back;
            T sum_y = running_sum_y;
            T u_delayed = u_hist[u_head];

            T y_t = alpha_ * y_prev
                  + beta_ * y_prev * sum_y
                  + gamma_ * u_delayed * u_t
                  + delta_;

            // Optional outer squashing. When enabled, y(t) is bounded in
            // (-1, 1) unconditionally, so fixed coefficients stay stable at any
            // order and the divergence guard below never fires (it's a cheap
            // no-op in this mode).
            if (tanh_wrap_)
                y_t = std::tanh(y_t);

            // Divergence guard. Sane NARMA series sit in [0, ~1]; anything
            // past kDivergenceThreshold means the recurrence is unstable
            // (typically a beta/N combination outside the schedule encoded
            // in NARMACoefficientsFor -- bare canonical beta=0.05 has no
            // real fixed point past N~=23). Magnitude check rather than
            // std::isfinite so it survives -ffast-math, which the Release
            // build enables and which lets the compiler assume no inf/NaN.
            constexpr T kDivergenceThreshold = T(1e6);
            if (y_t > kDivergenceThreshold || y_t < -kDivergenceThreshold)
            {
                std::ostringstream msg;
                msg.precision(6);
                msg << "NARMA_N_Generator diverged at step " << t
                    << " of " << total
                    << ": y(t)=" << y_t
                    << "  y_prev=" << y_prev
                    << "  sum_y=" << sum_y
                    << "  u(t)=" << u_t
                    << "  u(t-N)=" << u_delayed
                    << "; recurrence terms: alpha*y_prev=" << (alpha_ * y_prev)
                    << "  beta*y_prev*sum_y=" << (beta_ * y_prev * sum_y)
                    << "  gamma*u*u_delayed=" << (gamma_ * u_delayed * u_t)
                    << "  delta=" << delta_
                    << "; coefficients: N=" << N_
                    << "  alpha=" << alpha_
                    << "  beta=" << beta_
                    << "  gamma=" << gamma_
                    << "  delta=" << delta_
                    << "  u in [" << u_low_ << ", " << u_high_ << "]"
                    << "  beta*N=" << (beta_ * static_cast<T>(N_))
                    << ". Past N~=23 the canonical beta=0.05 quadratic has no"
                       " real fixed point. Build via MakeNARMATask (uses"
                       " NARMACoefficientsFor) or scale beta to keep beta*N=0.5.";
                throw std::runtime_error(msg.str());
            }

            y_series[t] = y_t;

            running_sum_y += y_t - y_hist[y_head];
            y_hist[y_head] = y_t;
            y_head = (y_head + 1) % N_;

            u_hist[u_head] = u_t;
            u_head = (u_head + 1) % N_;

            y_back = y_t;
        }

        // Return the post-warmup portion. inputs[t] and targets[t] are
        // index-aligned: targets[t] = y(t) is the NARMA output for u(t).
        // When warmup_steps == 0 (the path MakeNARMATask hits -- the task
        // carries the full series intact; the ESN driver consumes the
        // first `warmup` samples via ESN::Warmup before collecting states
        // from the rest, and `targets()` skips the same prefix via a
        // pointer offset), skip the per-element copy and move the full
        // series out.
        if (warmup_steps == 0)
            return {std::move(u_series), std::move(y_series)};

        std::vector<T> inputs(num_steps);
        std::vector<T> targets(num_steps);
        for (size_t t = 0; t < num_steps; ++t)
        {
            inputs[t]  = u_series[warmup_steps + t];
            targets[t] = y_series[warmup_steps + t];
        }
        return {std::move(inputs), std::move(targets)};
    }

private:
    size_t N_;
    T alpha_, beta_, gamma_, delta_, u_low_, u_high_;
    bool tanh_wrap_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<T> u_dist_;
};

/// Top-level knobs for a NARMA task -- order, the signal-side RNG seed handed
/// to the NARMA_N generator, and the tanh-wrap A/B selector.
struct NARMATaskConfig
{
    size_t   narma_order;
    uint64_t data_signal_seed = 1939;
    /// false -> legacy unsquashed NARMA with the order-dependent beta/delta
    /// schedule; true -> tanh-wrapped NARMA with fixed canonical coefficients
    /// (honest order-scaling). See the NARMA_N_Generator recurrence note.
    bool     tanh_wrap        = false;
};

/// Generator coefficients for the NARMA-N recurrence at a given order.
/// `alpha`, `gamma`, `u_low`, `u_high` are constants; `beta` and `delta`
/// are scheduled with N (see NARMACoefficientsFor below).
struct NARMACoefficients
{
    float alpha  = 0.3f;
    float beta   = 0.05f;
    float gamma  = 1.5f;
    float delta  = 0.1f;
    float u_low  = 0.0f;
    float u_high = 0.5f;
};

/// Single source of truth for the order-dependent coefficient schedule.
///
///   delta: drops to 0.01 at N >= 20 (keeps high-order NARMA bounded).
///   beta : scaled to keep beta*N = 0.5 at N >= 24 (recovers the canonical
///          NARMA-10 quadratic; below this threshold the unsquashed fixed
///          point is real and beta stays at 0.05).
inline NARMACoefficients NARMACoefficientsFor(size_t narma_order)
{
    NARMACoefficients c;
    c.delta = (narma_order >= 20) ? 0.01f : 0.1f;
    if (narma_order >= 24)
        c.beta = 0.05f * 10.0f / static_cast<float>(narma_order);
    return c;
}

/// A fully materialized NARMA task: rescaled reservoir input, raw
/// NARMA-N targets, the (warmup, collect, tr, te) sizing, and the
/// generator parameters used to build it (so logs can self-describe).
/// Plain non-template struct -- only the construction is DIM-aware.
///
/// `ri` is the ESN drive signal (already scaled to the reservoir's
/// expected [-1, +1] range). `y_full` carries the matching NARMA
/// outputs; the post-warmup window starts at
/// `targets() = y_full.data() + warmup`.
struct NARMATask
{
    std::vector<float> ri;
    std::vector<float> y_full;
    size_t warmup  = 0;
    size_t collect = 0;
    size_t tr      = 0;
    size_t te      = 0;
    // Generator parameters used to build the series, carried on the task so
    // every consumer reads from a single source instead of re-deriving them.
    size_t              narma_order      = 0;
    uint64_t            data_signal_seed = 0;
    NARMACoefficients   coeffs{};
    bool                tanh_wrap        = false;
    [[nodiscard]] const float* targets() const { return y_full.data() + warmup; }
};

/// Build a NARMA task for hypercube dimension `dim` and `tc.narma_order`.
///
///   - warmup       = explicit, or 0 -> auto (dim < 8, i.e. N < 256 ? 200 : 300)
///   - collect      = explicit (state count fed to the readout); the
///                    train/test split is 80/20 of it
///   - coefficients = tc.tanh_wrap ? canonical fixed (0.3/0.05/1.5/0.1)
///                                  : NARMACoefficientsFor(order) schedule.
///                    The tanh variant keeps coefficients fixed at every order
///                    (tanh bounds the recurrence), so difficulty scales with
///                    order; the legacy variant scales beta/delta down with N
///                    to stay bounded, which smooths high-order series.
///   - ri[t]        = u[t] * 4.0f - 1.0f  (maps the [0, 0.5] input draw
///                    onto [-1, +1], centered on 0)
///
/// The NARMA output y has a non-zero mean; the readout no longer centers
/// targets, so the driver in NARMA.cpp subtracts the train-set mean before
/// training and adds it back at prediction time.
inline NARMATask MakeNARMATask(size_t dim,
                               const NARMATaskConfig& tc,
                               size_t collect,
                               size_t warmup = 0)
{
    // N = 2^dim only feeds the warmup default; test dim directly to avoid the
    // shift (N < 256 <=> dim < 8). The reservoir's own dim range is enforced
    // when RunNARMATrial constructs the ESN.
    if (warmup == 0) warmup = (dim < 8) ? 200 : 300;

    // Fixed canonical coefficients for the tanh variant (the outer squashing
    // keeps it bounded at any order); the order-dependent schedule otherwise.
    const NARMACoefficients c = tc.tanh_wrap ? NARMACoefficients{}
                                             : NARMACoefficientsFor(tc.narma_order);

    NARMA_N_Generator<float> gen(tc.narma_order, tc.data_signal_seed,
                                 c.alpha, c.beta, c.gamma, c.delta,
                                 c.u_low, c.u_high, tc.tanh_wrap);

    auto [u, y] = gen.generate_prediction_task(warmup + collect,
                                               /*warmup_steps=*/0);

    // Rescale u in place from [0, 0.5] to [-1, +1] so we can move it into
    // task.ri without a second allocation. With the warmup_steps==0 fast
    // path in generate_prediction_task, the only series-sized allocations
    // are u_series and y_series themselves -- both moved straight into
    // task.ri and task.y_full instead of being copied into separate buffers.
    for (auto& x : u) x = x * 4.0f - 1.0f;

    NARMATask task;
    task.ri               = std::move(u);
    task.y_full           = std::move(y);
    task.warmup           = warmup;
    task.collect          = collect;
    task.tr               = static_cast<size_t>(collect * 0.8);
    task.te               = collect - task.tr;
    task.narma_order      = tc.narma_order;
    task.data_signal_seed = tc.data_signal_seed;
    task.coeffs           = c;
    task.tanh_wrap        = tc.tanh_wrap;
    return task;
}

/// Outcome of one NARMA trial. Metrics are computed on the held-out test
/// window [tr, tr+te); `test_pred`/`test_actual` carry that window on the
/// raw NARMA scale (predictions de-centered) so callers can print samples
/// without re-driving the reservoir.
struct NARMATrialResult
{
    double nrmse       = 0.0;
    double r2          = 0.0;
    double train_secs  = 0.0;
    double target_mean = 0.0;   // train-set mean subtracted before training
    std::vector<float> test_pred;
    std::vector<float> test_actual;
};

/// Run one NARMA trial end to end. **This is where the ESN is configured:**
/// `cfg` carries the reservoir and readout knobs (including `reservoir.dim`),
/// and `ESN(cfg)` is constructed here.
///
/// Sequence: center targets on the train mean -> ESN(cfg) ->
/// Warmup(ri, warmup) -> Run(ri + warmup, collect) -> Train(centered, tr) ->
/// NRMSE/R2 on [tr, tr+te).
///
/// NARMA output has a small positive mean and the readout no longer centers
/// targets (see docs/Readout.md), so this subtracts the train-set mean
/// before training and adds it back when filling `test_pred`. NRMSE and R2
/// are shift-invariant when target and prediction are shifted together, so
/// the reported metrics are identical in centered or raw space.
inline NARMATrialResult RunNARMATrial(const ESNConfig& cfg, const NARMATask& task)
{
    const float* y = task.targets();

    const double mean = std::accumulate(y, y + task.tr, 0.0)
                        / static_cast<double>(task.tr);

    std::vector<float> centered(task.collect);
    for (size_t t = 0; t < task.collect; ++t)
        centered[t] = y[t] - static_cast<float>(mean);

    ESN esn(cfg);
    esn.Warmup(task.ri.data(), task.warmup);
    esn.Run(task.ri.data() + task.warmup, task.collect);

    const auto t0 = std::chrono::steady_clock::now();
    esn.Train(centered.data(), task.tr);
    const auto t1 = std::chrono::steady_clock::now();

    NARMATrialResult r;
    r.nrmse       = esn.NRMSE(centered.data(), task.tr, task.te);
    r.r2          = esn.R2(centered.data(), task.tr, task.te);
    r.train_secs  = std::chrono::duration<double>(t1 - t0).count();
    r.target_mean = mean;

    r.test_pred.resize(task.te);
    r.test_actual.resize(task.te);
    for (size_t i = 0; i < task.te; ++i)
    {
        const size_t idx = task.tr + i;
        r.test_pred[i]   = esn.PredictRaw(idx) + static_cast<float>(mean);
        r.test_actual[i] = y[idx];
    }
    return r;
}
