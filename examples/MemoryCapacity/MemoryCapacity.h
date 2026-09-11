#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <new>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "Reservoir.h"
#include "MCLinalg.h"

/// @file MemoryCapacity.h
/// @brief Jaeger (2001) **linear short-term memory capacity** (MC) on a
/// HypercubeESN @ref Reservoir — ridge on raw state only (no HCNN, no @ref ESN).
///
/// Walkthrough + archived grids: `examples/MemoryCapacity/MemoryCapacity.md`
/// (and `MemoryCapacity_grids.md`).
///
/// ## What MC is
/// How much of the recent white-noise input history can be linearly reconstructed
/// from the present reservoir state. Drive with i.i.d. `u(t) ~ U[-1,+1]`; for each
/// lag k fit a ridge map from state to `u(t-k)`; score held-out squared Pearson
/// correlation r²(k); **TotalMC = sum_k r²(k)**. Ceiling is F (feature count),
/// not unlimited.
///
/// ## Protocol (one Measure)
/// 1. Warm up, then collect usable state rows into X (row-major, usable_rows × F).
/// 2. Split rows: first train_frac fit the readout; remainder evaluate.
/// 3. One train Gram factor: (X_train' X_train + λI); per lag k solve for w and
///    score r²(k) on the test rows.
/// 4. Report TotalMC, threshold lags (k50/k10/k01), early-stop / open-tail flags.
///
/// ## Vocabulary (easy to confuse)
/// - **history_depth** — reservoir delay-line depth M in @ref ReservoirConfig
///   (op-point). Not the row count of X.
/// - **usable rows** (`Samples()`) — t_collect − k_max; train+test rows in X.
///   Banners may call this M_usable; it is *not* history_depth.
/// - **F** — features = min(N, feature_cap), N = 2^dim. Full-state → F = N.
///
/// ## Two configs (deliberately split)
/// - @ref MCConfig — the *experiment* / meter (drive length, lags, ridge, split,
///   feature cap, early-stop). One drive series is generated at meter construction
///   and reused so every op-point cell sees a byte-identical task.
/// - @ref ReservoirConfig — the *operating point* (spectral_radius, leak,
///   history_depth, seed, input_scaling). Passed into @ref MemoryCapacityMeter::Measure
///   per cell.
///
/// ## Metric choices
/// Held-out Pearson r² (no intercept in the linear map) is the canonical MC score
/// and stays in [0, 1]. In-sample scores would overestimate. Theoretical ceiling
/// is F; open_tail means TotalMC is only a lower bound (curve had not decayed).
///
/// ## Cost
/// Dominated by the F×F train Gram and Cholesky — O(usable_train · F² + F³).
/// DIM 11–12 (F=2048–4096) is seconds per cell in Release. Use @ref RunSweep for
/// grids; @ref MemoryCapacityMeter::PerCellBytes for RAM-capped workers.

namespace mc
{
    /// @brief Experiment knobs for MC — independent of reservoir op-point.
    ///
    /// Build one @ref MemoryCapacityMeter from this; reuse it across many
    /// @ref ReservoirConfig cells so the white-noise drive stays fixed.
    struct MCConfig
    {
        std::size_t   feature_cap = 8192;       ///< Max features F = min(N, cap); full-state uses F = N when N ≤ cap
        std::size_t   t_warmup    = 2000;       ///< Steps before any state is recorded (also ≥ k_max)
        std::size_t   t_collect   = 15000;      ///< Post-warmup steps considered for collection
        std::size_t   k_max       = 2000;       ///< Largest lag k tested; usable rows = t_collect − k_max
        double        train_frac  = 0.7;        ///< Fraction of usable rows for ridge fit (rest = test)
        double        ridge_lambda = 1e-4;      ///< Tikhonov λ on the train Gram diagonal
        std::uint64_t input_seed  = 0xC0FFEEULL;///< RNG seed for the shared white-noise drive

        /// Early-stop (when @ref MeasureOptions::early_stop is true): stop after
        /// early_stop_patience consecutive lags with r²(k) < early_stop_thresh.
        /// Decayed curves do not revive; patience avoids cutting on a transient dip.
        /// Full-curve runs (early_stop=false) ignore these fields.
        double        early_stop_thresh   = 0.01; ///< r² streak threshold
        std::size_t   early_stop_patience = 20;   ///< consecutive sub-threshold lags

        /// Requires k_max > 0, k_max ≤ t_warmup, k_max < t_collect, train_frac in (0,1),
        /// feature_cap > 0. k_max may equal t_warmup: the earliest lag-k_max target
        /// still indexes a non-negative input, and the first kept state sits after
        /// enough washout.
        void Validate() const
        {
            if (k_max == 0)
                throw std::invalid_argument("MCConfig: k_max must be > 0");
            if (k_max > t_warmup)
                throw std::invalid_argument("MCConfig: k_max must not exceed t_warmup");
            if (k_max >= t_collect)
                throw std::invalid_argument("MCConfig: k_max must be smaller than t_collect");
            if (train_frac <= 0.0 || train_frac >= 1.0)
                throw std::invalid_argument("MCConfig: train_frac must be in (0, 1)");
            if (feature_cap == 0)
                throw std::invalid_argument("MCConfig: feature_cap must be > 0");
        }
    };

    /// @brief Outcome of one @ref MemoryCapacityMeter::Measure at a single op-point.
    struct MCResult
    {
        double total_mc    = 0.0;   ///< Σ r²(k) over scored lags only (lower bound if open_tail)
        float  realized_sr = 0.0f;  ///< Post-rescale spectral radius from construction
        int    k50 = 0;             ///< Last lag with r² > 0.50 (0 if none)
        int    k10 = 0;             ///< Last lag with r² > 0.10
        int    k01 = 0;             ///< Last lag with r² > 0.01
        bool   pd  = true;          ///< false → train Gram not PD; no lags scored
        bool   oom = false;         ///< true if the cell hit bad_alloc (sweep continues)
        std::size_t lags_scored = 0; ///< Number of lags actually scored (1..k_max)
        bool   early_stopped = false; ///< Lag loop exited on a sub-threshold streak
        double r2_tail = 0.0;       ///< r² at the last scored lag
        /// Last scored r² still ≥ early_stop_thresh: TotalMC is a **lower bound**
        /// (curve had not decayed by k_max / early exit). Closed tail → complete sum.
        bool   open_tail = false;
        std::vector<double> r2;     ///< r²(k) at index k-1; unscored lags remain 0
    };

    /// Per-call options for @ref MemoryCapacityMeter::Measure.
    struct MeasureOptions
    {
        bool        early_stop = true; ///< true for sweeps; false = full curve to kmax
        std::size_t kmax       = 0;    ///< 0 → use MCConfig::k_max (capped at that value)
    };

    /// @brief Fixed-drive MC meter for hypercube dimension @p dim.
    ///
    /// ## Lifecycle
    /// Construct once with dim + @ref MCConfig (generates the white-noise drive).
    /// Call @ref Measure for each @ref ReservoirConfig op-point. The drive is
    /// shared so grid cells are byte-comparable.
    ///
    /// ## Thread safety
    /// Measure is const and allocates working buffers on the stack/heap of the
    /// caller thread — concurrent Measure on one meter is supported (@ref RunSweep).
    ///
    /// ## Layout
    /// N = 2^dim, F = min(N, feature_cap), usable rows = t_collect − k_max.
    /// Measure forces rcfg.dim / num_inputs=1 / verbose=false so the feature
    /// layout cannot desync from the meter.
    class MemoryCapacityMeter
    {
    public:
        /// @param dim Hypercube dimension in **[5, 16]** (N = 2^dim). Fail-fast here
        ///        so an out-of-range dim never hits a bad shift or a silent sweep cell.
        /// @param cfg Experiment knobs; copied and validated via MCConfig::Validate.
        /// @throws std::invalid_argument if dim or cfg is out of range.
        MemoryCapacityMeter(std::size_t dim, const MCConfig& cfg)
            : cfg_(cfg), dim_(dim)
        {
            if (dim < 5 || dim > 16)
                throw std::invalid_argument("MemoryCapacityMeter: dim must be in [5, 16]");
            N_       = std::size_t{1} << dim_;
            cfg_.Validate();
            F_       = std::min<std::size_t>(N_, cfg_.feature_cap);
            M_       = cfg_.t_collect - cfg_.k_max;
            M_train_ = static_cast<std::size_t>(static_cast<double>(M_) * cfg_.train_frac);
            M_test_  = M_ - M_train_;
            GenerateDrive();
        }

        [[nodiscard]] const MCConfig& Config()   const { return cfg_; }
        [[nodiscard]] std::size_t     Dim()      const { return dim_; }
        [[nodiscard]] std::size_t     Size()     const { return N_; }       ///< N = 2^dim
        [[nodiscard]] std::size_t     Features() const { return F_; }       ///< F ≤ N
        [[nodiscard]] std::size_t     Samples()  const { return M_; }       ///< usable rows (not history_depth)
        [[nodiscard]] std::size_t     TrainRows() const { return M_train_; }
        [[nodiscard]] std::size_t     TestRows()  const { return M_test_; }

        /// Peak per-call heap for X + Gram (bytes) — for RAM-budgeted @ref RunSweep.
        [[nodiscard]] std::size_t PerCellBytes() const
        {
            return (M_ * F_ + F_ * F_) * sizeof(double);
        }

        /// @brief TotalMC (and lag curve) at one reservoir operating point.
        ///
        /// @param rcfg Op-point: spectral_radius, leak_rate, history_depth, seed,
        ///        input_scaling (and any other ReservoirConfig fields). dim /
        ///        num_inputs / verbose are **overwritten** by the meter.
        /// @param opts Early-stop and optional kmax cap.
        /// @return @ref MCResult; pd=false or oom may leave TotalMC at 0.
        MCResult Measure(const ReservoirConfig& rcfg, const MeasureOptions& opts = {}) const
        {
            const std::size_t kmax =
                (opts.kmax == 0 || opts.kmax > cfg_.k_max) ? cfg_.k_max : opts.kmax;

            MCResult r;
            r.r2.assign(cfg_.k_max, 0.0);

            // ---- 1-2. Drive reservoir, collect state matrix X ----
            std::vector<double> X(M_ * F_);
            r.realized_sr = DriveAndCollect(rcfg, X.data());

            // ---- 3. Build & factor the regularized train-Gram matrix ----
            // Gram is built on the first M_train rows only — the readout is fit on
            // those, and the held-out last M_test rows are used to score r².
            std::vector<double> G(F_ * F_);
            BuildGram(X.data(), M_train_, F_, G.data());
            for (std::size_t i = 0; i < F_; ++i) G[i * F_ + i] += cfg_.ridge_lambda;
            if (!CholeskyInPlace(G.data(), F_)) { r.pd = false; return r; }

            // ---- 4. Per-lag: fit on train rows, score r² on test rows ----
            std::vector<double> y(M_), w(F_);
            std::size_t below_streak = 0;
            for (std::size_t k = 1; k <= kmax; ++k)
            {
                // Targets across all M rows: for sample row `row` (state at step
                // t_warmup + k_max + row), the lag-k target is the input at step
                // t_warmup + k_max + row - k.
                for (std::size_t row = 0; row < M_; ++row)
                    y[row] = static_cast<double>(u_[cfg_.t_warmup + cfg_.k_max + row - k]);

                // Fit on train rows: w = (Xtrainᵀ Xtrain + λI)⁻¹ Xtrainᵀ ytrain.
                ComputeXtY(X.data(), y.data(), M_train_, F_, w.data());
                CholeskySolveInPlace(G.data(), w.data(), F_);

                const double r2_k = ScoreR2(X.data(), w.data(), y.data());
                r.r2[k - 1] = r2_k;
                r.total_mc += r2_k;
                r.lags_scored = k;
                r.r2_tail = r2_k;

                if (opts.early_stop)
                {
                    if (r2_k < cfg_.early_stop_thresh)
                    {
                        if (++below_streak >= cfg_.early_stop_patience)
                        {
                            r.early_stopped = true;
                            break;
                        }
                    }
                    else
                        below_streak = 0;
                }
            }

            // Open tail: memory still above the decay floor at the last scored lag.
            // With early_stop this is almost always false (streak ended below thresh).
            // Without early_stop (or if k_max is hit while r² is still high), TotalMC
            // is a lower bound — the sum was cut off before the curve decayed.
            r.open_tail = (r.lags_scored > 0) &&
                          (r.r2_tail >= cfg_.early_stop_thresh);

            r.k50 = LastAbove(r.r2, 0.50);
            r.k10 = LastAbove(r.r2, 0.10);
            r.k01 = LastAbove(r.r2, 0.01);
            return r;
        }

    private:
        void GenerateDrive()
        {
            u_.resize(cfg_.t_warmup + cfg_.t_collect);
            std::mt19937_64 rng(cfg_.input_seed);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto& v : u_) v = dist(rng);
        }

        /// Warmup + collect; write usable states into X (row-major, Samples()×F).
        /// Returns realized spectral radius. Forces dim / num_inputs=1 / verbose.
        float DriveAndCollect(const ReservoirConfig& rcfg_in, double* X) const
        {
            ReservoirConfig rcfg = rcfg_in;
            rcfg.dim        = dim_;
            rcfg.num_inputs = 1;
            rcfg.verbose    = false;

            auto reservoir = Reservoir::Create(rcfg);
            const float realized = reservoir->GetRealizedSpectralRadius();

            for (std::size_t t = 0; t < cfg_.t_warmup + cfg_.t_collect; ++t)
            {
                reservoir->InjectInput(0, u_[t]);
                reservoir->Step();
                if (t < cfg_.t_warmup) continue;
                const std::size_t row = t - cfg_.t_warmup;
                if (row < cfg_.k_max) continue; // discard first k_max rows; align with lag targets
                const std::size_t out_row = row - cfg_.k_max;
                const float* state = reservoir->Outputs();
                double* dst = X + out_row * F_;
                for (std::size_t f = 0; f < F_; ++f)
                    dst[f] = static_cast<double>(state[f]);
            }
            return realized;
        }

        /// Held-out squared Pearson r² between y and X·w on test rows — in [0, 1].
        double ScoreR2(const double* X, const double* w, const double* y) const
        {
            double sum_y = 0.0, sum_h = 0.0, sum_y2 = 0.0, sum_h2 = 0.0, sum_yh = 0.0;
            for (std::size_t t = 0; t < M_test_; ++t)
            {
                const double* xt = X + (M_train_ + t) * F_;
                double yhat = 0.0;
                for (std::size_t f = 0; f < F_; ++f) yhat += xt[f] * w[f];
                const double yt = y[M_train_ + t];
                sum_y += yt;
                sum_h += yhat;
                sum_y2 += yt * yt;
                sum_h2 += yhat * yhat;
                sum_yh += yt * yhat;
            }
            const double n = static_cast<double>(M_test_);
            const double num = n * sum_yh - sum_y * sum_h;
            const double den_y = n * sum_y2 - sum_y * sum_y;
            const double den_h = n * sum_h2 - sum_h * sum_h;
            return (den_y > 0.0 && den_h > 0.0) ? (num * num) / (den_y * den_h) : 0.0;
        }

        static int LastAbove(const std::vector<double>& r2, double thresh)
        {
            int last = 0;
            for (std::size_t i = 0; i < r2.size(); ++i)
                if (r2[i] > thresh) last = static_cast<int>(i + 1);
            return last;
        }

        MCConfig          cfg_;
        std::size_t       dim_ = 0, N_ = 0;
        std::size_t       F_ = 0, M_ = 0, M_train_ = 0, M_test_ = 0;
        std::vector<float> u_; ///< Shared white-noise drive (length t_warmup + t_collect)
    };

    /// Parallel @ref RunSweep knobs.
    struct SweepOptions
    {
        std::size_t max_workers   = 0;   ///< 0 → hardware_concurrency
        double      ram_budget_gb = 0.0; ///< 0 → no RAM cap; else limit workers by PerCellBytes
    };

    /// Worker count for a sweep: min(cells, max_workers or HW), optionally capped
    /// so workers × per_cell_bytes ≤ ram_budget_gb. Exposed so banners can print
    /// the same count RunSweep will use.
    inline std::size_t ResolveWorkerCount(std::size_t cells, std::size_t per_cell_bytes,
                                          const SweepOptions& opts)
    {
        if (cells == 0) return 0;
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        std::size_t workers = std::min<std::size_t>(cells, opts.max_workers ? opts.max_workers : hw);
        if (opts.ram_budget_gb > 0.0 && per_cell_bytes > 0)
        {
            const std::size_t by_ram = std::max<std::size_t>(
                1, static_cast<std::size_t>(opts.ram_budget_gb * 1e9 /
                                            static_cast<double>(per_cell_bytes)));
            workers = std::min(workers, by_ram);
        }
        return workers;
    }

    /// @brief Parallel @ref MemoryCapacityMeter::Measure over many op-points.
    ///
    /// One @ref MCResult per config, **in input order**. Workers self-schedule from
    /// an atomic counter (good when early-stop cells finish unevenly).
    /// std::bad_alloc in a cell → that result has oom=true; the sweep continues.
    /// Optional progress(done, total) is called under a mutex after each cell.
    std::vector<MCResult> RunSweep(const MemoryCapacityMeter& meter,
                                   const std::vector<ReservoirConfig>& configs,
                                   const SweepOptions& opts = {},
                                   const std::function<void(std::size_t, std::size_t)>& progress = {})
    {
        const std::size_t cells = configs.size();
        std::vector<MCResult> results(cells);
        if (cells == 0) return results;

        const std::size_t workers = ResolveWorkerCount(cells, meter.PerCellBytes(), opts);

        std::atomic<std::size_t> next{0};
        std::mutex mtx;
        std::size_t done = 0; // guarded by mtx — keeps the progress count monotonic

        auto worker = [&]
        {
            std::size_t idx;
            while ((idx = next.fetch_add(1)) < cells)
            {
                try { results[idx] = meter.Measure(configs[idx]); }
                catch (const std::bad_alloc&) { results[idx].oom = true; }
                if (progress)
                {
                    std::lock_guard<std::mutex> lk(mtx);
                    progress(++done, cells);
                }
            }
        };

        std::vector<std::thread> pool;
        pool.reserve(workers > 0 ? workers - 1 : 0);
        for (std::size_t t = 1; t < workers; ++t) pool.emplace_back(worker);
        worker(); // calling thread participates
        for (auto& th : pool) th.join();
        return results;
    }
} // namespace mc
