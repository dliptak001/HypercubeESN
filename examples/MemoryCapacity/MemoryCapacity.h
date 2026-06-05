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
/// @brief Class-based memory-capacity (MC) diagnostic for the hypercube
/// reservoir. See MemoryCapacity.md for the walkthrough and method definition.
///
/// Implements the standard Jaeger (2001) MC measurement:
///   1. Drive the reservoir with i.i.d. white noise u(t) ~ Uniform[-1, +1].
///   2. Collect M state vectors (rows) into the F-column matrix X.
///   3. Split the rows: first `train_frac` train the ridge readout, rest evaluate.
///   4. For each lag k in [1, k_max], fit (XᵀX + λI)w = Xᵀy on the train rows,
///      then score the squared Pearson r²(k) between target and prediction on
///      the held-out test rows.
///   5. MC = Σ_k r²(k); also report the last lag crossing r² > {0.5, 0.1, 0.01}.
///
/// The reservoir is never modified — only its raw state is read (no HCNN, no
/// ESN coupling). The two configs are kept deliberately separate:
///   - MCConfig          : the *experiment* (drive length, lag range, ridge,
///                         split, feature cap). Fixes a meter; one drive series
///                         is generated up front and reused across operating
///                         points so cells stay byte-comparable.
///   - ReservoirConfig   : the *operating point* (sr, leak, history_depth, seed,
///                         input_scaling). Passed into Measure() per cell — there
///                         is no parallel "extended parameter list".
///
/// Held-out evaluation matters: in-sample R² on M ~ a few × F samples
/// overestimates the population r² by a roughly lag-independent margin,
/// inflating the headline MC and every threshold crossing. We pay one Cholesky
/// factorization (over the train Gram) and do the per-lag evaluation on the
/// test split. We use squared Pearson correlation rather than the regression
/// R² = 1 - SS_res/SS_tot: they agree when the model has an intercept; our
/// linear readout has none, so the Pearson form is the canonical MC metric and
/// is always in [0, 1].
///
/// Cost: dominated by building the F×F train Gram and factoring it —
/// O(M_train·F² + F³). At DIM 12 with F=2048, M_train ~ 10k, one Measure() is
/// roughly 5–10 s in Release.

namespace mc
{
    /// Measurement parameters for an MC run. These define the *experiment*,
    /// independent of the reservoir operating point. A meter is built once from
    /// an MCConfig and reused across many ReservoirConfigs.
    struct MCConfig
    {
        std::size_t   feature_cap = 8192;       ///< cap on # reservoir features used as regressors
        std::size_t   t_warmup    = 2000;       ///< steps fed before any state is recorded
        std::size_t   t_collect   = 15000;      ///< post-warmup steps whose state is collected
        std::size_t   k_max       = 2000;       ///< largest lag tested
        double        train_frac  = 0.7;        ///< fraction of usable rows used to fit the readout
        double        ridge_lambda = 1e-4;      ///< Tikhonov regularization on the train Gram diagonal
        std::uint64_t input_seed  = 0xC0FFEEULL;///< white-noise drive RNG seed

        // Per-lag early-stop (Measure with early_stop=true). The memory function
        // decays with lag, so once r²(k) holds below early_stop_thresh for
        // early_stop_patience consecutive lags the sweep stops: remaining lags
        // contribute only noise-floor r² and never revive a decayed curve. The
        // patience window guards the headline against truncating on a transient
        // dip. A full-curve run (early_stop=false) ignores these.
        double        early_stop_thresh   = 0.01; ///< streak threshold on r²(k)
        std::size_t   early_stop_patience = 20;   ///< consecutive sub-threshold lags that end the sweep

        /// k_max may equal t_warmup: the earliest lag-k_max target references
        /// u[t_warmup] (the first post-warmup input, never a negative index), and
        /// the first kept state sits at step t_warmup+k_max, so the reservoir has
        /// had ≥t_warmup steps to forget its initial condition. Testing lags
        /// longer than the warmup is the only thing this rules out.
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

    /// Result of one MC measurement at a single operating point.
    struct MCResult
    {
        double total_mc    = 0.0;   ///< Σ_k r²(k) over computed lags
        float  realized_sr = 0.0f;  ///< realized post-rescale spectral radius
        int    k50 = 0;             ///< last lag with r² > 0.50
        int    k10 = 0;             ///< last lag with r² > 0.10
        int    k01 = 0;             ///< last lag with r² > 0.01
        bool   pd  = true;          ///< false if the train Gram was not positive-definite
        bool   oom = false;         ///< true if the cell threw std::bad_alloc (skipped)
        std::vector<double> r2;     ///< per-lag r²(k) at index k-1; lags past early-stop stay 0
    };

    /// Options for a single Measure() call.
    struct MeasureOptions
    {
        bool        early_stop = true; ///< stop after a sub-threshold streak (sweeps); false = full curve
        std::size_t kmax       = 0;    ///< lags to probe; 0 -> MCConfig::k_max
    };

    /// Memory-capacity meter for a hypercube reservoir of dimension `dim`.
    ///
    /// Construct once with a dim and an MCConfig: the white-noise drive (which
    /// depends only on input_seed / t_warmup / t_collect) is generated up front
    /// and reused by every Measure() call, so a whole sweep over reservoir
    /// operating points shares one drive sequence. Measure() is const and
    /// allocates all working buffers locally, so the same meter can be measured
    /// concurrently from many threads — which is exactly what RunSweep relies on.
    class MemoryCapacityMeter
    {
    public:
        /// @param dim Hypercube dimension; N = 2^dim, and the meter's feature
        ///        layout (F = min(N, feature_cap)) is fixed from it. Reservoir is
        ///        the authoritative range check; this mirrors its [5, 16] bound as
        ///        a fail-fast guard so an out-of-range dim can neither reach the
        ///        2^dim shift (UB for dim >= 64) nor surface from a sweep worker.
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
        [[nodiscard]] std::size_t     Size()     const { return N_; }
        [[nodiscard]] std::size_t     Features() const { return F_; }
        [[nodiscard]] std::size_t     Samples()  const { return M_; }
        [[nodiscard]] std::size_t     TrainRows() const { return M_train_; }
        [[nodiscard]] std::size_t     TestRows()  const { return M_test_; }

        /// Peak per-call working set (X + Gram), in bytes — for RAM-budgeted
        /// sweep sizing. The reservoir's own history_depth-scaled state is
        /// negligible beside X and G.
        [[nodiscard]] std::size_t PerCellBytes() const
        {
            return (M_ * F_ + F_ * F_) * sizeof(double);
        }

        /// Measure MC at one reservoir operating point. `rcfg` carries the
        /// operating point (sr, leak, history_depth, seed, input_scaling);
        /// everything about the *experiment* (drive, split, ridge, lag range) is
        /// fixed by the MCConfig this meter was built with. num_inputs and verbose
        /// are forced (single-channel probe, silent) regardless of `rcfg`.
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

                if (opts.early_stop)
                {
                    if (r2_k < cfg_.early_stop_thresh)
                    {
                        if (++below_streak >= cfg_.early_stop_patience) break;
                    }
                    else
                        below_streak = 0;
                }
            }

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

        /// Drive the reservoir over warmup+collect steps, writing post-warmup,
        /// post-k_max states into X (row-major, M_ rows × F_ features). Returns
        /// the realized (post-rescale) spectral radius. dim/num_inputs/verbose are
        /// forced here so a caller's rcfg can't desync the feature layout, break
        /// the single-channel probe, or garble stdout during concurrent construction.
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

        /// Squared Pearson correlation between target y and prediction X·w on the
        /// held-out test rows [M_train_, M_):
        ///   r² = (n·Σyŷ - Σy·Σŷ)² / ((n·Σy² - (Σy)²)·(n·Σŷ² - (Σŷ)²))
        /// Always in [0, 1]; the canonical MC metric.
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
        std::vector<float> u_; ///< shared white-noise drive (length t_warmup + t_collect)
    };

    /// Options for a parallel sweep.
    struct SweepOptions
    {
        std::size_t max_workers   = 0;   ///< 0 -> hardware concurrency
        double      ram_budget_gb = 0.0; ///< 0 -> no RAM cap; else cap workers so peak fits
    };

    /// Resolve the worker count for a sweep: min(cells, max_workers or hardware),
    /// optionally further capped so the estimated peak (workers × per_cell_bytes)
    /// fits ram_budget_gb. Exposed so a driver can print the same count it will
    /// actually run with.
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

    /// Run meter.Measure(cfg) for every cfg in `configs`, concurrently, returning
    /// one MCResult per config in input order. Cells are independent (Measure is
    /// const, all buffers local), so workers self-schedule off a shared atomic
    /// counter until the list is exhausted — no static partitioning, so uneven
    /// cell costs (not-PD/early-stop cells finish fast) don't idle a worker. A
    /// cell that throws std::bad_alloc is caught and flagged oom=true rather than
    /// aborting the sweep. If supplied, `progress(done, total)` is called under a
    /// lock after each cell completes.
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
