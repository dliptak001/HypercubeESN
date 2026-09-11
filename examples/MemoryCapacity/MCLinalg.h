#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

/// @file MCLinalg.h
/// @brief Dense double-precision linear-algebra kernels for the memory-capacity
/// diagnostic: a cache-blocked Gram build, an in-place Cholesky factorization,
/// and the matching triangular solve.
///
/// These are pure free functions (no state, no I/O) operating on row-major
/// double buffers. They are the numeric core shared by every MC measurement;
/// keeping them here lets MemoryCapacityMeter read as the *experiment* and these
/// read as the *math*.

namespace mc
{
    /// In-place lower-triangular Cholesky factorization of an n×n symmetric
    /// positive-definite matrix stored row-major. After return, the lower
    /// triangle of `G` holds L such that the original G = L · Lᵀ. Returns
    /// false if a non-positive pivot is encountered.
    inline bool CholeskyInPlace(double* G, std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t j = 0; j <= i; ++j)
            {
                double s = G[i * n + j];
                for (std::size_t k = 0; k < j; ++k)
                    s -= G[i * n + k] * G[j * n + k];
                if (i == j)
                {
                    if (s <= 0.0) return false;
                    G[i * n + i] = std::sqrt(s);
                }
                else
                {
                    G[i * n + j] = s / G[j * n + j];
                }
            }
        }
        return true;
    }

    /// Solve L · Lᵀ · x = b in place (b becomes x). L is the lower-triangular
    /// factor produced by CholeskyInPlace stored in the lower triangle of `L`.
    inline void CholeskySolveInPlace(const double* L, double* b, std::size_t n)
    {
        // Forward substitution: L·y = b.
        for (std::size_t i = 0; i < n; ++i)
        {
            double s = b[i];
            for (std::size_t j = 0; j < i; ++j)
                s -= L[i * n + j] * b[j];
            b[i] = s / L[i * n + i];
        }
        // Backward substitution: Lᵀ·x = y.
        for (std::size_t i = n; i > 0; --i)
        {
            const std::size_t ii = i - 1;
            double s = b[ii];
            for (std::size_t j = ii + 1; j < n; ++j)
                s -= L[j * n + ii] * b[j];
            b[ii] = s / L[ii * n + ii];
        }
    }

    /// Build G = Xᵀ·X (lower triangle, then mirrored) for X stored row-major as M
    /// rows of F features.
    ///
    /// Cache-blocked: each BS×BS block of G is accumulated to completion over all M
    /// samples while it sits resident in cache, instead of the naive rank-1 form
    /// that sweeps the entire F×F (here 134 MB) matrix once per sample. That naive
    /// form is memory-bandwidth-bound — ~O(M·F²) bytes of DRAM traffic — so several
    /// cells running concurrently saturate the memory bus and stall on it rather
    /// than using their cores. Blocking cuts traffic to ~O(M·F²/BS) (the X strips
    /// are re-read per block-pair; G is written once) and makes the kernel
    /// compute-bound, so cell-level workers actually run flat-out.
    ///
    /// Each G[i][j] is still summed over t in ascending order, so the result is
    /// bit-identical to the naive accumulation.
    inline void BuildGram(const double* X, std::size_t M, std::size_t F, double* G)
    {
        constexpr std::size_t BS = 128; // BS*BS doubles (128 KB) stays resident in L2
        std::vector<double> tile(BS * BS);

        for (std::size_t ii = 0; ii < F; ii += BS)
        {
            const std::size_t bi_n = std::min(ii + BS, F) - ii;
            for (std::size_t jj = 0; jj <= ii; jj += BS) // lower-triangle blocks only
            {
                const std::size_t bj_n = std::min(jj + BS, F) - jj;
                const bool diag = (ii == jj);

                std::fill(tile.begin(), tile.begin() + bi_n * bj_n, 0.0);

                for (std::size_t t = 0; t < M; ++t)
                {
                    const double* xi_strip = X + t * F + ii;
                    const double* xj_strip = X + t * F + jj;
                    for (std::size_t bi = 0; bi < bi_n; ++bi)
                    {
                        const double xi = xi_strip[bi];
                        double* trow = tile.data() + bi * bj_n;
                        const std::size_t bj_max = diag ? bi + 1 : bj_n; // j<=i on the diagonal block
                        for (std::size_t bj = 0; bj < bj_max; ++bj)
                            trow[bj] += xi * xj_strip[bj];
                    }
                }

                // Scatter the finished tile into G's lower triangle.
                for (std::size_t bi = 0; bi < bi_n; ++bi)
                {
                    double* Grow = G + (ii + bi) * F + jj;
                    const double* trow = tile.data() + bi * bj_n;
                    const std::size_t bj_max = diag ? bi + 1 : bj_n;
                    for (std::size_t bj = 0; bj < bj_max; ++bj)
                        Grow[bj] = trow[bj];
                }
            }
        }

        // Mirror the lower triangle into the upper.
        for (std::size_t i = 0; i < F; ++i)
            for (std::size_t j = 0; j < i; ++j)
                G[j * F + i] = G[i * F + j];
    }

    /// xty[f] = sum_t X[t,f] * y[t], over the first M rows of X (row-major, F wide).
    inline void ComputeXtY(const double* X, const double* y,
                           std::size_t M, std::size_t F, double* xty)
    {
        std::fill(xty, xty + F, 0.0);
        for (std::size_t t = 0; t < M; ++t)
        {
            const double yt = y[t];
            const double* xt = X + t * F;
            for (std::size_t f = 0; f < F; ++f)
                xty[f] += xt[f] * yt;
        }
    }
} // namespace mc
