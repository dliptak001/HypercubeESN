// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

#include <vector>
#include <cstdint>

namespace hcnn {

class ThreadPool;

enum class PoolType { MAX, AVG };

/**
 * @class HCNNPool
 * @brief Antipodal pooling layer.  Pairs each vertex `v` with its bitwise
 *        complement `v ^ (2^DIM - 1)` -- the maximally distant vertex on the
 *        hypercube -- and reduces DIM by 1.
 *
 * Reduction is exact: the lower-half vertex survives, the output is a
 * perfect (DIM-1)-dimensional sub-hypercube with N/2 vertices.  No
 * interpolation, no border handling, no overlap.
 *
 * Two reductions:
 *   - PoolType::MAX -- keep the larger of the pair (and remember which one
 *                      survived for the backward pass).
 *   - PoolType::AVG -- arithmetic mean of the pair.
 *
 * Stateless apart from the configured input dimension and reduction type;
 * carries no learnable parameters.  Forward writes max indices into a
 * caller-provided vector when MAX pooling needs them for backward.
 *
 * Power-user class: ordinary SDK consumers should use HCNN.
 */
class HCNNPool {
public:
    HCNNPool(int input_dim, PoolType type = PoolType::MAX);

    void forward(const float* in, float* out, int num_channels,
                 std::vector<int>* max_indices = nullptr) const;

    void backward(const float* grad_out, float* grad_in, int num_channels,
                  const std::vector<int>* max_indices) const;

    int get_input_dim() const { return input_dim; }
    int get_output_dim() const { return output_dim; }
    int get_input_N() const { return input_N; }
    int get_output_N() const { return output_N; }

    void set_thread_pool(ThreadPool* tp) { thread_pool = tp; }

private:
    int input_dim, output_dim, input_N, output_N;
    PoolType type;
    ThreadPool* thread_pool = nullptr;
};

} // namespace hcnn
