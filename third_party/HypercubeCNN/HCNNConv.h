// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

/**
 * @file HCNNConv.h
 * @brief Hypercube convolutional layer — sparse-vertex convolution on a
 *        binary hypercube using fixed XOR masks instead of spatial grids.
 *
 * An HCNNConv layer maps c_in input channels defined on the vertices of a
 * DIM-dimensional binary hypercube (N = 2^DIM vertices) to c_out output
 * channels on the same hypercube.  For each output vertex v, the layer
 * computes:
 *
 *   out_co(v) = b_co + sum over (ci, k) of w[co,ci,k] * in[ci, v ^ (1 << k)]
 *
 * where k ranges over [0, DIM), so each mask is a single-bit flip
 * selecting the nearest neighbor at Hamming distance 1 along bit k.
 *
 * Each mask selects exactly one neighbor per vertex; each gets its own learned
 * weight, shared across all vertices (CNN-style weight sharing).
 *
 * All geometry is bitwise — neighbor lookup uses XOR with single-bit masks;
 * there are no adjacency lists or spatial padding.
 *
 * Memory layout is **channel-major**: element [c*N + v] stores channel c,
 * vertex v.
 */

#pragma once

#include <vector>
#include <random>

namespace hcnn {

class ThreadPool;

/// Activation function applied after convolution (and optional batch normalization).
///
/// - `NONE`: identity, useful when the layer's output feeds directly into a
///   downstream nonlinearity (e.g. an antipodal max-pool that is itself the
///   network's nonlinearity).
/// - `RELU`, `LEAKY_RELU`: standard rectified-linear variants.  Single-sided,
///   non-smooth at zero, fast.  Use He/Kaiming initialization (set
///   automatically by `randomize_weights`).
/// - `TANH`: smooth, symmetric, bounded in (-1, 1).  The standard activation
///   for time-series and recurrent-network workloads (LSTM, GRU, ESN), and
///   the natural choice when HCNN is used as a regression readout for a
///   reservoir whose state is itself tanh-bounded -- the activations of the
///   conv layer then live in the same range as the reservoir state, and the
///   gradient is everywhere smooth (no kink at zero like RELU/LEAKY_RELU,
///   which interacts badly with the antipodal max-pool's already-non-smooth
///   gradient).  Uses Xavier/Glorot initialization.
enum class Activation { NONE, RELU, LEAKY_RELU, TANH };

/// Optimizer for weight updates.
enum class OptimizerType { SGD, ADAM };

/**
 * @class HCNNConv
 * @brief One hypercube convolutional layer.  Maps c_in input channels on a
 *        DIM-dimensional binary hypercube to c_out output channels on the
 *        same hypercube using K = DIM single-bit-flip XOR neighbor masks.
 *
 * Each output channel learns one weight per (input channel, neighbor
 * direction) pair plus an optional bias.  Weight sharing across vertices is
 * exact (the hypercube is vertex-transitive), so there is no padding,
 * border handling, or adjacency table.
 *
 * Owns: kernel + (optional) bias + (optional) batch-norm parameters, plus
 * the matching first / second moment buffers for SGD-momentum or Adam.
 *
 * Configurable per layer:
 *   - activation: NONE / RELU / LEAKY_RELU / TANH
 *   - use_bias: per-output-channel learnable bias
 *   - use_batchnorm: per-channel batch normalization between conv and activation
 *   - optimizer: SGD-with-momentum or Adam (set via set_optimizer)
 *
 * Two backward paths share the same gradient math but differ in where the
 * gradients land:
 *   - backward(): apply gradients in-place via the configured optimizer
 *     (used by single-sample TrainStep)
 *   - compute_gradients() + apply_gradients(): write raw gradients into
 *     caller-provided buffers, then apply once (used by mini-batch
 *     training to accumulate per-sample grads across threads)
 *
 * Threading: an optional ThreadPool parallelizes the inner vertex loop;
 * only kicks in when DIM >= 12, since fork-join overhead dominates below.
 * Disabled automatically during batch-parallel dispatch (LayerThreadGuard).
 *
 * Layout convention: all activation tensors are channel-major,
 * `data[c * N + v]` for channel c, vertex v.
 *
 * Power-user class: ordinary SDK consumers should use HCNN, which builds
 * and owns conv layers internally.
 */
class HCNNConv {
public:
    /**
     * @brief Construct a hypercube convolutional layer.
     *
     * Uses K = DIM nearest-neighbor XOR masks (computed inline).  Kernel and
     * bias weights are initialized to zero; call randomize_weights() before training.
     *
     * Requires dim >= 3 so that K >= 3.
     *
     * @param dim            Hypercube dimension.  The layer operates on N = 2^dim vertices.
     * @param c_in           Number of input channels.
     * @param c_out          Number of output channels (filters).
     * @param activation     Activation function (default: RELU).
     * @param use_bias       If true, add a learnable per-output-channel bias (default: true).
     * @param use_batchnorm  If true, apply batch normalization between conv and activation.
     */
    HCNNConv(int dim, int c_in, int c_out,
             Activation activation = Activation::RELU,
             bool use_bias = true, bool use_batchnorm = false);

    /**
     * @brief Initialize kernel weights.
     *
     * When scale > 0, uses uniform random values in [-scale, +scale].
     * When scale <= 0, auto-selects based on activation and depth:
     *   ReLU/LeakyReLU with c_in > 1: He/Kaiming uniform, s = sqrt(6 / fan_in).
     *   Otherwise (NONE, TANH, or first layer with c_in=1):
     *     Xavier/Glorot uniform, s = sqrt(6 / (fan_in + fan_out)).
     * fan_in = c_in * K, fan_out = c_out * K.
     *
     * Biases are reset to zero.  Momentum velocity buffers are cleared.
     *
     * @param scale  Half-width of the uniform range, or <= 0 for auto init.
     * @param rng    Mersenne Twister PRNG instance (caller-owned).
     */
    void randomize_weights(float scale, std::mt19937& rng);

    /**
     * @brief Execute the forward pass over all output channels.
     *
     * For each output channel and each vertex, looks up K specific neighbors
     * via XOR masks, multiplies by the corresponding kernel weight, sums,
     * adds bias, and applies the activation function.
     *
     * When batch normalization is enabled, normalization is applied between
     * the weighted sum and activation.  In training mode, per-sample statistics
     * are used and running statistics are updated.  In eval mode, running
     * statistics are used.
     *
     * @param[in]  in       Input activations, channel-major [c_in * N].
     * @param[out] out      Output activations, channel-major [c_out * N].
     * @param[out] pre_act  If non-null, receives the pre-activation values
     *                      [c_out * N].  Required by backward().
     * @param[out] bn_save  If non-null and BN enabled, receives per-channel
     *                      inv_std values [c_out].  Required by backward() in
     *                      training mode.
     */
    void forward(const float* in, float* out, float* pre_act = nullptr,
                 float* bn_save = nullptr) const;

    /**
     * @brief Backward pass: compute input gradients and update weights via SGD.
     *
     * Applies the chain rule through the activation function, then:
     *   -# Computes grad_in (if non-null) using the same XOR-lookup structure
     *      as forward (XOR is self-inverse, so the transpose is itself).
     *   -# Updates kernel weights using momentum SGD:
     *      v <- mu*v + g,  w <- w - eta*v
     *   -# Updates bias weights similarly (if bias is enabled).
     *
     * @param[in]  grad_out      Gradient of loss w.r.t. output activations [c_out * N].
     * @param[in]  in            Input activations from the forward pass [c_in * N].
     * @param[in]  pre_act       Pre-activation values from the forward pass [c_out * N].
     * @param[out] grad_in       Gradient of loss w.r.t. input activations [c_in * N],
     *                           or nullptr if not needed (e.g. first layer).
     * @param      learning_rate SGD learning rate (eta).
     * @param      momentum      SGD momentum coefficient (mu); default 0 (no momentum).
     * @param      weight_decay  L2 regularization coefficient; default 0 (no decay).
     * @param[in]  post_act      Optional post-activation values [c_out * N] from the
     *                           forward pass.  When supplied and activation==TANH,
     *                           the activation derivative is computed as 1 - y^2
     *                           from the post-activation, avoiding a redundant
     *                           std::tanh call per element.  Numerically equivalent.
     */
    void backward(const float* grad_out, const float* in, const float* pre_act,
                  float* grad_in, float learning_rate, float momentum = 0.0f,
                  float weight_decay = 0.0f, const float* bn_save = nullptr,
                  int timestep = 0, const float* post_act = nullptr);

    /**
     * @brief Compute gradients without applying an SGD update.
     *
     * Identical to the gradient-computation portion of backward(), but writes
     * raw gradients into caller-provided buffers instead of updating internal
     * weights.  Used for numerical gradient checking.
     *
     * @param[in]  grad_out    Gradient of loss w.r.t. output activations [c_out * N].
     * @param[in]  in          Input activations from the forward pass [c_in * N].
     * @param[in]  pre_act     Pre-activation values from the forward pass [c_out * N].
     * @param[out] grad_in     Gradient of loss w.r.t. input activations [c_in * N],
     *                         or nullptr if not needed.
     * @param[out] kernel_grad Gradient of loss w.r.t. kernel weights [c_out * c_in * K].
     * @param[out] bias_grad   Gradient of loss w.r.t. bias [c_out],
     *                         or nullptr if bias is disabled.
     * @param[in]  post_act    Optional post-activation values [c_out * N] from the
     *                         forward pass.  When supplied and activation==TANH,
     *                         the activation derivative is computed as 1 - y^2
     *                         from the post-activation, avoiding a redundant
     *                         std::tanh call per element.  Numerically equivalent.
     */
    void compute_gradients(const float* grad_out, const float* in, const float* pre_act,
                           float* grad_in, float* kernel_grad, float* bias_grad,
                           float* work_buf = nullptr, const float* bn_save = nullptr,
                           float* bn_gamma_grad = nullptr,
                           float* bn_beta_grad = nullptr,
                           const float* post_act = nullptr) const;

    /**
     * @brief Apply externally computed gradients via momentum SGD.
     *
     * Used by mini-batch training: gradients are computed per-sample via
     * compute_gradients(), averaged across the batch, then applied here.
     *
     * @param kernel_grad  Averaged kernel gradients [c_out * c_in * K].
     * @param bias_grad    Averaged bias gradients [c_out], or nullptr if no bias.
     * @param learning_rate SGD learning rate.
     * @param momentum      SGD momentum coefficient.
     * @param weight_decay  L2 regularization coefficient; default 0.
     */
    void apply_gradients(const float* kernel_grad, const float* bias_grad,
                         float learning_rate, float momentum, float weight_decay = 0.0f,
                         const float* bn_gamma_grad = nullptr,
                         const float* bn_beta_grad = nullptr,
                         int timestep = 0);

    /** @name Accessors */
    ///@{
    int get_dim() const { return DIM; }       ///< Hypercube dimension.
    int get_N() const { return N; }           ///< Vertex count (2^DIM).
    int get_c_in() const { return c_in; }     ///< Number of input channels.
    int get_c_out() const { return c_out; }   ///< Number of output channels.
    int get_K() const { return K; }           ///< Number of connection masks (= DIM).
    ///@}

    /// Set the thread pool for parallel execution (nullptr = single-threaded).
    void set_thread_pool(ThreadPool* pool) { thread_pool = pool; }

    /// Set training mode (true) or eval mode (false) for batch normalization.
    void set_training(bool training) const { training_ = training; }

    /// Current training-mode flag (for RAII save/restore in inference paths).
    bool is_training() const { return training_; }

    /// Skip running-stats EMA updates in forward() (for batch-parallel mode).
    void set_skip_running_stats(bool skip) const { skip_running_stats_ = skip; }

    /// Configure the optimizer. Allocates second-moment buffers for Adam.
    void set_optimizer(OptimizerType type, float beta1 = 0.9f,
                       float beta2 = 0.999f, float eps = 1e-8f);

    /// Whether this layer has batch normalization enabled.
    bool has_batchnorm() const { return use_batchnorm; }

    /// Size of the bn_save buffer needed by forward/backward.
    /// Layout: [inv_std(c_out), mean(c_out), var(c_out)] — 3*c_out if BN, else 0.
    /// backward/compute_gradients only read inv_std (first c_out).
    int get_bn_save_size() const { return use_batchnorm ? 3 * c_out : 0; }

    /// Size of the BN gamma/beta gradient buffers (c_out if BN, else 0).
    int get_bn_grad_size() const { return use_batchnorm ? c_out : 0; }

    /// Update running mean/var from externally computed batch statistics.
    /// Applies Bessel's correction (N/(N-1)) to var before EMA update.
    void update_running_stats(const float* mean, const float* var);

    /** @name Raw weight access (for serialization and gradient checking) */
    ///@{
    float* get_kernel_data() { return kernel.data(); }                           ///< Pointer to kernel weight array.
    const float* get_kernel_data() const { return kernel.data(); }              ///< Const pointer to kernel weight array.
    int get_kernel_size() const { return static_cast<int>(kernel.size()); }      ///< Total kernel weight count.
    float* get_bias_data() { return bias.data(); }                               ///< Pointer to bias array.
    const float* get_bias_data() const { return bias.data(); }                  ///< Const pointer to bias array.
    int get_bias_size() const { return static_cast<int>(bias.size()); }          ///< Bias element count (0 if bias disabled).
    ///@}

private:
    int DIM;          ///< Hypercube dimension.
    int N;            ///< Number of vertices, always 2^DIM.
    int c_in;         ///< Input channel count.
    int c_out;        ///< Output channel count (number of filters).
    int K;            ///< Number of connection masks (= DIM).
    Activation activation;  ///< Activation function applied after convolution.
    bool use_bias;       ///< Whether a learnable bias term is added per output channel.
    bool use_batchnorm;  ///< Whether batch normalization is applied between conv and activation.
    mutable bool training_ = true; ///< Training mode (true) or eval mode (false) for BN.
    mutable bool skip_running_stats_ = false; ///< When true, forward() skips EMA updates (batch-parallel mode).

    std::vector<float> kernel;          ///< Kernel weights, layout [c_out * c_in * K].
    std::vector<float> bias;            ///< Per-output-channel bias, size c_out (empty if bias disabled).
    std::vector<float> kernel_m;        ///< First moment (SGD velocity / Adam m) for kernel.
    std::vector<float> bias_m;          ///< First moment for bias.
    std::vector<float> kernel_m2;       ///< Second moment (Adam only) for kernel.
    std::vector<float> bias_m2;         ///< Second moment (Adam only) for bias.

    // Batch normalization parameters (empty if BN disabled)
    std::vector<float> bn_gamma;          ///< BN scale parameter [c_out].
    std::vector<float> bn_beta;           ///< BN shift parameter [c_out].
    mutable std::vector<float> bn_running_mean; ///< BN running mean [c_out] (mutable: updated in const forward).
    mutable std::vector<float> bn_running_var;  ///< BN running variance [c_out] (mutable: updated in const forward).
    std::vector<float> bn_gamma_m;        ///< First moment for BN gamma [c_out].
    std::vector<float> bn_beta_m;         ///< First moment for BN beta [c_out].
    std::vector<float> bn_gamma_m2;       ///< Second moment (Adam only) for BN gamma [c_out].
    std::vector<float> bn_beta_m2;        ///< Second moment (Adam only) for BN beta [c_out].
    static constexpr float bn_momentum_ = 0.1f;  ///< EMA momentum for running stats.
    static constexpr float bn_eps_ = 1e-5f;      ///< Epsilon for numerical stability.

    // Optimizer configuration
    OptimizerType optimizer_type_ = OptimizerType::SGD;
    float adam_beta1_ = 0.9f, adam_beta2_ = 0.999f, adam_eps_ = 1e-8f;

    std::vector<float> backward_work_;  ///< Persistent scratch for backward() [c_out * N], grown on demand.

    ThreadPool* thread_pool = nullptr;  ///< Optional thread pool for parallel execution.

    /**
     * @brief Compute the flat index into the kernel array.
     * @param co Output channel index.
     * @param ci Input channel index.
     * @param k  Mask index (0 .. K-1).
     * @return   Index into the kernel vector.
     */
    int kernel_idx(int co, int ci, int k) const {
        return (co * c_in + ci) * K + k;
    }

    static constexpr float leaky_alpha_ = 0.01f; ///< LeakyReLU negative slope.

    float activate(float x) const;
    float activate_derivative(float x) const;
};

} // namespace hcnn
