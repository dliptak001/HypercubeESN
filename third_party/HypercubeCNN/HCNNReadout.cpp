// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#include "HCNNReadout.h"
#include <algorithm>
#include <cmath>

namespace hcnn {

HCNNReadout::HCNNReadout(int nc, int ic)
    : num_outputs(nc), input_channels(ic),
      weights(nc * ic, 0.0f), bias(nc, 0.0f),
      weight_m(nc * ic, 0.0f), bias_m(nc, 0.0f) {}

void HCNNReadout::randomize_weights(float scale, std::mt19937& rng) {
    // Xavier/Glorot uniform for the linear layer.
    if (scale <= 0.0f) {
        scale = std::sqrt(6.0f / static_cast<float>(input_channels + num_outputs));
    }
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto& w : weights) {
        w = dist(rng);
    }
    for (auto& b : bias) b = 0.0f;
    std::fill(weight_m.begin(), weight_m.end(), 0.0f);
    std::fill(bias_m.begin(), bias_m.end(), 0.0f);
    std::fill(weight_m2.begin(), weight_m2.end(), 0.0f);
    std::fill(bias_m2.begin(), bias_m2.end(), 0.0f);
}

void HCNNReadout::set_optimizer(OptimizerType type, float beta1, float beta2, float eps) {
    optimizer_type_ = type;
    adam_beta1_ = beta1;
    adam_beta2_ = beta2;
    adam_eps_ = eps;
    if (type == OptimizerType::ADAM) {
        weight_m2.assign(weights.size(), 0.0f);
        bias_m2.assign(bias.size(), 0.0f);
    } else {
        weight_m2.clear(); weight_m2.shrink_to_fit();
        bias_m2.clear(); bias_m2.shrink_to_fit();
    }
}

void HCNNReadout::forward(const float* in, float* out, int N,
                          float* work_buf) const {
    std::vector<float> avg_storage;
    float* channel_avg;
    if (work_buf) {
        channel_avg = work_buf;
    } else {
        avg_storage.resize(input_channels, 0.0f);
        channel_avg = avg_storage.data();
    }
    for (int c = 0; c < input_channels; ++c) {
        const float* chan = in + c * N;
        double sum = 0.0;
        for (int v = 0; v < N; ++v) sum += chan[v];
        channel_avg[c] = static_cast<float>(sum / N);
    }

    for (int cls = 0; cls < num_outputs; ++cls) {
        float sum = bias[cls];
        for (int c = 0; c < input_channels; ++c) {
            sum += weights[cls * input_channels + c] * channel_avg[c];
        }
        out[cls] = sum;
    }
}

void HCNNReadout::backward(const float* grad_logits, const float* in, int N,
                           float* grad_in, float learning_rate, float momentum,
                           float weight_decay, int timestep,
                           float* work_buf) {
    const bool use_adam = (optimizer_type_ == OptimizerType::ADAM && timestep > 0);
    const float bc1 = use_adam ? 1.0f - static_cast<float>(std::pow(adam_beta1_, timestep)) : 1.0f;
    const float bc2 = use_adam ? 1.0f - static_cast<float>(std::pow(adam_beta2_, timestep)) : 1.0f;
    std::vector<float> avg_storage;
    float* channel_avg;
    if (work_buf) {
        channel_avg = work_buf;
    } else {
        avg_storage.resize(input_channels);
        channel_avg = avg_storage.data();
    }
    for (int c = 0; c < input_channels; ++c) {
        double sum = 0.0;
        for (int v = 0; v < N; ++v) sum += in[c * N + v];
        channel_avg[c] = static_cast<float>(sum / N);
    }

    if (grad_in) {
        for (int c = 0; c < input_channels; ++c) {
            float g = 0.0f;
            for (int cls = 0; cls < num_outputs; ++cls) {
                g += grad_logits[cls] * weights[cls * input_channels + c];
            }
            g /= static_cast<float>(N);
            for (int v = 0; v < N; ++v) {
                grad_in[c * N + v] = g;
            }
        }
    }

    // Weight update
    for (int cls = 0; cls < num_outputs; ++cls) {
        for (int c = 0; c < input_channels; ++c) {
            int wi = cls * input_channels + c;
            float g = grad_logits[cls] * channel_avg[c];
            if (use_adam) {
                weight_m[wi] = adam_beta1_ * weight_m[wi] + (1.0f - adam_beta1_) * g;
                weight_m2[wi] = adam_beta2_ * weight_m2[wi] + (1.0f - adam_beta2_) * g * g;
                float mh = weight_m[wi] / bc1;
                float vh = weight_m2[wi] / bc2;
                weights[wi] -= learning_rate * (mh / (std::sqrt(vh) + adam_eps_) + weight_decay * weights[wi]);
            } else {
                g += weight_decay * weights[wi];
                weight_m[wi] = momentum * weight_m[wi] + g;
                weights[wi] -= learning_rate * weight_m[wi];
            }
        }
        float bg = grad_logits[cls];
        if (use_adam) {
            bias_m[cls] = adam_beta1_ * bias_m[cls] + (1.0f - adam_beta1_) * bg;
            bias_m2[cls] = adam_beta2_ * bias_m2[cls] + (1.0f - adam_beta2_) * bg * bg;
            float mh = bias_m[cls] / bc1;
            float vh = bias_m2[cls] / bc2;
            bias[cls] -= learning_rate * mh / (std::sqrt(vh) + adam_eps_);
        } else {
            bias_m[cls] = momentum * bias_m[cls] + bg;
            bias[cls] -= learning_rate * bias_m[cls];
        }
    }
}

void HCNNReadout::compute_gradients(const float* grad_logits, const float* in, int N,
                                    float* grad_in, float* weight_grad, float* bias_grad,
                                    float* work_buf) const {
    // work_buf must be at least input_channels floats if provided.
    std::vector<float> avg_storage;
    float* channel_avg;
    if (work_buf) {
        channel_avg = work_buf;
    } else {
        avg_storage.resize(input_channels);
        channel_avg = avg_storage.data();
    }
    for (int c = 0; c < input_channels; ++c) {
        double sum = 0.0;
        for (int v = 0; v < N; ++v) sum += in[c * N + v];
        channel_avg[c] = static_cast<float>(sum / N);
    }

    if (grad_in) {
        for (int c = 0; c < input_channels; ++c) {
            float g = 0.0f;
            for (int cls = 0; cls < num_outputs; ++cls) {
                g += grad_logits[cls] * weights[cls * input_channels + c];
            }
            g /= static_cast<float>(N);
            for (int v = 0; v < N; ++v) {
                grad_in[c * N + v] = g;
            }
        }
    }

    for (int cls = 0; cls < num_outputs; ++cls) {
        for (int c = 0; c < input_channels; ++c) {
            weight_grad[cls * input_channels + c] = grad_logits[cls] * channel_avg[c];
        }
        if (bias_grad) bias_grad[cls] = grad_logits[cls];
    }
}

void HCNNReadout::apply_gradients(const float* weight_grad, const float* bias_grad,
                                  float learning_rate, float momentum, float weight_decay,
                                  int timestep) {
    const bool use_adam = (optimizer_type_ == OptimizerType::ADAM && timestep > 0);
    const float bc1 = use_adam ? 1.0f - static_cast<float>(std::pow(adam_beta1_, timestep)) : 1.0f;
    const float bc2 = use_adam ? 1.0f - static_cast<float>(std::pow(adam_beta2_, timestep)) : 1.0f;
    int total_w = num_outputs * input_channels;

    if (use_adam) {
        for (int i = 0; i < total_w; ++i) {
            float g = weight_grad[i];
            weight_m[i] = adam_beta1_ * weight_m[i] + (1.0f - adam_beta1_) * g;
            weight_m2[i] = adam_beta2_ * weight_m2[i] + (1.0f - adam_beta2_) * g * g;
            float mh = weight_m[i] / bc1;
            float vh = weight_m2[i] / bc2;
            weights[i] -= learning_rate * (mh / (std::sqrt(vh) + adam_eps_) + weight_decay * weights[i]);
        }
    } else {
        for (int i = 0; i < total_w; ++i) {
            float g = weight_grad[i] + weight_decay * weights[i];
            weight_m[i] = momentum * weight_m[i] + g;
            weights[i] -= learning_rate * weight_m[i];
        }
    }

    if (bias_grad) {
        for (int cls = 0; cls < num_outputs; ++cls) {
            if (use_adam) {
                float g = bias_grad[cls];
                bias_m[cls] = adam_beta1_ * bias_m[cls] + (1.0f - adam_beta1_) * g;
                bias_m2[cls] = adam_beta2_ * bias_m2[cls] + (1.0f - adam_beta2_) * g * g;
                float mh = bias_m[cls] / bc1;
                float vh = bias_m2[cls] / bc2;
                bias[cls] -= learning_rate * mh / (std::sqrt(vh) + adam_eps_);
            } else {
                bias_m[cls] = momentum * bias_m[cls] + bias_grad[cls];
                bias[cls] -= learning_rate * bias_m[cls];
            }
        }
    }
}

} // namespace hcnn
