// StreamingText — streaming memorization of a character corpus.
//
// The corpus is a ring buffer of length L = text.size().  The reservoir is
// driven through it continuously (pos = (pos + 1) % L); each full traversal is
// a "lap".  At every character we score the model PREQUENTIALLY — predict the
// next char from the current live state, score that prediction, and only THEN
// fold (state, target) into the next online training batch.  Because the
// prediction precedes the weight update, the rolling loss is a fair online
// signal.  Progress is also shown via periodic TEACHER-FORCED readouts:
// predicted-vs-actual characters at the stream's current position (no reset,
// no autoregression, nothing to restore).  See StreamingText.md.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "CharEmbedding.h"
#include "Config.h"
#include "Corpus.h"
#include "ESN.h"  // ESN, CosineLR (via Readout.h)

/*
 * Plain ESNs historically achieve BPC in the 1.5–2.0+ range on datasets like text8/enwik8
 * https://mattmahoney.net/dc/textdata.html
 */
namespace streaming_text {

using config::kDIM;

namespace {

/// Escape control chars so a sample line stays on one row.
std::string EscapeChar(char c)
{
    if (c == '\n') return "\\n";
    if (c == '\r') return "\\r";
    if (c == '\t') return "\\t";
    return std::string(1, c);
}

}  // namespace

int Run()
{
    const config::Cfg& cfg = config::kCfg;

    if (cfg.corpus_path.empty()) {
        std::cerr << "error: config::kCfg.corpus_path is empty\n";
        return 1;
    }

    Corpus corpus;
    if (!LoadCorpus(cfg.corpus_path, corpus)) {
        std::cerr << "error: could not load corpus from " << cfg.corpus_path << "\n";
        return 2;
    }
    const std::size_t L = corpus.text.size();

    const std::size_t prefix = cfg.warmup_chars + cfg.warmup_train_chars + 1;
    if (L < prefix) {
        std::cerr << "error: corpus has " << L << " chars, need at least "
                  << prefix << " for warmup + warmup_train + 1\n";
        return 2;
    }

    std::cerr << "[stext] corpus=" << cfg.corpus_path
              << " chars=" << L
              << " vocab_size=" << corpus.vocab.size() << "\n";

    ESNConfig esn_cfg = cfg.esn;
    const float lr_max = esn_cfg.readout.lr_max;
    const float lr_min = lr_max * cfg.lr_min_frac;

    const std::size_t N = (1ULL << kDIM);
    std::cerr << "[stext] DIM=" << kDIM << " N=" << N
              << " kCharEmbedDim=" << kCharEmbedDim
              << " kInputHistory=" << config::kInputHistory
              << " reservoir_seed=" << esn_cfg.reservoir.seed << "\n";
    std::cerr << "[stext] reservoir: spectral_radius=" << esn_cfg.reservoir.spectral_radius
              << " input_scaling=" << esn_cfg.reservoir.input_scaling
              << " leak_rate=" << esn_cfg.reservoir.leak_rate
              << " num_inputs=" << esn_cfg.reservoir.num_inputs << "\n";

    ESN esn(esn_cfg);

    const std::uint64_t embed_seed = esn_cfg.reservoir.seed ^ config::kCharEmbedSeedXor;
    CharEmbedding char_embed(corpus, embed_seed);
    std::cerr << "[stext] char_embed: vocab=" << char_embed.VocabSize()
              << " seed=" << embed_seed << "\n";

    // Rolling K-char input shift register, continuous across all phases.
    RollingCharWindow window(char_embed, config::kInputHistory);

    // --- Phase 1: reservoir warmup (no training). ---
    std::size_t corpus_pos = 0;
    esn.ResetReservoirOnly();
    window.Clear();
    for (std::size_t i = 0; i < cfg.warmup_chars; ++i) {
        window.Push(corpus.text[corpus_pos++]);
        esn.Warmup(window.Inputs(), 1);
    }

    // --- Phase 2: drive warmup_train_chars through InitOnline (builds the CNN). ---
    const std::size_t channels_per_step = window.InputSize();
    std::vector<float> warmup_embed(cfg.warmup_train_chars * channels_per_step);
    for (std::size_t i = 0; i < cfg.warmup_train_chars; ++i) {
        window.Push(corpus.text[corpus_pos + i]);
        std::memcpy(warmup_embed.data() + i * channels_per_step,
                    window.Inputs(),
                    channels_per_step * sizeof(float));
    }
    esn.InitOnline(warmup_embed.data(), cfg.warmup_train_chars);
    warmup_embed.clear();
    warmup_embed.shrink_to_fit();
    corpus_pos += cfg.warmup_train_chars;  // InitOnline advanced the reservoir to here

    std::cerr << "[stext] CNN cfg: nl=" << esn_cfg.readout.num_layers
              << " ch=" << esn_cfg.readout.conv_channels
              << " lr_max=" << lr_max << " lr_min=" << lr_min
              << " weight_decay=" << esn_cfg.readout.weight_decay
              << " num_outputs=" << esn_cfg.readout.num_outputs << "\n";
    std::cerr << "[stext] stream: warmup=" << cfg.warmup_chars
              << " warmup_train=" << cfg.warmup_train_chars
              << " total_steps=" << cfg.total_steps
              << " (~" << (cfg.total_steps / L) << " laps)"
              << " mini_batch=" << cfg.mini_batch_size << "\n";

    // --- Phase 3: single continuous ring loop. ---
    const std::size_t train_start_pos = corpus_pos % L;
    std::size_t pos = train_start_pos;

    const int K = cfg.mini_batch_size;
    const std::size_t state_dim   = esn.NumOutputVerts();
    const std::size_t num_outputs = esn.NumOutputs();

    std::vector<float> accum_states(static_cast<std::size_t>(K) * state_dim);
    std::vector<int>   accum_targets(K);
    int accum_count = 0;

    std::vector<float> logits(num_outputs);

    const std::size_t total_batches =
        (cfg.total_steps + static_cast<std::size_t>(K) - 1) / static_cast<std::size_t>(K);
    std::size_t batch_index = 0;
    float step_lr = lr_max;

    // Rolling prequential metric: circular buffers over the last report_window
    // chars, with running sums so each step is O(1).
    const std::size_t W = std::max<std::size_t>(1, cfg.report_window);
    std::vector<double> loss_ring(W, 0.0);
    std::vector<unsigned char> hit_ring(W, 0);
    double loss_sum = 0.0;
    std::size_t hit_sum = 0;

    // Teacher-forced sample window state.
    bool in_sample = false;
    std::size_t sample_left = 0;
    std::size_t sample_origin_pos = 0;
    std::size_t sample_origin_lap = 0;
    std::string sample_actual, sample_pred;

    auto t_start = std::chrono::steady_clock::now();

    for (std::size_t step = 0; step < cfg.total_steps; ++step) {
        // 1. Advance reservoir one char (true corpus char).
        window.Push(corpus.text[pos]);
        esn.Warmup(window.Inputs(), 1);

        // 2. Read live state into the accumulation slot and predict next char.
        float* slot = accum_states.data() + static_cast<std::size_t>(accum_count) * state_dim;
        esn.CopyLiveState(slot);
        esn.PredictFromState(slot, logits.data());

        const std::size_t next_pos = (pos + 1) % L;
        const char next_ch = corpus.text[next_pos];
        const int target = CharToClass(corpus, next_ch);

        // 3. Prequential metric: softmax cross-entropy (nats) + top-1 hit.
        float max_logit = logits[0];
        std::size_t argmax = 0;
        for (std::size_t k = 1; k < num_outputs; ++k) {
            if (logits[k] > max_logit) { max_logit = logits[k]; argmax = k; }
        }
        double sum_exp = 0.0;
        for (std::size_t k = 0; k < num_outputs; ++k)
            sum_exp += std::exp(static_cast<double>(logits[k]) - max_logit);
        double loss = 0.0;  // nats
        if (target >= 0 && static_cast<std::size_t>(target) < num_outputs) {
            const double log_prob =
                (static_cast<double>(logits[static_cast<std::size_t>(target)]) - max_logit)
                - std::log(sum_exp);
            loss = -log_prob;
        }
        const unsigned char hit =
            (static_cast<int>(argmax) == target) ? 1u : 0u;

        const std::size_t ring_idx = step % W;
        loss_sum -= loss_ring[ring_idx];
        hit_sum  -= hit_ring[ring_idx];
        loss_ring[ring_idx] = loss;
        hit_ring[ring_idx]  = hit;
        loss_sum += loss;
        hit_sum  += hit;

        // 4. Teacher-forced sample display (reuses the logits above).
        if (cfg.sample_every && !in_sample && (step % cfg.sample_every == 0)) {
            in_sample = true;
            sample_left = cfg.sample_len;
            sample_origin_pos = next_pos;
            sample_origin_lap = (train_start_pos + step) / L;
            sample_actual.clear();
            sample_pred.clear();
        }
        if (in_sample && sample_left > 0) {
            sample_actual += EscapeChar(next_ch);
            sample_pred   += EscapeChar(ClassToChar(corpus, static_cast<int>(argmax)));
            if (--sample_left == 0) {
                in_sample = false;
                std::cerr << "[sample] lap=" << sample_origin_lap
                          << " pos=" << sample_origin_pos << "\n"
                          << "  actual: " << sample_actual << "\n"
                          << "  pred  : " << sample_pred << "\n";
            }
        }

        // 5. Online training: accumulate (state, target), flush every K.
        accum_targets[accum_count] = target;
        ++accum_count;
        if (accum_count == K) {
            const float frac = static_cast<float>(batch_index)
                             / static_cast<float>(total_batches);
            step_lr = CosineLR(frac, lr_max, lr_min);
            esn.TrainLiveBatch(accum_states.data(), accum_targets.data(), K, step_lr);
            ++batch_index;
            accum_count = 0;
        }

        // 6. Advance ring position.
        pos = next_pos;

        // 7. Periodic rolling-metric line (live; no interim tables).
        if (cfg.verbose && cfg.report_every && step
            && (step % cfg.report_every == 0)) {
            const std::size_t n = std::min(step + 1, W);
            const double bpc = (loss_sum / static_cast<double>(n)) / std::log(2.0);
            const double top1 = static_cast<double>(hit_sum) / static_cast<double>(n);
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            std::cerr << "[stext] step=" << step
                      << " lap=" << ((train_start_pos + step) / L)
                      << " roll_bpc=" << bpc
                      << " roll_top1=" << top1
                      << " lr=" << step_lr
                      << " elapsed=" << elapsed << "s\n";
        }
    }

    // Flush any partial batch.
    if (accum_count > 0) {
        const float frac = static_cast<float>(batch_index)
                         / static_cast<float>(total_batches);
        step_lr = CosineLR(frac, lr_max, lr_min);
        esn.TrainLiveBatch(accum_states.data(), accum_targets.data(), accum_count, step_lr);
    }

    const std::size_t n_final = std::min(cfg.total_steps, W);
    const double bpc_final = n_final
        ? (loss_sum / static_cast<double>(n_final)) / std::log(2.0) : 0.0;
    const double top1_final = n_final
        ? static_cast<double>(hit_sum) / static_cast<double>(n_final) : 0.0;
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::cerr << "[stext] done: total_steps=" << cfg.total_steps
              << " laps=" << (cfg.total_steps / L)
              << " final_roll_bpc(" << n_final << ")=" << bpc_final
              << " final_roll_top1=" << top1_final
              << " elapsed=" << elapsed << "s\n";
    return 0;
}

}  // namespace streaming_text

int main()
{
    return streaming_text::Run();
}
