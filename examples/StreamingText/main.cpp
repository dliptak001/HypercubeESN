// StreamingText — streaming memorization of a character corpus.
//
// The corpus is a ring buffer of length L = text.size().  The reservoir is
// driven through it continuously (pos = (pos + 1) % L); each full traversal is
// a "lap".  At every character we score the model PREQUENTIALLY — predict the
// next char from the current live readout input, score that prediction, and
// only THEN fold (readout input, target) into the next online training batch.
// Because the prediction precedes the weight update, the rolling loss is a
// fair online signal.  Progress is also shown via periodic TEACHER-FORCED
// readouts: predicted-vs-actual characters at the stream's current position
// (no reset, no autoregression, nothing to restore).  See StreamingText.md.

#include <algorithm>
#include <cassert>
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

namespace {

/// Keep sample lines single-row; only newline is in the fixed vocab.
std::string EscapeChar(char c)
{
    if (c == '\n') return "\\n";
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
    if (cfg.mini_batch_size <= 0) {
        std::cerr << "error: config::kCfg.mini_batch_size must be > 0\n";
        return 1;
    }

    Corpus corpus;
    if (!LoadCorpus(cfg.corpus_path, corpus)) {
        std::cerr << "error: could not load corpus from " << cfg.corpus_path << "\n";
        return 2;
    }
    const std::size_t L = corpus.text.size();

    const std::size_t prefix = cfg.warmup_chars + 1;
    if (L < prefix) {
        std::cerr << "error: corpus has " << L << " chars, need at least "
                  << prefix << " for warmup + 1\n";
        return 2;
    }

    ESNConfig esn_cfg = cfg.esn;
    const float lr_max = esn_cfg.readout.lr_max;
    const float lr_min = lr_max * cfg.lr_min_frac;
    const std::uint64_t embed_seed =
        esn_cfg.reservoir.seed ^ config::kCharEmbedSeedXor;

    std::cerr << "[stext] corpus=" << cfg.corpus_path
              << " chars=" << L
              << " vocab=" << corpus.vocab.size() << "\n";
    std::cerr << "[stext] seeds: reservoir=" << esn_cfg.reservoir.seed
              << " readout=" << esn_cfg.readout.seed
              << " char_embed=" << embed_seed
              << " fsf=" << esn_cfg.reservoir.fsf_seed << "\n";

    ESN esn(esn_cfg);

    std::cerr << "[stext] reservoir: DIM=" << esn.ReservoirHypercubeDimension()
              << " N=" << esn.ReservoirNeuronCount()
              << " embed_dim=" << kCharEmbedDim
              << " sr=" << esn_cfg.reservoir.spectral_radius
              << " in_scale=" << esn_cfg.reservoir.input_scaling
              << " leak=" << esn_cfg.reservoir.leak_rate
              << " history_depth=" << esn_cfg.reservoir.history_depth
              << " num_inputs=" << esn_cfg.reservoir.num_inputs;
    if (esn_cfg.reservoir.full_state_feedback)
        std::cerr << " FSF=ON fsf_scaling=" << esn_cfg.reservoir.fsf_scaling;
    else
        std::cerr << " FSF=OFF";
    std::cerr << "\n";

    // Arch (reservoir weight count + HCNN stack). Reservoir ctor verbose is off.
    std::cerr << esn.ReadoutArchSummary();

    std::cerr << "[stext] stream: warmup=" << cfg.warmup_chars
              << " total_steps=" << cfg.total_steps
              << " (~" << (cfg.total_steps / L) << " laps)"
              << " mini_batch=" << cfg.mini_batch_size
              << " lr_max=" << lr_max << " lr_min=" << lr_min
              << " lr_min_frac=" << cfg.lr_min_frac
              << " weight_decay=" << esn_cfg.readout.weight_decay
              << " readout_width=" << esn.ReadoutInputWidth()
              << " (B=" << esn.ReadoutBlockCount() << ")\n";

    CharEmbedding char_embed(corpus, embed_seed);
    CharInput drive(char_embed);

    // --- Reservoir-only warmup (no readout forward / no train). ---
    std::size_t corpus_pos = 0;
    esn.ReservoirClear();
    if (cfg.warmup_chars > 0) {
        constexpr std::size_t channels = CharInput::Size();
        std::vector<float> warmup_embed(cfg.warmup_chars * channels);
        for (std::size_t i = 0; i < cfg.warmup_chars; ++i) {
            drive.Set(corpus.text[corpus_pos + i]);
            std::memcpy(warmup_embed.data() + i * channels,
                        drive.Data(),
                        channels * sizeof(float));
        }
        esn.ReservoirWarmup(warmup_embed.data(), cfg.warmup_chars);
        corpus_pos += cfg.warmup_chars;
    }

    // --- Ring train loop. ---
    const std::size_t train_start_pos = corpus_pos % L;
    std::size_t pos = train_start_pos;

    const int K = cfg.mini_batch_size;
    // TrainStepBatch / PredictFromState consume assembled readout input (B×N).
    const std::size_t readout_width = esn.ReadoutInputWidth();
    const std::size_t num_outputs   = esn.NumOutputs();
    assert(num_outputs > 0);

    std::vector<float> accum_states(static_cast<std::size_t>(K) * readout_width);
    std::vector<float> accum_targets(K);  // class indices as floats
    std::vector<float> logits(num_outputs);
    int accum_count = 0;

    const std::size_t total_batches =
        (cfg.total_steps + static_cast<std::size_t>(K) - 1) / static_cast<std::size_t>(K);
    std::size_t batch_index = 0;
    float step_lr = lr_max;

    const std::size_t W = std::max<std::size_t>(1, cfg.report_window);
    std::vector<double> loss_ring(W, 0.0);
    std::vector<unsigned char> hit_ring(W, 0);
    double loss_sum = 0.0;
    std::size_t hit_sum = 0;

    bool in_sample = false;
    std::size_t sample_left = 0;
    std::size_t sample_origin_pos = 0;
    std::size_t sample_origin_lap = 0;
    std::string sample_actual, sample_pred;

    auto t_start = std::chrono::steady_clock::now();

    for (std::size_t step = 0; step < cfg.total_steps; ++step) {
        // 1. Advance reservoir one char (true corpus char).
        drive.Set(corpus.text[pos]);
        esn.ReservoirWarmup(drive.Data(), 1);

        // 2. Assemble full readout input; predict next char from that vector.
        float* slot = accum_states.data()
                    + static_cast<std::size_t>(accum_count) * readout_width;
        esn.CopyReadoutInput(slot);
        esn.PredictFromState(slot, logits.data());

        const std::size_t next_pos = (pos + 1) % L;
        const char next_ch = corpus.text[next_pos];
        const int target = CharToClass(corpus, next_ch);
        assert(target >= 0 && static_cast<std::size_t>(target) < num_outputs);

        // 3. Prequential metric: softmax CE (nats) + top-1 hit.
        float max_logit = logits[0];
        std::size_t argmax = 0;
        for (std::size_t k = 1; k < num_outputs; ++k) {
            if (logits[k] > max_logit) { max_logit = logits[k]; argmax = k; }
        }
        double sum_exp = 0.0;
        for (std::size_t k = 0; k < num_outputs; ++k)
            sum_exp += std::exp(static_cast<double>(logits[k]) - max_logit);
        const double log_prob =
            (static_cast<double>(logits[static_cast<std::size_t>(target)]) - max_logit)
            - std::log(sum_exp);
        const double loss = -log_prob;
        const unsigned char hit =
            (static_cast<int>(argmax) == target) ? 1u : 0u;

        const std::size_t ring_idx = step % W;
        loss_sum -= loss_ring[ring_idx];
        hit_sum  -= hit_ring[ring_idx];
        loss_ring[ring_idx] = loss;
        hit_ring[ring_idx]  = hit;
        loss_sum += loss;
        hit_sum  += hit;

        // 4. Teacher-forced sample at end of each lap (every L train steps).
        // When (step+1) is a multiple of L, one full corpus traversal just finished;
        // sample the next sample_len chars from the lap boundary (next_pos).
        if (cfg.sample_len && !in_sample && L > 0
            && ((step + 1) % L == 0)) {
            in_sample = true;
            sample_left = cfg.sample_len;
            sample_origin_pos = next_pos;
            sample_origin_lap = (step + 1) / L;  // 1-based count of laps completed
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

        // 5. Online training: accumulate (readout input, target), flush every K.
        accum_targets[accum_count] = static_cast<float>(target);
        ++accum_count;
        if (accum_count == K) {
            const float frac = static_cast<float>(batch_index)
                             / static_cast<float>(total_batches);
            step_lr = CosineLR(frac, lr_max, lr_min);
            esn.TrainStepBatch(accum_states.data(), accum_targets.data(), K, step_lr,
                               esn_cfg.readout.weight_decay);
            ++batch_index;
            accum_count = 0;
        }

        // 6. Advance ring position.
        pos = next_pos;

        // 7. Periodic rolling-metric line.
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

    if (accum_count > 0) {
        const float frac = static_cast<float>(batch_index)
                         / static_cast<float>(total_batches);
        step_lr = CosineLR(frac, lr_max, lr_min);
        esn.TrainStepBatch(accum_states.data(), accum_targets.data(), accum_count, step_lr,
                           esn_cfg.readout.weight_decay);
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
