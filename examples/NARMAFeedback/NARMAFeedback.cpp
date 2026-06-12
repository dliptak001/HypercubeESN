/// @file NARMAFeedback.cpp
/// @brief §9.4 streaming A/B benchmark: feedback-trained vs. compute-matched
/// control on NARMA-30 (docs/FeedbackTrainingMethodology.md).
///
/// Three measurements per seed, two training runs:
///   LIVE   — feedback_scaling = 0.5: the full Pass-1/Pass-2 scheme.
///   CTRL   — feedback_scaling = 0:   §6.13 config kill-switch. Same RNG
///            draws, same ~4x probe overhead, F provably frozen (every cycle
///            tie-rejects) — the compute-matched control.
///   LESION — LIVE's trained system re-validated with force_zero: how much
///            of LIVE's performance the closed loop carries at evaluation.
///
/// LIVE and CTRL share every seed (paired comparison), and the NARMA series
/// is byte-identical across all arms and seeds (fixed data seeds) — any
/// NRMSE difference is attributable to the feedback signal alone. The §7.4
/// variance gauge then attributes a LIVE win: state-dependent signal
/// (std(F) on the order of epsilon or more) vs. glorified bias (std << eps).
///
/// The existing batch NARMA example (examples/NARMA) is untouched; this
/// harness reuses its NARMA_N_Generator (tanh-wrapped NARMA-30, canonical
/// coefficients) in streaming form.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ESN.h"
#include "../NARMA/NARMA_N.h"

namespace
{
    struct Budgets
    {
        size_t warmup = 500; ///< W: InitOnline warmup = §6.17 validation washout.
        size_t pretrain = 10000; ///< P pre-train examples (mirrored into cfg).
        size_t alternation = 40000; ///< Probe/commit cycles after pretrain.
        size_t val_every = 1000; ///< Validation cadence, in examples.
        size_t val_scored = 1000; ///< Scored validation steps (after the W washout).
        size_t num_seeds = 5;
        const char* label = "full";
    };

    struct ValPoint
    {
        size_t example = 0;
        double nrmse = 0.0;
        ESN::FeedbackTelemetry tel;
    };

    struct ArmResult
    {
        std::vector<ValPoint> curve;
        double final_nrmse = 0.0;
        double best_nrmse = 0.0;
        double lesioned_nrmse = 0.0; ///< force_zero re-validation of the trained system.
        ESN::FeedbackTelemetry final_tel;
    };

    /// One telemetry table row. NaN-sentinel gauges are gated on the integer
    /// fields (window/accepts) — std::isnan is folded to false under this
    /// build's -ffast-math, so the sentinels are invisible to it.
    void PrintRow(size_t example, const char* phase, double nrmse,
                  const ESN::FeedbackTelemetry& t)
    {
        std::printf("  %7zu  %-3s  %8.5f", example, phase, nrmse);
        if (t.window == 0)
        {
            std::printf("  %s\n", "(no alternation cycles yet)");
            return;
        }
        std::printf("  %5.1f%%  %9.3e  %+8.4f  %4.1f%%  %5.3f",
                    100.0 * t.accept_rate, t.var_f, t.mean_f,
                    100.0 * t.saturation_frac, t.mean_lever);
        if (t.accept_rate > 0.0)
            std::printf("  %6.3f  %+6.3f", t.mean_realization, t.sign_balance);
        else
            std::printf("  %6s  %6s", "--", "--");
        std::printf("  %7.4f\n", t.state_rms_max);
    }

    void PrintHeader()
    {
        std::printf("  %7s  %-3s  %8s  %6s  %9s  %8s  %5s  %5s  %6s  %6s  %7s\n",
                    "example", "ph", "valNRMSE", "acc%", "var_f", "mean_f",
                    "sat%", "lever", "realiz", "sgnbal", "rms_max");
    }

    /// Train one arm on the (shared) streams and return its validation curve.
    ArmResult RunArm(const ESNConfig& cfg, const char* label,
                     const std::vector<float>& train_u, const std::vector<float>& train_y,
                     const std::vector<float>& val_u, const std::vector<float>& val_y,
                     const Budgets& B)
    {
        std::printf("\n  --- arm %s (feedback_scaling=%.2f) ---\n",
                    label, cfg.reservoir.feedback_scaling);
        PrintHeader();

        ESN esn(cfg);
        esn.InitOnline(train_u.data(), B.warmup); // closed-loop warmup (§6.15)

        ArmResult r;
        r.best_nrmse = 1e30;
        const size_t total = B.pretrain + B.alternation;
        const size_t val_count = B.warmup + B.val_scored;

        for (size_t i = 0; i < total; ++i)
        {
            const size_t idx = B.warmup + i;
            (void)esn.TrainFeedbackCycle(&train_u[idx], &train_y[idx]);

            if ((i + 1) % B.val_every == 0 || i + 1 == total)
            {
                ValPoint p;
                p.example = i + 1;
                p.nrmse = esn.ValidateClosedLoop(val_u.data(), val_y.data(), val_count);
                p.tel = esn.GetFeedbackTelemetry();
                PrintRow(p.example, i < B.pretrain ? "pre" : "alt", p.nrmse, p.tel);
                std::fflush(stdout);
                r.best_nrmse = std::min(r.best_nrmse, p.nrmse);
                r.curve.push_back(p);
            }
        }

        r.final_nrmse = r.curve.back().nrmse;
        r.final_tel = r.curve.back().tel;

        // LESION: same trained system, loop value forced dead at the seam.
        // For CTRL this must match its own validation (the weights are dead
        // either way) — a free consistency check.
        esn.SetForceZeroFeedback(true);
        r.lesioned_nrmse = esn.ValidateClosedLoop(val_u.data(), val_y.data(), val_count);
        esn.SetForceZeroFeedback(false);

        std::printf("  arm %s: final=%.5f  best=%.5f  lesioned-eval=%.5f\n",
                    label, r.final_nrmse, r.best_nrmse, r.lesioned_nrmse);
        return r;
    }
} // namespace

int main(int argc, char* argv[])
{
    Budgets B; // full preset
    if (argc > 1 && std::strcmp(argv[1], "--quick") == 0)
        B = {500, 2000, 8000, 1000, 500, 2, "quick"};
    else if (argc > 1 && std::strcmp(argv[1], "--smoke") == 0)
        B = {200, 300, 1200, 500, 300, 1, "smoke"};

    constexpr size_t DIM = 8;
    constexpr size_t narma_order = 30;
    constexpr uint64_t data_seed_train = 1939; // matches the batch NARMA example
    constexpr uint64_t data_seed_val = 7331; // held-out: never trained on

    std::printf("=== HypercubeESN: NARMA-%zu streaming feedback A/B (S9.4) ===\n",
                narma_order);
    std::printf("Preset: %s  (W=%zu, pretrain=%zu, alternation=%zu, "
                "val every %zu, %zu scored, %zu seeds)\n",
                B.label, B.warmup, B.pretrain, B.alternation,
                B.val_every, B.val_scored, B.num_seeds);

    // ---- Streams: byte-identical across all arms and seeds -----------------
    // tanh-wrapped NARMA-30, canonical coefficients (the honest-difficulty
    // variant — see NARMA_N.h). The generator discards its own 500-step
    // transient internally.
    const size_t train_len = B.warmup + B.pretrain + B.alternation;
    const size_t val_len = B.warmup + B.val_scored;

    NARMA_N_Generator<float> gen_train(narma_order, data_seed_train,
                                       0.3f, 0.05f, 1.5f, 0.1f, 0.0f, 0.5f,
                                       /*tanh_wrap=*/true);
    auto [train_u, train_y] = gen_train.generate_prediction_task(train_len);
    NARMA_N_Generator<float> gen_val(narma_order, data_seed_val,
                                     0.3f, 0.05f, 1.5f, 0.1f, 0.0f, 0.5f,
                                     /*tanh_wrap=*/true);
    auto [val_u, val_y] = gen_val.generate_prediction_task(val_len);

    // ---- Base config --------------------------------------------------------
    ESNConfig base;
    base.reservoir.dim = DIM;
    base.reservoir.verbose = false;
    base.reservoir.spectral_radius = 0.92f;
    base.reservoir.leak_rate = 1.0f;
    base.reservoir.input_scaling = 0.5f;
    base.reservoir.history_depth = 32; // M-sweep knee sits near M = order
    base.reservoir.num_feedback_channels = 1;
    base.readout.task = ReadoutTask::Regression;
    base.feedback.pretrain_steps = B.pretrain;

    std::printf("Config: DIM=%zu N=%u M=%zu sr=%.2f  feedback: eps=%.3f lr=%.0e "
                "p_lr=%.0e margin=%g  F: %dx%dch seed-base %u\n",
                DIM, 1u << DIM, base.reservoir.history_depth,
                base.reservoir.spectral_radius,
                base.feedback.epsilon, base.feedback.lr, base.feedback.p_lr,
                base.feedback.margin,
                base.feedback.readout.num_layers, base.feedback.readout.conv_channels,
                base.feedback.readout.seed);

    // ---- Paired seed sweep ---------------------------------------------------
    struct SeedResult { uint64_t seed; ArmResult live, ctrl; };
    std::vector<SeedResult> results;

    for (size_t k = 0; k < B.num_seeds; ++k)
    {
        ESNConfig live = base;
        live.reservoir.seed = 73895 + k;
        live.readout.seed = 42 + static_cast<unsigned>(k);
        live.feedback.readout.seed = 43 + static_cast<unsigned>(k);
        live.reservoir.feedback_scaling = 0.5f;

        ESNConfig ctrl = live;
        ctrl.reservoir.feedback_scaling = 0.0f; // §6.13 config arm

        std::printf("\n=== seed %llu (%zu/%zu) ===\n",
                    static_cast<unsigned long long>(live.reservoir.seed),
                    k + 1, B.num_seeds);
        SeedResult sr;
        sr.seed = live.reservoir.seed;
        sr.live = RunArm(live, "LIVE", train_u, train_y, val_u, val_y, B);
        sr.ctrl = RunArm(ctrl, "CTRL", train_u, train_y, val_u, val_y, B);

        const double delta = sr.ctrl.final_nrmse - sr.live.final_nrmse;
        std::printf("  seed %llu paired delta (CTRL - LIVE, >0 = live wins): %+0.5f\n",
                    static_cast<unsigned long long>(sr.seed), delta);
        results.push_back(std::move(sr));
    }

    // ---- Aggregate report (pre-registered verdict rules) ---------------------
    std::printf("\n=== A/B summary: NARMA-%zu, %zu seed(s), final validation NRMSE ===\n",
                narma_order, results.size());
    std::printf("  %10s  %9s  %9s  %9s  %9s  %9s  %9s\n",
                "seed", "LIVE", "CTRL", "delta", "LIVEbest", "CTRLbest", "lesioned");
    size_t wins = 0;
    double delta_sum = 0.0, var_sum = 0.0;
    for (const auto& s : results)
    {
        const double delta = s.ctrl.final_nrmse - s.live.final_nrmse;
        wins += delta > 0.0;
        delta_sum += delta;
        var_sum += s.live.final_tel.var_f;
        std::printf("  %10llu  %9.5f  %9.5f  %+9.5f  %9.5f  %9.5f  %9.5f\n",
                    static_cast<unsigned long long>(s.seed),
                    s.live.final_nrmse, s.ctrl.final_nrmse, delta,
                    s.live.best_nrmse, s.ctrl.best_nrmse, s.live.lesioned_nrmse);
    }
    const double n = static_cast<double>(results.size());
    const double mean_delta = delta_sum / n;
    const double mean_var = var_sum / n;
    const double mean_std_f = std::sqrt(std::max(0.0, mean_var));
    const float eps = base.feedback.epsilon;

    std::printf("\n  mean paired delta: %+0.5f   live wins: %zu/%zu   "
                "mean var_f: %.3e (std %.4f vs eps %.3f)\n",
                mean_delta, wins, results.size(), mean_var, mean_std_f, eps);

    // §7.4 attribution: "material" variation = F's committed output varies by
    // at least a probe step across visited states (heuristic yardstick).
    const bool consistent_win = wins * 5 >= results.size() * 4; // >= 4/5
    const bool material_var = mean_std_f >= static_cast<double>(eps);
    if (consistent_win && mean_delta > 0.0 && material_var)
        std::printf("  VERDICT: state-dependent signal — LIVE wins consistently and "
                    "F varies materially across states (S7.4). Proceed to Lorenz.\n");
    else if (consistent_win && mean_delta > 0.0)
        std::printf("  VERDICT: LIVE wins but std(F) << eps — a glorified bias "
                    "(S7.4), not a state-dependent signal.\n");
    else
        std::printf("  VERDICT: no consistent LIVE win. Read accept rate / "
                    "realization / saturation above before blaming the idea "
                    "(S6.14 defaults are tuning candidates).\n");
    return 0;
}
