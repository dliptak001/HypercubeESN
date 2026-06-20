/// @file LorenzOnline.cpp
/// @brief Closed-loop (generative) free-run of Lorenz-63, with the readout
/// trained ONLINE (streaming, one pass) instead of in batch.
///
/// This is the online-trained sibling of Lorenz.cpp (which batch-trains the
/// readout over many epochs). It exists for two reasons:
///   1. It is the SINGLE-MEMBER analog of the forthcoming ensemble free-run
///      harness — EnsembleESN is online-only, so the honest baseline for "does
///      consensus coupling help?" must itself be online, not batch.
///   2. It measures the online-vs-batch training gap directly: identical
///      reservoir (same seed/config) and identical free-run scoring, so any VPT
///      difference vs Lorenz.cpp isolates training quality alone (one streaming
///      pass with a fixed lr vs many batch epochs).
///
/// Only the TRAINING stage differs from Lorenz.cpp. The reservoir trajectory
/// through warmup -> train -> test is identical (online training updates only
/// the readout, never the reservoir), so the held-out one-step parity number is
/// a clean apples-to-apples comparison.
///
/// Pipeline:
///   1. Integrate Lorenz-63 (RK4), discard transient, standardize each coord.
///   2. TEACHER-FORCED ONLINE training (open loop): drive the reservoir with the
///      true state on 4 channels [x, y, z, x*y*z]; at each step train the readout
///      on the live state toward the next-step increment (3 outputs), one pass.
///   3. One-step open-loop R2/NRMSE on a held-out tail — the PARITY baseline.
///   4. FREE-RUN (closed loop): resync on true data, then feed each predicted
///      state back as the next input. Measure Valid Prediction Time (VPT): steps
///      until the normalized error exceeds VPT_THRESH, in Lyapunov times.
///
/// The activation (A_lorentz vs tanh) is the LORENTZ_GAMMA knob in the config
/// block below (0 => tanh, 1.1 => A) — runtime, no recompile of Reservoir.cpp.
///
/// This example takes NO command-line arguments. To change the run, edit the
/// CONFIGURATION block below and rebuild.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>
#include "ESN.h"

// ============================================================================
//  CONFIGURATION  —  edit these, then rebuild. Everything else is mechanism.
// ============================================================================
namespace config
{
    // ---- (A) Reservoir / model ----
    constexpr size_t   DIM             = 8;            // hypercube dimension
    constexpr size_t   N               = 1ULL << DIM;  // neuron count = 2^DIM = 256
    constexpr uint64_t SEED            = 673895;       // reservoir realization (match Lorenz.cpp
                                                       // to compare online vs batch at one seed).
    constexpr float    SPECTRAL_RADIUS = 0.90f;        // A(x): ~0.90,  tanh(x): ~0.95 (tune per arm)
    constexpr float    INPUT_SCALING   = 0.10f;        // shared across all input channels
    constexpr float    LEAK_RATE       = 1.0f;         // 1.0 = continuous flow; <1.0 (leaky) worth a sweep
    constexpr size_t   HISTORY_DEPTH   = 16;           // delay-line depth (16 beats 32 here: ~0.7 LT more VPT)

    // Activation shape — THE discriminating knob (Lorentzian envelope A(x)):
    //   LORENTZ_GAMMA = 0     => plain tanh (the baseline arm)
    //   LORENTZ_GAMMA = 1.1   => the "A" region-selective central-gain map
    //   LORENTZ_GAMMA < 0     => non-monotone "fold"
    constexpr float    LORENTZ_GAMMA      = 1.1f;      // set 0.0f for the tanh baseline arm
    constexpr float    LORENTZ_INV_SIGMA2 = 250.0f;    // 1/sigma^2 of the envelope

    // ---- (B) Readout (HCNN), trained ONLINE (single-sample, multi-epoch) ----
    constexpr float    ONLINE_LR           = 0.0015f;  // per-step online learning rate (Adam)
    constexpr float    ONLINE_WEIGHT_DECAY = 0.0f;     // L2 on readout weights
    constexpr size_t   ONLINE_EPOCHS       = 600;      // single-sample passes over the train window.
                                                       // ONLINE_EPOCHS * train_size = total sample-gradients;
                                                       // 600 * 32000 = 19.2M, matching the batch budget.

    // ---- (C) Free-run experiment (all in integration steps; dt is fixed below) ----
    constexpr size_t   DISCARD        = 5000;   // Lorenz transient onto the attractor
    constexpr size_t   WARMUP         = 1000;   // reservoir washout (teacher-forced)
    constexpr size_t   COLLECT        = 40000;  // teacher-forced states (train stream + test tail)
    constexpr size_t   GAP            = 2000;   // separation before the free-run region
    constexpr size_t   RESYNC         = 500;    // teacher-forced re-sync before each launch
    constexpr size_t   HORIZON        = 2000;   // max free-run length scored per launch
    constexpr size_t   NUM_LAUNCH     = 30;     // independent free-run launches
    constexpr size_t   LAUNCH_STRIDE  = 800;    // spacing between launch points
    constexpr double   TRAIN_FRACTION = 0.8;    // of COLLECT (the rest is the held-out one-step test)
    constexpr double   VPT_THRESH     = 0.4;    // Pathak et al. normalized-error threshold

    // ---- structural (tied to the task's input/target encoding; changing one of
    //      these means editing the channel construction in main, not just the
    //      number) ----
    constexpr size_t   NUM_INPUTS  = 4;   // [x, y, z, x*y*z]  (4 | N; 3 does not divide 256)
    constexpr int      NUM_OUTPUTS = 3;   // per-step increment (dx, dy, dz)
}

namespace
{
    // --- Lorenz-63 canonical parameters (the dynamical system; not tuning) ---
    constexpr double kSigma = 10.0;
    constexpr double kRho   = 28.0;
    constexpr double kBeta  = 8.0 / 3.0;
    constexpr double kDt    = 0.02;   // integration step
    // Largest Lyapunov exponent of Lorenz-63 (Wolf et al. 1985 ~0.906/time-unit).
    // One Lyapunov time = 1/lambda; with kDt one LT spans ~1/(0.906*0.02) ~= 55 steps.
    constexpr double kLambdaMax = 0.9056;

    struct Vec3 { double x, y, z; };

    inline Vec3 Deriv(const Vec3& s)
    {
        return { kSigma * (s.y - s.x),
                 s.x * (kRho - s.z) - s.y,
                 s.x * s.y - kBeta * s.z };
    }

    inline Vec3 Rk4Step(const Vec3& s, double dt)
    {
        const Vec3 k1 = Deriv(s);
        const Vec3 s2 = { s.x + 0.5 * dt * k1.x, s.y + 0.5 * dt * k1.y, s.z + 0.5 * dt * k1.z };
        const Vec3 k2 = Deriv(s2);
        const Vec3 s3 = { s.x + 0.5 * dt * k2.x, s.y + 0.5 * dt * k2.y, s.z + 0.5 * dt * k2.z };
        const Vec3 k3 = Deriv(s3);
        const Vec3 s4 = { s.x + dt * k3.x, s.y + dt * k3.y, s.z + dt * k3.z };
        const Vec3 k4 = Deriv(s4);
        return { s.x + dt / 6.0 * (k1.x + 2 * k2.x + 2 * k3.x + k4.x),
                 s.y + dt / 6.0 * (k1.y + 2 * k2.y + 2 * k3.y + k4.y),
                 s.z + dt / 6.0 * (k1.z + 2 * k2.z + 2 * k3.z + k4.z) };
    }
}

int main()
{
    using namespace config;

    // ---- derived budgets (windows into the integrated trajectory) ----
    const size_t warm_start    = DISCARD;
    const size_t collect_start = warm_start + WARMUP;
    const size_t collect_end   = collect_start + COLLECT;       // target needs s up to here
    const size_t freerun_start = collect_end + GAP;
    const size_t first_launch  = freerun_start + RESYNC;
    const size_t last_launch   = first_launch + (NUM_LAUNCH - 1) * LAUNCH_STRIDE;
    const size_t TOTAL         = last_launch + HORIZON + 2;     // +1 for target lookahead

    const size_t train_size = static_cast<size_t>(COLLECT * TRAIN_FRACTION);
    const size_t test_size  = COLLECT - train_size;
    const size_t train_end  = collect_start + train_size;       // first held-out test step

    std::cout << "=== HypercubeESN: Lorenz-63 Free-Run (closed-loop, ONLINE-trained) ===\n\n";
    std::cout << "Task: teacher-force the readout (ONLINE, one streaming pass) to\n";
    std::cout << "one-step-predict the Lorenz state, then cut the input and let the\n";
    std::cout << "network generate its own trajectory by feeding predictions back.\n\n";

    // ---- 1. integrate Lorenz-63 ----
    std::vector<Vec3> s(TOTAL);
    Vec3 cur = { 1.0, 1.0, 1.0 };
    for (size_t t = 0; t < TOTAL; ++t) { s[t] = cur; cur = Rk4Step(cur, kDt); }

    // ---- 2. standardize each coordinate (stats over the training window) ----
    double mx = 0, my = 0, mz = 0;
    for (size_t t = collect_start; t < collect_end; ++t) { mx += s[t].x; my += s[t].y; mz += s[t].z; }
    mx /= COLLECT; my /= COLLECT; mz /= COLLECT;
    double vx = 0, vy = 0, vz = 0;
    for (size_t t = collect_start; t < collect_end; ++t)
    {
        vx += (s[t].x - mx) * (s[t].x - mx);
        vy += (s[t].y - my) * (s[t].y - my);
        vz += (s[t].z - mz) * (s[t].z - mz);
    }
    const double sx = std::sqrt(vx / COLLECT), sy = std::sqrt(vy / COLLECT), sz = std::sqrt(vz / COLLECT);

    std::vector<float> nx(TOTAL), ny(TOTAL), nz(TOTAL);
    for (size_t t = 0; t < TOTAL; ++t)
    {
        nx[t] = static_cast<float>((s[t].x - mx) / sx);
        ny[t] = static_cast<float>((s[t].y - my) / sy);
        nz[t] = static_cast<float>((s[t].z - mz) / sz);
    }

    // Standardize the nonlinear cross-feature channel (x*y*z in normalized
    // space) to zero-mean / unit-std over the training window, so all four
    // input channels share one meaningful input_scaling.
    double mp = 0.0;
    for (size_t t = collect_start; t < collect_end; ++t)
        mp += double(nx[t]) * ny[t] * nz[t];
    mp /= COLLECT;
    double vp = 0.0;
    for (size_t t = collect_start; t < collect_end; ++t)
    {
        const double p = double(nx[t]) * ny[t] * nz[t];
        vp += (p - mp) * (p - mp);
    }
    const double sp = std::sqrt(vp / COLLECT);
    const double inv_sp = (sp > 1e-12) ? 1.0 / sp : 1.0;

    // 4-channel input from a (normalized) state: [x, y, z, standardized(x*y*z)].
    // Shared by teacher forcing and free-run so the product channel is built
    // identically in both regimes. (This is where NUM_INPUTS = 4 is realized.)
    auto fill_input = [&](float xn, float yn, float zn, float* out)
    {
        out[0] = xn;
        out[1] = yn;
        out[2] = zn;
        out[3] = static_cast<float>((double(xn) * yn * zn - mp) * inv_sp);
    };
    auto make_input = [&](size_t idx, float* out)
    {
        fill_input(nx[idx], ny[idx], nz[idx], out);
    };
    // Per-step increment target delta = s(t+1) - s(t) (NUM_OUTPUTS components).
    auto fill_target = [&](size_t idx, float* out)
    {
        out[0] = nx[idx + 1] - nx[idx];
        out[1] = ny[idx + 1] - ny[idx];
        out[2] = nz[idx + 1] - nz[idx];
    };

    // ---- 3. configure the ESN (all knobs from the config block above) ----
    ESNConfig cfg;
    cfg.reservoir.dim                = DIM;
    cfg.reservoir.history_depth      = HISTORY_DEPTH;
    cfg.reservoir.num_inputs         = NUM_INPUTS;
    cfg.reservoir.spectral_radius    = SPECTRAL_RADIUS;
    cfg.reservoir.input_scaling      = INPUT_SCALING;
    cfg.reservoir.leak_rate          = LEAK_RATE;
    cfg.reservoir.seed               = SEED;
    cfg.reservoir.lorentz_gamma      = LORENTZ_GAMMA;
    cfg.reservoir.lorentz_inv_sigma2 = LORENTZ_INV_SIGMA2;
    cfg.readout.task                 = ReadoutTask::Regression;
    cfg.readout.num_outputs          = NUM_OUTPUTS;   // predict per-step increment (dx, dy, dz)
    cfg.readout.activation           = ReadoutActivation::TANH;  // CNN readout activation (distinct from the reservoir A/tanh arm)
    // Note: readout.epochs / batch_size are unused by the online path; training
    // is driven by ONLINE_LR via TrainLiveStepRegression below.
    ESN esn(cfg);

    std::cout << "  Config: DIM=" << DIM << "  N=" << N
              << "  inputs=" << NUM_INPUTS << " [x,y,z,xyz]  outputs=" << NUM_OUTPUTS << "\n";
    std::cout << "  spectral_radius=" << cfg.reservoir.spectral_radius
              << " (realized ~ measured at build)"
              << "  input_scaling=" << cfg.reservoir.input_scaling
              << "  leak=" << cfg.reservoir.leak_rate << "\n";
    std::cout << "  dt=" << kDt << "  Lyapunov time ~ "
              << (1.0 / (kLambdaMax * kDt)) << " steps"
              << "  seed=" << cfg.reservoir.seed << "\n";
    std::cout << "  activation: A_lorentz  gamma=" << cfg.reservoir.lorentz_gamma
              << "  inv_sigma2=" << cfg.reservoir.lorentz_inv_sigma2
              << (cfg.reservoir.lorentz_gamma == 0.0f ? "  (== tanh)" : "")
              << "\n";
    std::cout << "  training: ONLINE single-sample  " << ONLINE_EPOCHS << " epochs  lr=" << ONLINE_LR
              << "  weight_decay=" << ONLINE_WEIGHT_DECAY << "\n\n";

    // ---- 4. teacher-forced ONLINE training (open loop, single-sample) ----
    // Match the BATCH sample-gradient budget with pure single-sample updates:
    // ONLINE_EPOCHS passes over the train window = ONLINE_EPOCHS * train_size
    // single-sample gradient steps (batch reaches 19.2M via 600 minibatch
    // epochs). Each epoch resets and re-warms the reservoir, so every pass sees
    // the identical (state, target) pairs the batch readout trains on; only the
    // readout (CNN) accumulates across passes.
    std::vector<float> warmup_inputs(WARMUP * NUM_INPUTS);
    for (size_t i = 0; i < WARMUP; ++i)
        make_input(warm_start + i, &warmup_inputs[i * NUM_INPUTS]);

    const size_t total_steps = ONLINE_EPOCHS * train_size;
    std::cout << "  Online training (" << ONLINE_EPOCHS << " epochs x " << train_size
              << " = " << total_steps << " sample-gradients, lr=" << ONLINE_LR << ")..." << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    float in4[NUM_INPUTS];
    float tgt[NUM_OUTPUTS];
    for (size_t epoch = 0; epoch < ONLINE_EPOCHS; ++epoch)
    {
        esn.ResetReservoirOnly();                           // identical reservoir trajectory each pass
        esn.Warmup(warmup_inputs.data(), WARMUP);
        for (size_t t = collect_start; t < train_end; ++t)
        {
            make_input(t, in4);
            esn.StepLive(in4);                              // live state now reflects input s[t]
            fill_target(t, tgt);                            // increment s[t+1] - s[t]
            esn.TrainLiveStepRegression(tgt, ONLINE_LR, ONLINE_WEIGHT_DECAY);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cout << " done (" << std::fixed << std::setprecision(1)
              << std::chrono::duration<double>(t1 - t0).count() << "s)\n\n";

    // ---- 5. one-step open-loop parity baseline (held-out test tail) ----
    // Continue the reservoir over the test window (no training), collecting
    // states, then reuse the batch R2/NRMSE machinery for an apples-to-apples
    // one-step parity number against Lorenz.cpp.
    std::vector<float> test_inputs(test_size * NUM_INPUTS);
    std::vector<float> test_targets(test_size * NUM_OUTPUTS);
    for (size_t i = 0; i < test_size; ++i)
    {
        make_input(train_end + i, &test_inputs[i * NUM_INPUTS]);
        fill_target(train_end + i, &test_targets[i * NUM_OUTPUTS]);
    }
    esn.Run(test_inputs.data(), test_size);                 // continues reservoir, collects states
    const double r2    = esn.R2(test_targets.data(), 0, test_size);
    const double nrmse = esn.NRMSE(test_targets.data(), 0, test_size);
    esn.ClearStates();                                      // done with collected states

    std::cout << "  -- One-step open-loop increment (teacher-forced, parity baseline) --\n";
    std::cout << "     R2:    " << std::setprecision(6) << r2 << "\n";
    std::cout << "     NRMSE: " << std::setprecision(6) << nrmse
              << "   (increment-normalized; match A to tanh here before comparing VPT)\n\n";

    // ---- 6. free-run / VPT ----
    // Error denominator: RMS of the full 3-vector norm over the free-run region
    // (Pathak normalization). With unit-std coords this is ~sqrt(3).
    double denom_acc = 0.0; size_t denom_n = 0;
    for (size_t t = freerun_start; t < TOTAL; ++t)
    {
        denom_acc += double(nx[t]) * nx[t] + double(ny[t]) * ny[t] + double(nz[t]) * nz[t];
        ++denom_n;
    }
    const double err_denom = std::sqrt(denom_acc / denom_n);

    std::vector<double> vpt_steps(NUM_LAUNCH, 0);
    float pred[NUM_OUTPUTS];

    for (size_t k = 0; k < NUM_LAUNCH; ++k)
    {
        const size_t L = first_launch + k * LAUNCH_STRIDE;

        // re-sync the reservoir onto the true trajectory ending just before L
        esn.ResetReservoirOnly();
        for (size_t t = L - RESYNC; t < L; ++t) { make_input(t, in4); esn.StepLive(in4); }

        // cut the cord: predict the increment, reconstruct the next state by
        // adding it to the current estimate, feed that back. Current estimate
        // starts at the last true input fed during resync (s[L-1]).
        double cx = nx[L - 1], cy = ny[L - 1], cz = nz[L - 1];
        size_t reached = HORIZON;
        for (size_t h = 0; h < HORIZON; ++h)
        {
            esn.PredictLiveRaw(pred);             // pred = predicted increment delta
            const double px = cx + pred[0];       // reconstructed normalized s[L+h]
            const double py = cy + pred[1];
            const double pz = cz + pred[2];
            const size_t tt = L + h;
            const double ex = px - nx[tt];
            const double ey = py - ny[tt];
            const double ez = pz - nz[tt];
            const double e  = std::sqrt(ex * ex + ey * ey + ez * ez) / err_denom;
            if (e > VPT_THRESH) { reached = h; break; }

            fill_input(static_cast<float>(px), static_cast<float>(py), static_cast<float>(pz), in4);
            esn.StepLive(in4);
            cx = px; cy = py; cz = pz;
        }
        vpt_steps[k] = static_cast<double>(reached);

        // detailed trace for the first launch
        if (k == 0)
        {
            std::cout << "  -- Free-run sample (launch 0), denormalized Lorenz units --\n";
            std::cout << "    step |    x_true   x_pred  |    z_true   z_pred  |  norm.err\n";
            std::cout << "    -----+---------------------+---------------------+----------\n";
            esn.ResetReservoirOnly();
            for (size_t t = L - RESYNC; t < L; ++t) { make_input(t, in4); esn.StepLive(in4); }
            double dx = nx[L - 1], dy = ny[L - 1], dz = nz[L - 1];
            for (size_t h = 0; h < 20; ++h)
            {
                esn.PredictLiveRaw(pred);
                const double px = dx + pred[0], py = dy + pred[1], pz = dz + pred[2];
                const size_t tt = L + h;
                const double ex = px - nx[tt], ey = py - ny[tt], ez = pz - nz[tt];
                const double e  = std::sqrt(ex * ex + ey * ey + ez * ez) / err_denom;
                const double xt = nx[tt] * sx + mx, xp = px * sx + mx;
                const double zt = nz[tt] * sz + mz, zp = pz * sz + mz;
                std::cout << "    " << std::setw(4) << h
                          << " | " << std::setw(8) << std::setprecision(3) << xt
                          << " " << std::setw(8) << xp
                          << "  | " << std::setw(8) << zt
                          << " " << std::setw(8) << zp
                          << "  | " << std::setw(8) << std::setprecision(4) << e << "\n";
                fill_input(static_cast<float>(px), static_cast<float>(py), static_cast<float>(pz), in4);
                esn.StepLive(in4);
                dx = px; dy = py; dz = pz;
            }
            std::cout << "\n";
        }
    }

    // ---- 7. VPT statistics ----
    std::vector<double> sorted = vpt_steps;
    std::sort(sorted.begin(), sorted.end());
    double mean = 0; for (double v : vpt_steps) mean += v; mean /= NUM_LAUNCH;
    const double median = sorted[NUM_LAUNCH / 2];
    const double vmin = sorted.front(), vmax = sorted.back();
    auto to_lt = [&](double steps) { return steps * kDt * kLambdaMax; };

    std::cout << "  -- Valid Prediction Time over " << NUM_LAUNCH
              << " launches (threshold " << VPT_THRESH << ") --\n";
    std::cout << std::setprecision(2);
    std::cout << "     mean   : " << std::setw(7) << mean   << " steps  = "
              << std::setw(5) << to_lt(mean)   << " Lyapunov times\n";
    std::cout << "     median : " << std::setw(7) << median << " steps  = "
              << std::setw(5) << to_lt(median) << " Lyapunov times\n";
    std::cout << "     min    : " << std::setw(7) << vmin   << " steps  = "
              << std::setw(5) << to_lt(vmin)   << " Lyapunov times\n";
    std::cout << "     max    : " << std::setw(7) << vmax   << " steps  = "
              << std::setw(5) << to_lt(vmax)   << " Lyapunov times";
    if (vmax >= HORIZON) std::cout << "  (capped at HORIZON)";
    std::cout << "\n\n";

    std::cout << "Median VPT in Lyapunov times is the headline closed-loop number.\n";
    std::cout << "This is the ONLINE-trained baseline; compare with Lorenz.cpp (batch)\n";
    std::cout << "at the same seed to read the online-vs-batch training gap.\n\n";

    // Compact one-line summary (one-step NRMSE + the headline VPT numbers).
    std::cout << "RESULT online lr=" << std::setprecision(4) << ONLINE_LR
              << " sr=" << cfg.reservoir.spectral_radius
              << " is=" << cfg.reservoir.input_scaling
              << " one_step_nrmse=" << std::setprecision(6) << nrmse
              << " vpt_med_steps=" << std::setprecision(4) << median
              << " vpt_med_lt=" << to_lt(median)
              << " vpt_mean_lt=" << to_lt(mean) << "\n";

    return 0;
}
