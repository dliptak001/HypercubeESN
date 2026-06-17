/// @file Lorenz.cpp
/// @brief Closed-loop (generative) free-run of the Lorenz-63 attractor — the
/// discriminating A(x)-vs-tanh test that open-loop tasks cannot resolve.
///
/// Open-loop tasks (NARMA, sine, MC) re-inject ground truth every step, which
/// continuously corrects internal drift and collapses A-vs-tanh to an
/// operating-point offset (parity). Free-run removes that correction: the
/// reservoir iterates on its OWN output, error compounds through the
/// recurrence, and the activation's return map governs how long the generated
/// trajectory tracks the true one. This is where A's region-selective central
/// gain (a genuinely different map than tanh) could finally cash out — or fail.
///
/// Pipeline:
///   1. Integrate Lorenz-63 (RK4), discard transient, standardize each coord.
///   2. TEACHER-FORCED training (open loop): drive the reservoir with the true
///      state on 4 input channels [x, y, z, x*y*z], train the readout to
///      predict the next state (3 outputs). The 4th channel is a nonlinear
///      cross-feature; 4 divides N = 2^DIM cleanly (3 does not).
///   3. One-step open-loop R2/NRMSE on a held-out tail — the PARITY baseline.
///   4. FREE-RUN (closed loop): resync on true data, then feed each predicted
///      state back as the next input (rebuilding the x*y*z channel from the
///      prediction). Measure Valid Prediction Time (VPT): steps until the
///      normalized error exceeds 0.4, reported in Lyapunov times.
///
/// The activation (A_lorentz vs std::tanh) is the compile-time toggle in
/// Reservoir.cpp::UpdateState. Run this once per build and compare VPT.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include "ESN.h"

namespace
{
    // --- Lorenz-63 canonical parameters ---
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

int main(int argc, char* argv[])
{
    // Optional CLI overrides for sweeps: argv[1]=spectral_radius,
    // argv[2]=input_scaling, argv[3]=reservoir seed, argv[4]=lorentz_gamma,
    // argv[5]=lorentz_inv_sigma2. The activation shape is now runtime too
    // (gamma=0 => tanh, gamma=1.1 => current A, gamma<0 => fold), so a full
    // activation/seed sweep needs no recompile. gamma can be 0 or negative,
    // so it is gated on argc presence, not a positive sentinel.
    const double cli_sr   = (argc > 1) ? std::atof(argv[1]) : -1.0;
    const double cli_is   = (argc > 2) ? std::atof(argv[2]) : -1.0;
    const long   cli_seed = (argc > 3) ? std::atol(argv[3]) : -1;
    const bool   has_gamma = (argc > 4);
    const double cli_gamma = has_gamma ? std::atof(argv[4]) : 0.0;
    const bool   has_isig  = (argc > 5);
    const double cli_isig  = has_isig ? std::atof(argv[5]) : 0.0;

    // ---- geometry / budgets ----
    constexpr size_t DIM     = 8;
    constexpr size_t N       = 1ULL << DIM;   // 256
    constexpr size_t DISCARD = 5000;          // Lorenz transient onto the attractor
    constexpr size_t WARMUP  = 1000;          // reservoir washout (teacher-forced)
    constexpr size_t COLLECT = 40000;         // teacher-forced training states
    constexpr size_t GAP     = 2000;          // separation before the free-run region
    constexpr size_t RESYNC  = 500;           // teacher-forced re-sync before each launch
    constexpr size_t HORIZON = 2000;          // max free-run length scored per launch
    constexpr size_t NUM_LAUNCH   = 30;       // independent free-run launches
    constexpr size_t LAUNCH_STRIDE = 800;     // spacing between launch points
    constexpr double TRAIN_FRACTION = 0.8;    // of COLLECT (rest = one-step test)
    constexpr double VPT_THRESH = 0.4;        // Pathak et al. normalized-error threshold

    const size_t warm_start    = DISCARD;
    const size_t collect_start = warm_start + WARMUP;
    const size_t collect_end   = collect_start + COLLECT;       // target needs s up to here
    const size_t freerun_start = collect_end + GAP;
    const size_t first_launch  = freerun_start + RESYNC;
    const size_t last_launch   = first_launch + (NUM_LAUNCH - 1) * LAUNCH_STRIDE;
    const size_t TOTAL         = last_launch + HORIZON + 2;     // +1 for target lookahead

    std::cout << "=== HypercubeESN: Lorenz-63 Free-Run (closed-loop) ===\n\n";
    std::cout << "Task: teacher-force the readout to one-step-predict the Lorenz\n";
    std::cout << "state, then cut the input and let the network generate its own\n";
    std::cout << "trajectory by feeding predictions back. Closed-loop free-run is\n";
    std::cout << "the test that separates the activation's return map from tanh's.\n\n";

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
    // input channels share one meaningful input_scaling. The raw triple product
    // is wider and heavier-tailed than the linear channels, so without this it
    // carries a different effective input gain.
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
    // identically in both regimes.
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

    // ---- 3. configure the ESN ----
    ESNConfig cfg;
    cfg.reservoir.dim            = DIM;
    // depth 16 beats 32: ~0.7 lt more VPT for both arms at seed 73895, and it
    // matches the doc-era default (#8-13). 32 was an unintended drift.
    cfg.reservoir.history_depth = 16;
    cfg.reservoir.num_inputs     = 4;     // x, y, z, x*y*z  (4 | N; 3 does not divide 256)
    cfg.reservoir.spectral_radius = 0.90; // A(x): 0.90,  tanh(x): 0.95   (tune per activation)
    cfg.reservoir.input_scaling  = 0.1;   // A(x): 0.1,   tanh(x): 0.1    (tune per activation)
    // LOCKED general-purpose seeds (M=16, serve both tanh and A; ranked by
    // worst-of-two-arm VPT): 23, 42, 73895. Use these for A-vs-tanh comparisons.
    //   seed 23    : tanh 4.20 / A 3.95 lt  (strong on both)
    //   seed 42    : tanh 3.64 / A 4.51 lt  (A champion, tanh solid)
    //   seed 73895 : tanh 3.71 / A 3.55 lt  (balanced; reproducibility anchor / default)
    if (cli_sr > 0)    cfg.reservoir.spectral_radius = static_cast<float>(cli_sr);
    if (cli_is > 0)    cfg.reservoir.input_scaling   = static_cast<float>(cli_is);
    if (cli_seed >= 0) cfg.reservoir.seed            = static_cast<uint64_t>(cli_seed);
    if (has_gamma)     cfg.reservoir.lorentz_gamma      = static_cast<float>(cli_gamma);
    if (has_isig)      cfg.reservoir.lorentz_inv_sigma2 = static_cast<float>(cli_isig);
    cfg.reservoir.leak_rate      = 1.0;   // continuous flow; <1.0 (leaky) is worth a sweep
    cfg.readout.task             = ReadoutTask::Regression;
    cfg.readout.num_outputs      = 3;     // predict per-step increment (dx, dy, dz)
    cfg.readout.epochs           = 600;
    cfg.readout.batch_size       = 64;
    cfg.readout.activation       = ReadoutActivation::TANH;
    ESN esn(cfg);

    std::cout << "  Config: DIM=" << DIM << "  N=" << N
              << "  inputs=4 [x,y,z,xyz]  outputs=3\n";
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
              << "\n\n";

    // ---- 4. teacher-forced training (open loop) ----
    std::vector<float> inputs_wc((WARMUP + COLLECT) * 4);
    for (size_t i = 0; i < WARMUP + COLLECT; ++i)
        make_input(warm_start + i, &inputs_wc[i * 4]);

    // Target is the per-step INCREMENT delta = s(t+1) - s(t), not the next state
    // directly: for an ODE flow the readout learns the small change with less
    // systematic bias, and free-run reconstructs s(t+1) = s(t) + delta.
    std::vector<float> targets(COLLECT * 3);
    for (size_t i = 0; i < COLLECT; ++i)
    {
        const size_t t0 = collect_start + i;      // just-input state
        const size_t t1 = t0 + 1;                 // next state
        targets[i * 3 + 0] = nx[t1] - nx[t0];
        targets[i * 3 + 1] = ny[t1] - ny[t0];
        targets[i * 3 + 2] = nz[t1] - nz[t0];
    }

    const size_t train_size = static_cast<size_t>(COLLECT * TRAIN_FRACTION);
    const size_t test_size  = COLLECT - train_size;

    esn.Warmup(inputs_wc.data(), WARMUP);
    esn.Run(inputs_wc.data() + WARMUP * 4, COLLECT);

    std::cout << "  Training readout (" << cfg.readout.epochs << " epochs)..." << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    esn.Train(targets.data(), train_size);
    auto t1 = std::chrono::steady_clock::now();
    std::cout << " done (" << std::fixed << std::setprecision(1)
              << std::chrono::duration<double>(t1 - t0).count() << "s)\n\n";

    // ---- 5. one-step open-loop parity baseline ----
    const double r2    = esn.R2(targets.data(), train_size, test_size);
    const double nrmse = esn.NRMSE(targets.data(), train_size, test_size);
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
    float pred[3];
    float in4[4];

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
    std::cout << "Compare A(x) vs std::tanh by rebuilding with the other activation\n";
    std::cout << "(Reservoir.cpp::UpdateState) at MATCHED one-step NRMSE.\n\n";

    // Machine-readable one-liner for sweeps.
    std::cout << "RESULT sr=" << std::setprecision(4) << cfg.reservoir.spectral_radius
              << " is=" << cfg.reservoir.input_scaling
              << " one_step_nrmse=" << std::setprecision(6) << nrmse
              << " vpt_med_steps=" << std::setprecision(4) << median
              << " vpt_med_lt=" << to_lt(median)
              << " vpt_mean_lt=" << to_lt(mean) << "\n";

    return 0;
}
