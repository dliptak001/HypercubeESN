# Lorenz-63 Free-Run: A(x) vs tanh — Experiment Tracking

> Status: **active campaign**. This is the closed-loop (generative) test that
> open-loop tasks could not resolve. Open-loop is settled parity (NARMA, sine,
> anomaly, classification; tanh wins linear MC) — A differs only by a gentler
> operating point. Free-run is where the activation's *return map* governs
> compounded error, so a difference here would be the first result NOT reducible
> to an operating-point offset. See `docs/ActivationFunctionA.md` and the
> closed-loop plan in memory.

## The harness — `examples/Lorenz/Lorenz.cpp`

```
 TRAIN (teacher-forced, open loop)        FREE-RUN (closed loop, the real test)
 true s(t) ──[x,y,z,xyz]──▶ Step           resync on true s, then cut the cord:
        │                                    PredictLiveRaw ▶ ŝ(t+1)
        ▼                                          │
   readout: state ▶ s(t+1)                  [x̂,ŷ,ẑ, x̂·ŷ·ẑ] ──▶ StepLive ──┐
   (3 outputs)                                     └── fed back as next input ◀┘
                                            error compounds → VPT
```

- **Lorenz-63**, canonical σ=10, ρ=28, β=8/3; RK4 at `dt=0.02`. Largest Lyapunov
  exponent λ≈0.9056 → **1 Lyapunov time (λt) = 55.2 steps**.
- **I/O:** `num_inputs=4` = (x, y, z, **x·y·z**) — the 4th is a nonlinear
  cross-feature, and 4 divides N=256 cleanly (3 does not). `num_outputs=3`,
  predict the next standardized state. The x·y·z channel is rebuilt from each
  prediction during free-run.
- **Geometry/budgets:** DIM=8 (N=256), discard 5000, warmup 1000, collect 40000
  (train 0.8 / one-step test 0.2), 30 free-run launches spaced 800 apart, 500-step
  teacher-forced resync per launch, horizon 2000.
- **Metrics:** (1) one-step open-loop R²/NRMSE — the *parity baseline* and the
  **precondition** for a fair free-run comparison; (2) **VPT** = steps until the
  Pathak normalized error ‖ŝ−s‖/rms exceeds 0.4, reported in Lyapunov times.
- Activation is the compile-time toggle in `Reservoir.cpp::UpdateState`; one
  build per arm.

## The fairness rule (why one-step NRMSE must match first)

`EstimateSpectralRadius` measures only the recurrent block. A's internal
central-slope gain (~1+γ = 2.1× near the origin) sits **outside** the sr target,
so **A and tanh at the same nominal sr are NOT at the same effective loop gain**.
A VPT comparison only attributes to the activation's *shape* when the two arms
are matched on one-step NRMSE; otherwise a longer VPT is just the better
one-step map free-running. **Match one-step NRMSE, then compare VPT.**

## Results

VPT over 30 launches; λt = Lyapunov times (1 λt = 55.2 steps). Config common
unless noted: DIM=8, N=256, leak=1.0, readout 600 epochs, batch 64, TANH readout.

| # | activation | sr | input_scaling | one-step NRMSE | VPT median | VPT mean | VPT max | note |
|---|---|---|---|---|---|---|---|---|
| 1 | std::tanh | 0.95 | 0.10 | 0.00099 | 3.24 λt (179) | 3.16 λt | 6.90 λt | **reference** |
| 2 | A_lorentz(1.1, 250) | 0.95 | 0.10 | 0.00703 | 2.17 λt (120) | 2.20 λt | 4.75 λt | unmatched — A ~7× worse one-step; overdriven |
| 3 | A_lorentz(1.1, 250) | 0.90 | 0.10 | 0.00142 | 2.37 λt (131) | 2.37 λt | 3.84 λt | gentling sr fixes overdrive (one-step 5× better) |
| 4 | A_lorentz(1.1, 250) | 0.89 | 0.10 | 0.00126 | 2.10 λt (116) | 2.32 λt | — | matched band |
| 5 | A_lorentz(1.1, 250) | 0.88 | 0.10 | 0.00113 | **2.57 λt** (142) | 2.50 λt | — | **A's best VPT**, one-step matched |
| 6 | A_lorentz(1.1, 250) | 0.86 | 0.10 | 0.00106 | 2.32 λt (128) | 2.23 λt | — | one-step ≈ tanh's |
| 7 | A_lorentz(1.1, 250) | 0.90 | 0.07 | 0.00140 | 1.99 λt (110) | 2.08 λt | — | input_scaling lever; no help |

### Verdict — closed-loop does NOT favor A (provisional, 1 seed)

The sweep matched A to tanh on the fairness precondition: runs #5/#6 hit one-step
NRMSE 0.00113 / 0.00106, on top of tanh's 0.00099. **At matched one-step accuracy,
tanh still free-runs meaningfully longer than A:**

- **tanh: 3.24 λt** median VPT.
- **A's best (any matched point): 2.57 λt** (sr=0.88). Across A's entire matched
  operating range (sr 0.86–0.90) VPT sits at **2.0–2.6 λt** — consistently
  **~20–40% short of tanh**.

The hypothesis the whole campaign was built to test — *A free-runs a learned
attractor more stably / longer than tanh because its region-selective return map
is a genuinely different (and supposedly better) map* — is **not supported**. As
the comparison was made fair, the lead stayed with tanh; A never closed it. The
linear→nonlinear capacity reallocation that A demonstrably does (Dambre tradeoff,
MC sweep) does **not** cash out as a free-run advantage on Lorenz-63.

This is consistent with the open-loop story, not a reversal of it: A's only
established distinguishing property remains the **gentler operating point**
(open-loop parity at lower sr / smaller drive). Closed-loop free-run, the most
likely place a qualitative win could have appeared, instead shows tanh ahead.

### Caveats (why "provisional")

- **Single reservoir seed, single γ/σ.** VPT has real scatter (A's matched points
  span 2.0–2.6 λt — internal noise ~0.5 λt on 30 launches). The tanh lead (~0.7 λt
  over A's *best*) exceeds that scatter, so the direction is trusted, but tight
  confidence intervals need multiple reservoir seeds.
- **tanh not swept.** tanh's 3.24 λt is at its natural operating point (one-step
  already 0.00099); a tanh sr-sweep would only raise its ceiling, not lower it —
  it cannot overturn an A-loses result, only widen it.
- **leak=1, fixed.** Leaky integration matched to dt is an untried lever for *both*
  arms; it could lift absolute VPT but is not expected to change the A-vs-tanh
  ordering.

### If pursued further

Multi-seed (≥5 reservoir seeds) VPT confidence intervals on tanh@0.95 vs A@0.88
would convert "provisional" to "settled." A different γ (steeper/narrower central
boost) is the only remaining way A could in principle find a better return map, but
there is no evidence so far that it would.
