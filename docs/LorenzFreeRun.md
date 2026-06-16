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

## Harness versions

- **v1 (direct next-state):** readout predicts the next standardized state; the
  x·y·z channel is the raw product. Runs #1–#7 below.
- **v2 (increment + standardized product):** readout predicts the per-step
  increment Δ = s(t+1) − s(t) and free-run reconstructs s(t+1) = s(t) + Δ; the
  x·y·z channel is standardized to unit std so all four inputs share one
  `input_scaling`. Both changes are *symmetric across arms* (they cannot bias the
  A-vs-tanh ordering), but they lift absolute VPT and — as it turned out —
  changed the verdict. Runs #8+ below. NRMSE is increment-normalized in v2, so
  its scale is NOT comparable to v1.

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

**v2 — increment + standardized x·y·z (NRMSE increment-normalized, not v1-comparable):**

| # | activation | sr | is | one-step incr NRMSE | VPT median | VPT mean | note |
|---|---|---|---|---|---|---|---|
| 8  | std::tanh | 0.95 | 0.10 | 0.00274 | 3.24 λt (179) | 3.39 λt | tanh reference |
| 9  | std::tanh | 0.90 | 0.10 | 0.00208 | 3.01 λt (166) | 2.96 λt | (non-monotonic → VPT noise) |
| 10 | std::tanh | 0.88 | 0.10 | 0.00222 | 3.71 λt (205) | 3.59 λt | **tanh's best** |
| 11 | A_lorentz(1.1, 250) | 0.90 | 0.10 | 0.00410 | 3.24 λt (179) | 3.39 λt | |
| 12 | A_lorentz(1.1, 250) | 0.88 | 0.10 | 0.00391 | **3.79 λt** (209) | 3.36 λt | **A's best** |
| 13 | A_lorentz(1.1, 250) | 0.86 | 0.10 | 0.00367 | 3.77 λt (208) | 3.67 λt | |

### Verdict v2 — parity (and the v1 "tanh lead" was partly a harness artifact)

Switching to increment prediction + a standardized product channel lifted **both**
arms by ~1–1.5 λt and **erased the v1 gap**. Best-of-each: **A 3.79 λt (sr=0.88)
vs tanh 3.71 λt (sr=0.88)** — a 0.08 λt difference, far inside the launch/seed
scatter (within-arm VPT swings ~0.5–0.7 λt across sr; tanh is even non-monotonic,
run #9). **On the better-practice harness, A and tanh are at parity on Lorenz
free-run, ~3.0–3.8 λt, with no separation that survives noise.**

Two honest observations:

1. **The v1 conclusion did not survive a better harness.** v1 (direct next-state)
   showed tanh ~20–40% ahead; v2 (increment) shows parity. The earlier "tanh wins
   free-run" was substantially an artifact of the suboptimal direct-state target,
   not a property of the activation. A cautionary data point on harness design —
   exactly what the deep review was for.
2. **A reaches parity VPT *despite worse one-step accuracy*.** In v2 A's one-step
   increment NRMSE is ~1.7× tanh's (0.0037–0.0041 vs 0.0021–0.0027), yet at matched
   sr=0.88 A's VPT is marginally *higher* (3.79 vs 3.71). The fairness rule's
   concern (better one-step → longer VPT) would predict tanh ahead — it isn't. So
   the parity is **not** explained by one-step accuracy; if anything A compounds
   error slightly more gracefully per unit one-step error. This is a faint,
   noise-level signal *toward* A, the opposite of v1 — but it is within noise and
   must not be overclaimed.

Net: the closed-loop test now reads **parity**, consistent with the open-loop
story (parity everywhere; A's distinguishing trait is the gentler operating point).
The strong negative verdict from v1 is **retracted** — it was harness-dependent.

### Caveats (why "provisional")

- **Single reservoir seed, single γ/σ.** This is now the binding limitation: the
  v2 A-vs-tanh gap (0.08 λt best-of-each) is *well inside* the within-arm scatter
  (~0.5–0.7 λt across sr; tanh non-monotonic). "Parity" is the honest call, but
  only multi-seed confidence intervals can distinguish true parity from a small
  real effect in either direction.
- **One-step NRMSE not equalized in v2.** A is ~1.7× worse one-step than tanh and
  cannot reach tanh's one-step floor at these operating points. The VPT parity
  holds *despite* that A disadvantage (noted above), so it is not a one-step
  confound — but a clean matched-NRMSE pair was not achievable here.
- **leak=1, fixed.** Leaky integration matched to dt is an untried lever for *both*
  arms; it could lift absolute VPT but is not expected to change the ordering.

### If pursued further

Multi-seed (≥5 reservoir seeds) VPT confidence intervals on tanh@0.88 vs A@0.88
(v2's best-of-each points) would convert "parity, provisional" to a settled
parity-or-not. Given v1→v2 flipped the verdict, the multi-seed run should use the
v2 harness. A different γ (steeper/narrower central boost) remains the lever by
which A could in principle find a better return map.

## How our absolute VPT compares to the literature

Comparison is matched on the two axes that matter most: the **0.4 normalized-error
threshold** (the field's convention, and ours) and **reservoir size** (literature
baselines use N≈300; we use N=256). VPT in Lyapunov times.

| Regime | VPT (λt) | Notes |
|---|---|---|
| Basic / untuned ESN | ~2–3 | e.g. NRMSE<0.2 until ~2.6 λt |
| **Well-tuned standard RC** | **~10–15** | N≈300; the common literature baseline |
| Heavily optimized RC | >30 | near-zero ridge (~1e−20), edge-of-chaos sr, high-precision solver (dt~1e−3) |
| Special architectures | up to ~99 | autonomous ESN + sparse-obs data assimilation |
| **This work (best)** | **~3.8** | N=256, v2 harness, median over 30 launches |

**Honest placement:** our ~3.8 λt sits in the **basic/untuned-ESN range — roughly
4× below well-tuned standard RC (~10–15 λt)** at matched threshold and comparable
size. This is *absolute*-fidelity standing, and does **not** bear on the A-vs-tanh
result (both arms share the identical harness/tuning, so parity stands).

Why the gap, and why it is expected rather than alarming:
1. **Readout recipe.** The ~15 λt results lean on a *linear* readout solved to
   near-zero regularization (ridge ~1e−20). We use a *nonlinear CNN* readout with
   default weight decay — a different machine, not the recipe RC-Lorenz work is
   tuned around.
2. **Minimal tuning.** Only sr/input_scaling were swept. Regularization, leak
   (still 1.0), and solver/dt are untouched — yet regularization + edge-of-chaos sr
   are exactly the levers the high-VPT work credits.
3. **Coarser dt.** We integrate at dt=0.02; the >30 λt regime uses dt~1e−3 with a
   high-precision solver. Reservoir size (256 vs ~300) is *not* the main gap.

Identified headroom toward the ~15 λt class: regularization, leak, finer dt, and
possibly the readout choice — none of which change the A-vs-tanh ordering.

**Sources:**
- Reservoir computing with large valid prediction time for the Lorenz system —
  arXiv:2508.06730 — <https://arxiv.org/abs/2508.06730> (baseline ~15 λt at N≈300;
  >30 λt with near-zero ridge + edge-of-chaos; 0.4 threshold).
- Chaotic climate system forecasting using an improved ESN with sparse
  observations — *Science China Earth Sciences* —
  <https://link.springer.com/article/10.1007/s11430-024-1593-9> (AESN-SAO ~99 λt).
- A systematic study of Echo State Network topologies for chaotic time series
  prediction — *Neurocomputing* —
  <https://www.sciencedirect.com/science/article/pii/S0925231224018034>.
