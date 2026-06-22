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
        │                                    Predict ▶ ŝ(t+1)
        ▼                                          │
   readout: state ▶ s(t+1)                  [x̂,ŷ,ẑ, x̂·ŷ·ẑ] ──▶ Step     ──┐
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

**v3 — multi-seed at sr=0.88, runtime activation (2026-06-16):**

Harness change: activation shape is now a **runtime** config field
(`ReservoirConfig::lorentz_gamma` / `lorentz_inv_sigma2`), consumed in
`UpdateState` via `A_lorentz(s, γ, 1/σ²)`. γ=0 ⇒ algebraically `std::tanh`,
γ=1.1 ⇒ the A arm, γ<0 ⇒ non-monotone "fold". CLI:
`Lorenz.exe [sr] [is] [seed] [gamma] [inv_sigma2]`. Also: `noise_rng_` is now
explicitly seeded in `Initialize()`. **Both changes shifted the absolute baseline
— see the reproducibility flag below; do NOT cross-compare these absolutes to
runs #1–#13.** All 14 runs share one build; arms differ only by γ.

6 reservoir seeds, paired tanh (γ=0) vs A (γ=1.1), median VPT:

| seed | tanh steps / λt | A steps / λt | Δ λt | tanh NRMSE | A NRMSE |
|---|---|---|---|---|---|
| 73895 | 167 / 3.02 | 166 / 3.01 | −0.02 | 0.006292 | 0.005220 |
| 11 | 140 / 2.54 | 188 / 3.41 | +0.87 | 0.007558 | 0.005779 |
| 23 | 148 / 2.68 | 181 / 3.28 | +0.60 | 0.006238 | 0.005597 |
| 42 | 177 / 3.21 | 177 / 3.21 | 0.00 | 0.007049 | 0.005440 |
| 101 | 159 / 2.88 | 180 / 3.26 | +0.38 | 0.005981 | 0.004794 |
| 202 | 136 / 2.46 | 148 / 2.68 | +0.22 | 0.006137 | 0.005174 |
| **mean ± sd** | **2.80 ± 0.29** | **3.14 ± 0.26** | **+0.34** | 0.00655 | 0.00534 |

Paired diff +0.34 λt, SD(diff)=0.35, paired t≈2.40 (df=5, two-tailed p≈0.06);
A ≥ tanh in 5/6 seeds (one exact tie, one −0.02 loss).

**Fold (γ=−1.1):** seed 73895 → 41 steps / **0.74 λt**; seed 11 → 109 / 1.97 λt.
Collapses far below both arms. Non-viable as-is.

### Verdict v3 — A leads in direction, but the result is doubly confounded AND the baseline moved

Two confounds keep this from being a clean return-map win:

1. **One-step NRMSE is not matched** — A is *lower* (better) than tanh in all 6
   seeds. The fairness rule requires matched NRMSE; here A's VPT lead rides on a
   fit-quality lead, so shape is not isolated.
2. **Effective loop gain not matched** — A's central-slope gain sits outside
   nominal sr (the standing method control).

> ⚠ **Reproducibility flag (must resolve before any strong claim).** Since γ=0 is
> algebraically `std::tanh`, the v3 tanh arm should reproduce the v2 tanh arm.
> It does **not**: v2 tanh@0.88 (#10) = 3.71 λt / NRMSE 0.00222; v3 tanh@0.88
> seed 73895 = 3.02 λt / NRMSE 0.00629 — VPT down ~0.7 λt, NRMSE ~2.8× worse.
> This is **far** larger than the ~1-step `-ffast-math` reassociation jitter, so
> it is not sub-ULP noise. The NRMSE *ordering* also flipped vs v2 (there A was
> 1.7× worse one-step; here A is better). Most likely cause: the new explicit
> `noise_rng_` seeding changed the noise realization for both arms. Until the
> tanh baseline is reconciled to v2, the v3 A-lead is recorded but **not** tied
> to the campaign and **not** to be over-read. Diagnosis is the next action.
>
> **✅ RESOLVED (v4 below).** The noise guess was wrong — noise is off
> (`noise_scaling=0`, double-gated, Lorenz never enables it). The real cause was
> an **uncommitted `cfg.reservoir.history_depth = 32`** that drifted in during the
> refactor; doc-era runs used the default **16**. Pinning depth back to 16
> reproduces the v2 tanh anchor *exactly* (3.71 λt / 0.00222) — no bug — and the
> A-vs-tanh ordering reverses (see v4).

**v4 — multi-seed at history_depth=16 (the doc-era / better operating point, 2026-06-16):**

`history_depth` was the v3 confound. At seed 73895, depth 16 beats 32 by ~0.7 λt
for *both* arms and matches the doc default, so the comparison was re-run with
depth pinned to 16 (`Lorenz.cpp`; one build, arms differ only by γ). Same 6 seeds,
sr=0.88, is=0.10:

| seed | tanh steps / λt | A steps / λt | Δ λt | tanh NRMSE | A NRMSE |
|---|---|---|---|---|---|
| 73895 | 205 / 3.71 | 196 / 3.55 | −0.16 | 0.002217 | 0.003895 |
| 11 | 216 / 3.91 | 186 / 3.37 | −0.54 | 0.002315 | 0.003505 |
| 23 | 232 / 4.20 | 218 / 3.95 | −0.25 | 0.002226 | 0.003431 |
| 42 | 201 / 3.64 | 249 / 4.51 | +0.87 | 0.002379 | 0.003420 |
| 101 | 196 / 3.55 | 179 / 3.24 | −0.31 | 0.002180 | 0.003510 |
| 202 | 202 / 3.66 | 167 / 3.02 | −0.63 | 0.002478 | 0.003433 |
| **mean ± sd** | **3.78 ± 0.24** | **3.61 ± 0.54** | **−0.17** | 0.00230 | 0.00353 |

Paired diff −0.17 λt (tanh nominally ahead), SD(diff)=0.54, paired t=−0.78
(df=5, p≈0.47 — **not significant**); tanh ≥ A in 5/6 seeds (only seed 42 favors A).

**Reproducibility gate passed:** seed 73895 tanh = 3.71 λt / 0.002217 reproduces
v2 #10 (3.71 / 0.00222) *exactly* → the v3 collapse was entirely the depth drift.
A's NRMSE also reproduces (#12: 0.00391); A's VPT is −0.24 λt off #12, the known
literal→runtime `-ffast-math` free-run jitter on the A path (γ=0 cancels it for tanh).

### Verdict v4 — multi-seed PARITY; the v3 A-lead was a depth-32 (degraded-regime) artifact

Side by side (cross-seed mean of medians):

| depth | tanh | A | Δ (A−tanh) | paired p | one-step A vs tanh |
|---|---|---|---|---|---|
| 16 (good) | **3.78** | **3.61** | −0.17 | 0.47 (n.s.) | A ~1.5× **worse** |
| 32 (degraded) | 2.80 | 3.14 | +0.34 | ~0.06 | A better |

Depth 16 lifts both arms ~0.5–1.0 λt and **reverses** the ordering. The v3 "A
+0.34 λt" existed only in the worse depth-32 regime; at the better operating point
it's **parity, tanh nominally ahead by 0.17 λt (not significant)**, and A again
carries ~1.5× worse one-step NRMSE — matching every doc-era single-seed run. The
v3 NRMSE "flip" was itself a depth-32 artifact. **This is the multi-seed
confirmation of the campaign's standing conclusion: parity on Lorenz free-run; A's
only distinguishing trait is a gentler operating point, not a better return map.**

Sub-finding: **A is less seed-robust** (σ 0.54 vs tanh 0.24) — it owns the campaign
max (seed 42, 4.51 λt) but also a lower floor (seed 202, 3.02). tanh is tighter.

`history_depth` is now a known axis and **degrades monotonically above 16**:
cross-seed mean of medians is tanh 3.78 / A 3.61 at M=16, 3.53 / 3.37 at M=20,
2.80 / 3.14 at M=32. M=16 is the best of the three for both arms; depth also
changes the A-vs-tanh one-step ordering. Below 16 (M=8/12) is untested — out of
scope for the A-vs-tanh question, which v4 settles as parity.

### Locked general-purpose seeds (M=16)

For consistent future A-vs-tanh work, three seeds are locked as the canonical set —
chosen to serve **both** activations, ranked by worst-of-two-arm VPT (so neither
arm is sacrificed). Driver: `examples/Lorenz/sweep/run_sweep_canon.ps1`.

| seed | tanh λt | A λt | why |
|---|---|---|---|
| **23** | 4.20 | 3.95 | strong on both — the clear #1 |
| **42** | 3.64 | 4.51 | A champion; tanh still solid (floor 3.64) |
| **73895** | 3.71 | 3.55 | balanced; reproducibility anchor / default seed |

Seed 11 (3.91 / 3.37) was rejected as too tanh-lopsided (sub-mean on A); 101 and
202 are below average on both.

### A(x) operating-point grid — SR × IS at M=16 (2026-06-16)

A-only (γ=1.1, inv_σ²=250), `history_depth=16`, `leak_rate=1.0`, swept over
SR ∈ {0.85, 0.875, 0.90} × IS ∈ {0.20, 0.10, 0.05} at all three locked seeds
(23, 42, 73895) = 27 runs. Driver: `examples/Lorenz/sweep/run_sweep_A_srXis.ps1`;
raw output in `examples/Lorenz/sweep/A_srXis_d16/`. Goal: find A's *free-run*
operating point, since the SR≈0.88 sweet spot we'd been using came from open-loop /
NARMA tuning and was never validated under closed-loop generation.

**Seed-averaged median VPT (λt):**

| SR \ IS | 0.20 | 0.10 | 0.05 |
|---|---|---|---|
| **0.85** | 3.43 | 3.46 | 3.45 |
| **0.875** | 3.44 | 3.27 | 3.32 |
| **0.90** | 3.36 | **3.73** ⭐ | 3.62 |

Per-seed maxima all sit in the SR=0.90 row: seed 23 → 4.08 @ IS=0.10, seed 42 →
3.89 @ IS=0.10, seed 73895 → 3.71 @ IS=0.05.

**Findings:**
- **Best cell: SR 0.90 / IS 0.10 = 3.73 λt** (seed-mean), the clear maximum — but the
  *least robust* cell: seeds spread 3.21→4.08 (range 0.87 λt). Carried by 23 & 42;
  73895 prefers IS=0.05 there.
- **Robust runner-up: SR 0.90 / IS 0.05 = 3.62 λt**, tight spread (3.44–3.71, range
  0.27). The stable choice if peak-vs-robustness matters.
- **The ~0.88 sweet spot does NOT transfer to free-run for A.** The SR=0.875 row is
  the *weakest* at mid/low IS (3.27–3.32). Free-run wants SR pushed **up to 0.90**.
- **IS effect is non-monotone and seed-dependent.** IS=0.10 wins on average only via
  the SR=0.90 row. Lower IS reliably gives better *one-step* NRMSE (IS=0.05 ≈
  0.0029–0.0037 vs IS=0.20 ≈ 0.0040–0.0047) but that does not convert to VPT — the
  usual one-step/free-run decoupling.
- **⚠ SR=0.90 is the top edge of the grid and the surface is still climbing.** The
  true A optimum may sit *above* 0.90 — **open follow-up: extend SR to 0.925 / 0.95**
  (at IS 0.10 & 0.05, 3 seeds) to find where VPT rolls over.

`leak_rate` is now a CLI arg (argv[6], positive-sentinel like sr/is; -1/absent keeps
the source default 1.0) — added alongside this grid but left at 1.0 throughout it.

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
