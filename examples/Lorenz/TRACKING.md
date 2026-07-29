# Lorenz — experiment tracking (Janus harness)

Live result log for the **current** half-anchored free-run harness in this
folder. Protocol and scoring live in [`README.md`](README.md). Knobs in
[`Lorenz.h`](Lorenz.h) `config::`.

This is **not** the historical unassisted A(x)-vs-tanh campaign
([`docs/LorenzFreeRun.md`](../../docs/LorenzFreeRun.md)) — different I/O,
geometry, and free-run policy. Do not mix numbers across those docs.

**Note (API):** full-state linear feedback (FSF) has been **removed** from the
reservoir. Runs 1–3 below remain as a historical A/B log; new surveys no longer
have FSF knobs.

---

## How to read this log

| Column | Meaning |
|--------|---------|
| **#** | Run id (append only; never renumber) |
| **FSF** | Historical only — OFF, or ON with seed/scaling (feature removed) |
| **Survey** | `NUM_THREADS × NUM_RUNS` (`Lorenz.exe T R`) |
| **VPT mean*** | Mean of per-ESN-seed VPT means (lt), unless noted |
| **RMSE mean*** | Mean of per-ESN-seed free-run RMSE means |
| **VPT max** | Best single free-run in the survey (lt) |

\* Pooled over ESN seeds when the survey has more than one trial.

**Protocol defaults for recent rows** (unless a row overrides):

| Knob | Value |
|------|--------|
| Architecture | Conv(1→8, TANH) → MaxPool → Linear(8192→3) · 24683 params |
| Reservoir | dim=11 (N=2048), M=`HISTORY_DEPTH`=24, leak=1, SR target 0.99 |
| Drive | `input_scaling` 0.005 · `feedback_scaling` 0.04 |
| Train | epochs 100 · window 20000 · online HCNN |
| Free-run | window 2000 · washout on same window · VPT threshold 0.3 (channel-RMS) |
| Lyapunov | λ≈0.9056 → 1 lt ≈ 55.2 steps at dt=0.02 |
| ESN seeds (survey base) | 5941978990 … + (NUM_RUNS trials−1) as printed |
| Orbits | 10 fixed `orbit_seed`s per ESN seed (same set across A/B rows below) |

**Claim discipline**

- Half-anchored / Janus free-run only — not Pathak-style unassisted VPT.
- n=40 free-runs (4×10) is exploratory; orbit and ESN-seed variance dominate small
  deltas. Prefer larger `NUM_RUNS` before strong FSF claims.
- VPT = prediction horizon; free-run RMSE = full-window trajectory fidelity.
  They often disagree on which free-run is “best.”

---

## Results

### Run 1 — FSF A/B at default op-point (`4 × 10`)

**Date:** 2026-07-22 (session)  
**Command:** `Lorenz.exe 4 10`  
**Only delta between arms:** `FULL_STATE_FEEDBACK` false vs true  
(`FSF_SEED=9089361`, `FSF_SCALING=0.005` when ON).

#### Aggregate (mean of per-ESN-seed stats)

| Arm | VPT mean (lt) | VPT max (lt) | RMSE mean | Notes |
|-----|---------------:|-------------:|----------:|-------|
| **1a FSF OFF** | ~2.18 | 5.23 | ~0.440 | Baseline |
| **1b FSF ON**  | ~2.24 | 5.18 | ~0.432 | Mild RMSE edge; VPT ≈ wash |

**Δ (ON − OFF):** VPT mean ~+0.06 lt (~noise); RMSE mean ~−0.008 (~1.8% relative).

#### Per ESN seed

| ESN seed | VPT mean OFF → ON | VPT max OFF → ON | RMSE mean OFF → ON |
|----------|------------------:|-----------------:|-------------------:|
| 5941978990 | 2.44 → 2.37 | 5.16 → 5.18 | 0.439 → **0.428** |
| 5941978991 | 2.18 → 2.18 | 4.09 → 4.08 | 0.439 → **0.433** |
| 5941978992 | 2.35 → 2.22 | **5.23** → 3.10 | 0.441 → **0.433** |
| 5941978993 | 1.74 → **2.17** | 2.95 → **4.27** | 0.441 → **0.434** |

#### Orbit notes (same 10 orbits both arms)

| Orbit seed | Role in this set |
|------------|------------------|
| `16519405431026665550` | Often easiest by VPT (top or top-3 on most ESN seeds). OFF peaks 5.16 / 5.23 lt; ON drops that seed’s long tail on ESN 8992 (5.23 → 3.10) while RMSE still improves overall. |
| `3731162586949553184` | Consistently hard (~0.34–0.38 lt) **both** arms — FSF did not rescue it. |
| Others | Mid-pack; ranking shuffles with ESN seed and FSF. |

#### Verdict (Run 1)

At this scale, FSF looks like a **small free-run RMSE regularizer**, not a VPT
breakthrough. Effect size is small vs **orbit variance** and **ESN-seed
variance**. Worth larger surveys and/or an `FSF_SCALING` sweep before claiming a
win.

---

### Run 2 — FSF scaling: weak 0.005 vs stronger 0.008 (`4 × 10`)

**Date:** 2026-07-22 (session)  
**Command:** `Lorenz.exe 4 10`  
**Fixed:** FSF ON, `FSF_SEED=9089361`, same ESN seeds / 10 orbits / dim=11 protocol as Run 1.  
**Only delta:** `FSF_SCALING` **0.005** (weak) vs **0.008** (stronger).  
Weak arm is the same realization as Run **1b**.

#### Aggregate (mean of per-ESN-seed stats)

| Arm | FSF scale | VPT mean (lt) | VPT max (lt) | RMSE mean | Notes |
|-----|----------:|--------------:|-------------:|----------:|-------|
| **2a weak** | 0.005 | ~2.24 | 5.18 | ~0.432 | = Run 1b |
| **2b stronger** | 0.008 | ~2.19 | 5.16 | ~0.441 | VPT wash; **RMSE worse** |
| *(Run 1a OFF)* | — | ~2.18 | 5.23 | ~0.440 | for context |

**Δ (0.008 − 0.005):** VPT mean ~−0.05 lt (noise); RMSE mean **~+0.009** (worse).  
Stronger arm’s RMSE is back near **FSF OFF**, so the mild 0.005 RMSE edge is lost.

#### Per ESN seed

| ESN seed | VPT mean 0.005 → 0.008 | VPT max 0.005 → 0.008 | RMSE mean 0.005 → 0.008 |
|----------|-----------------------:|----------------------:|------------------------:|
| 5941978990 | 2.37 → **2.16** | 5.18 → 4.13 | 0.428 → **0.438** |
| 5941978991 | 2.18 → **2.46** | 4.08 → **5.16** | 0.433 → 0.433 |
| 5941978992 | 2.22 → 2.28 | 3.10 → **4.17** | 0.433 → 0.437 |
| 5941978993 | 2.17 → **1.85** | 4.27 → 3.06 | 0.434 → **0.456** |

Mixed per-seed VPT (one up, two down, one flat); **RMSE not improved on any seed** and clearly worse on 8990 / 8993.

#### Orbit notes

| Orbit seed | Role |
|------------|------|
| `16519405431026665550` | Still often easy, but not always VPT #1 at 0.008 (e.g. 8990 prefers `898352…` at 4.13 lt; easy orbit drops 5.18 → 3.12 on that seed). At 0.008 / 8991 it recovers to 5.16 lt. |
| `3731162586949553184` | Still hard (~0.34–0.38 lt) at both scalings — stronger FSF does not help. |

#### Verdict (Run 2)

In this bracket, **0.005 beats 0.008 on free-run RMSE** with no VPT gain from turning FSF up. Stronger feedback looks more like **overdrive / re-randomized difficulty** across seeds than a better free-run operating point. Prefer **0.005** (or a finer probe below 0.008) over pushing higher on this fixed seed/orbit set. n=40 free-runs — still exploratory.

---

### Run 3 — FSF scaling 0.003 (fill-in on the ladder)

**Date:** 2026-07-22 (session)  
**Command:** `Lorenz.exe 4 10`  
**FSF ON** `FSF_SEED=9089361` **`FSF_SCALING=0.003`**. Same ESN seeds / orbits / protocol as Runs 1–2.

#### Aggregate ladder (mean of per-ESN-seed stats)

| Arm | Scale | VPT mean (lt) | VPT max (lt) | RMSE mean |
|-----|------:|--------------:|-------------:|----------:|
| 1a OFF | — | ~2.18 | 5.23 | ~0.440 |
| **3 @ 0.003** | **0.003** | **~2.25** | **5.85** | **~0.436** |
| 2a / 1b | 0.005 | ~2.24 | 5.18 | **~0.432** |
| 2b | 0.008 | ~2.19 | 5.16 | ~0.441 |

**0.003 vs OFF:** VPT ~+0.07 lt; RMSE ~−0.004 (mild).  
**0.003 vs 0.005:** VPT ≈ tie; RMSE **worse than 0.005 by ~0.004** (0.005 still best RMSE).  
**0.003 vs 0.008:** better on both aggregates than the strong arm.

#### Per ESN seed @ 0.003

| ESN seed | VPT mean | VPT max | RMSE mean |
|----------|---------:|--------:|----------:|
| 5941978990 | 2.36 | 5.14 | **0.422** |
| 5941978991 | 2.12 | 4.11 | 0.442 |
| 5941978992 | 2.30 | 5.47 | 0.442 |
| 5941978993 | 2.23 | **5.85** | 0.441 |

RMSE edge vs OFF is concentrated on seed **8990** (0.439 → 0.422); other seeds ≈ flat vs OFF. Best single free-run in the whole ladder so far: **5.85 lt** on seed 8993 / orbit `1642173382990087416` (not the usual easy orbit).

#### Orbit notes @ 0.003

| Orbit seed | Role |
|------------|------|
| `16519405431026665550` | Still often strong (5.14 / 3.08 / 5.47 / 2.97 lt); pairs with best RMSE on seed 8990 (0.374). |
| `1642173382990087416` | New VPT champion this arm (5.85 lt on 8993) — orbit ranking still seed-dependent. |
| `3731162586949553184` | Still hard (~0.34–0.38 lt). |

#### Verdict (Run 3 + scaling ladder)

Monotone-ish **RMSE vs scale** in this set: **OFF (0.440) → 0.003 (0.436) → 0.005 (0.432) → 0.008 (0.441)**, with a sweet spot near **0.005** and a clear downturn by 0.008. VPT means stay in a ~2.2 lt noise band; max VPT is lottery-like. Working default for further Lorenz work: **FSF ON @ 0.005** (or OFF if you want the no-FSF baseline). n still exploratory.

---

### Run 4 — Janus baseline 30×100 (`FORWARD_ONLY = false`)

**Date:** 2026-07-28  
**Command:** `Lorenz.exe 30 100` (Release)  
**Arm:** Janus — real past + future external feedback  
**Raw log:** `examples/Lorenz/Lorenz.exe-30-100_future_and_past.txt`

**Protocol deltas vs Runs 1–3 defaults:** post-FSF harness; HCNN Conv(1→2) → MaxPool →
Linear(4096→3) · 12319 params · B=2; free-run window **1000** (not 2000); ESN seeds
`21978990`…`21979019` (30 trials); **100** free-runs per trial (not 10). GS proxies
(duty / n_relock / meanLock) **not present** in this capture.

#### Aggregate (mean of 30 per-ESN-seed stats)

| Arm | Survey | VPT mean (lt) | VPT max (lt) | RMSE mean | Notes |
|-----|--------|--------------:|-------------:|----------:|-------|
| **4 Janus** | 30×100 | **1.91** | **10.07** | **0.422** | Across-seed std VPT 0.15, RMSE 0.004 |

Per-seed VPT means: min 1.50 / median 1.93 / max 2.19.  
Per-seed VPT medians sit lower (~1.56 mean) — right-skewed; long tails exist.  
Best single free-run RMSE among leaderboards: **0.277** (9.18 lt).

#### Soft / strong ESN seeds (by mean VPT)

| | ESN seed | VPT mean | RMSE mean |
|--|---------:|---------:|----------:|
| Soft | 21979018 | 1.50 | 0.437 |
| Soft | 21979017 | 1.63 | 0.424 |
| Soft | 21979014 | 1.64 | 0.426 |
| Strong | 21978997 | 2.19 | 0.420 |
| Strong | 21979019 | 2.12 | 0.417 |
| Strong | 21979013 | 2.09 | 0.421 |

#### Peak free-runs

| Orbit seed | ESN seed | VPT | RMSE | Role |
|------------|---------:|----:|-----:|------|
| `9333312947715283458` | 21978990 | **10.07 lt** (556 steps) | 0.294 | Survey VPT ceiling |
| `18109467897393425238` | 21979013 | 9.18 lt | **0.277** | Best RMSE on boards |

#### Verdict (Run 4)

Solid Janus reference at production survey size: **~1.9 lt** typical first-upcrossing,
with rare ceiling runs to **~10 lt**. VPT is orbit-lottery. **(A)** anchor for the
forward-only ablation (Run 5).

---

### Run 5 — Forward-only 30×100 (`FORWARD_ONLY = true`)

**Date:** 2026-07-28  
**Command:** `Lorenz.exe 30 100` (Release)  
**Arm:** forward-only — past input zeroed every ReservoirStep (train + washout + free-run)  
**Raw log:** `Lorenz.exe-30-100_future_channels2.txt`  
**Matched to Run 4:** same dim/M/seeds/100 orbits protocol.

GS proxies **present** in this capture (duty / n_relock / n_unlock / meanLock).

#### Aggregate (mean of 30 per-ESN-seed stats) — VPT + GS primary

| Arm | Survey | VPT mean (lt) | VPT max (lt) | duty mean | n_relock mean | Notes |
|-----|--------|--------------:|-------------:|----------:|--------------:|-------|
| 4 Janus | 30×100 | 1.91 | 10.07 | *(not in log)* | *(not in log)* | past + future |
| **5 fwd-only** | 30×100 | **2.03** | **9.33** | **0.475** | **15.7** | past=0 |

Per-seed VPT means (fwd-only): min 1.61 / median 2.05 / max 2.29.  
Duty means cluster tightly (~0.45–0.49). n_relock means ~15–16 per free-run window.

#### A/B (Δ = fwd-only − Janus) on headline VPT

| Metric | Janus (4) | Fwd-only (5) | Δ |
|--------|----------:|-------------:|--:|
| VPT mean (lt) | 1.91 | **2.03** | **+0.12** |
| VPT max (lt) | **10.07** | 9.33 | −0.74 |
| duty mean | — | 0.475 | GS only on arm B log |

**Do not over-read +0.12 lt** — same order as across-seed VPT scatter (Run 4 std ~0.15). Ceiling still higher on Janus (storefront clip orbit). Forward-only does **not** collapse first-upcrossing; if anything it sits slightly above Janus mean in this pair of 30×100 runs.

#### Verdict (Run 5 + A/B)

Zeroing past is **not** a VPT cliff at this op-point. Mean VPT holds (~2 lt); GS duty ~0.48 with ~16 relocks/window shows re-lock is active under forward-only. Prefer GS-complete Janus re-run if claiming duty/relock A/B (arm A log lacked those lines). Primary claim so far: **past is not load-bearing for mean VPT** in this matched survey.

---

## Queue / next experiments

### Decision pending — lock-in tomorrow (2026-07-29)

Parked after Run 4 vs 5 re-read (both logs now have full GS; mean-of-means
VPT 1.91 vs 2.03, duty 0.472 vs 0.475, RMSE ~0.422 both; Janus ceiling 10.07 vs
fwd 9.33; fwd better VPT mean on 26/30 seeds).

> If the goal is a clean free-run storefront and literature-comparable story,
> default the **example** path to forward-only (or a true unassisted free-run
> harness). **Janus is not discarded** — it moves into a Research Areas tree for
> parameter-space exploration (past dose, lag curriculum, dual-drive stress,
> ablations, vanilla-ESN-under-Janus, etc.). Run 4/5 only shows past is not
> load-bearing for mean VPT/GS at *this* op-point; that is one point in a large
> space, not a verdict on Janus as a research program.

**To lock in (example only):** simplify `examples/Lorenz` narrative / default
toward forward free-run; keep Janus alive under Research Areas — do not re-run
A/B at the same dose unless exploring a new point in param space.

### Verify duty (suspected reverse) + err-vs-lt plots

- [x] **Review duty-cycle math** in `Lorenz::FreeRun` (2026-07-28 code audit) —
  **not reversed.** Definitions and implementation agree:
  - `locked ⇔ channel-RMS err ≤ θ` (`θ = VPT_THRESHOLD = 0.30`)
  - `duty = locked_steps / steps` = fraction of generative steps **on-track**
  - VPT = first step with `err > θ` (complementary at equality: `err == θ` is
    locked, does **not** fire VPT)
  - CSV `locked` column, survey label `duty (<=theta)`, README, and
    `plot_freerun_trace.py` all use the same convention
  - State machine (relock/unlock/meanLock) reimplemented in Python against
    synthetic traces: duty matches independent `mean(err≤θ)`; sojourn sum
    equals locked step count; relock only after a prior unlock
  - Plausibility: mean VPT ~2 lt on a 1000-step (~18 lt) window would give
    duty ~0.11 if error never recovered; reported duty ~0.47 implies frequent
    re-lock after first upcrossing — that is the GS story, not a sign flip
  - Fixed comment drift only: numerator was documented as `< θ`, code is `≤ θ`
- [ ] **Plot a few free-runs:** error vs Lyapunov time with horizontal θ line
  (use `--trace` + `plot_freerun_trace.py`). Visually count fraction of steps
  above/below threshold vs reported duty / VPT. Suggested: orbit
  `9333312947715283458` on Janus `21978990` (VPT 10.07) and fwd-only
  `21978993` (VPT 9.33); plus one median-ish run.

### Post-FSF (current code — FSF removed)

- [x] **Janus baseline** (`FORWARD_ONLY = false`): Run 4 — `30 × 100`  
- [x] **Forward-only** (`FORWARD_ONLY = true`): Run 5 — `30 × 100`  
- [x] Side-by-side VPT + GS (duty/relock) — Run 4 vs 5 both logs complete  
- [ ] **Lock-in** storefront default = forward-only / unassisted; Janus optional (see note above)
- [x] **Duty math audit** — not reversed (see note above)  
- [ ] **err-vs-lt visual plots** still open (see note above)
- [ ] **Free-run re-lock storefront story** (see “tomorrow” section below)

### Tomorrow (2026-07-29) — free-run re-lock as the demo (not Janus)

Janus vs forward-only is a **wash** (past not load-bearing). The potentially
compelling result is the free-run **signature**, present on **both** arms:

> Short mean VPT (~2 LT) + high duty (~0.47) + many relocks (~16 / window) —
> **not** the pure “die after VPT” classical ESN free-run picture
> (where duty ≈ VPT/T ≈ 0.11 for a 1000-step / ~18 LT window and n_relock ≈ 0).

Hero metric shift: less “max first-upcrossing VPT vs literature 4–15 LT,” more
**intermittent re-entrant tracking** under closed-loop free-run (same θ as VPT).
Sell **forward-only / unassisted-style** free-run; Janus optional.

**To do tomorrow:**

- [ ] **err-vs-lt plots** (`--trace` + `plot_freerun_trace.py`): storefront orbit
  `933331…` on Janus `21978990` and fwd-only `21978993`; plus a median-ish run.
  Confirm structured re-entries under θ with pred re-aligning to true, not random
  thrashing around the bar.
- [ ] **θ sensitivity:** duty / n_relock at θ ∈ {0.1, 0.2, 0.3} (same seeds if cheap).
  Kill “θ too soft vs climate floor.”
- [ ] **Climate / null:** late-window or shuffled channel-RMS vs θ; post-VPT locked
  fraction separately from full-window duty.
- [ ] **Same-metric weak baseline** (optional but strong): classic ridge ESN or
  weak op-point under **identical** θ, dt, window, scoring — show die-after-VPT
  vs HypercubeESN re-lock signature.
- [ ] **Lock-in narrative:** forward-only default; README/storefront lead with
  re-lock figure + duty/n_relock; VPT secondary; do **not** claim Janus enables
  GS or beat literature VPT from current means alone.
- [ ] **Lock-in code/docs** from decision note above (`FORWARD_ONLY` default /
  example wording).

**Claim-safe frame (if plots + nulls hold):** HypercubeESN free-run on Lorenz-63
shows persistent intermittent lock under closed-loop generation — early first
failure, substantial time still under θ with O(10) re-locks per ~18 LT window —
unlike textbook die-after-VPT free-run.

### Tomorrow — Research Areas split + simplify example Lorenz

**Intent:** Janus stays. Research Areas is where Janus **grows** (param sweeps,
dose, curriculum, stress tests, vanilla comparison) — not a graveyard. The
example path gets simpler so storefront ≠ research cockpit.

- [ ] Create top-level **`Research Areas/`** (or `ResearchAreas/`) folder in the
  repo and **copy** the full Lorenz project there as the **active Janus research
  sandbox** (history, TRACKING, dual-cursor, ablation knobs, room for sweeps).
- [ ] Keep **`examples/Lorenz/`** as the public example path, but **rip out Janus
  cursor wiring there only**: simple forward-moving free-run timebase (likely
  relabel the “future” cursor); drop past/anchor complexity from the *example*.
- [ ] Example target: clean free-run storefront (literature-comparable story).
- [ ] Research target: Janus param-space exploration (input/feedback scaling,
  window/lag geometry, θ, FORWARD_ONLY × dose, train-vs-free-run ablations,
  re-lock signature, vanilla ESN under Janus, etc.) without burdening example
  README.

### Research — vanilla ESN under the Janus cursor method (Python)

Not “vanilla free-run without Janus.” Goal: **classical sparse random ESN + ridge
(or simple linear) readout**, driven by the **same Janus method** as the C++
harness (dual cursors, 4+4 past/future ports, lag curriculum, free-run = real
past + self-feedback future, same θ VPT/duty/relock scoring).

- [ ] **Python MVP** (~1–1.5 day): NumPy vanilla ESN; Janus geometry
      (port/reimplement); train + free-run; metrics aligned with Runs 4–5;
      small multi-seed table under Research Areas (preferred home).
- [ ] Answer: under Janus, does classic ESN (a) wash like Hypercube, (b) need
      the past, (c) die-after-VPT, or (d) collapse?
- [ ] Optional later: matched orbit/seed survey vs Hypercube Janus log; C++
      twin only if the same CLI/survey binary is required.

### Historical FSF (feature removed — log only)

- [ ] Larger A/B: same seeds, `Lorenz.exe 4 50` at OFF vs 0.005 if claiming the RMSE edge  
- [x] `FSF_SCALING` ladder at fixed `FSF_SEED=9089361`: OFF / 0.003 / 0.005 / 0.008 (Runs 1–3)  
- [ ] Alternate `FSF_SEED` at best scaling (~0.005)  
- [x] Hard orbit `373116…` still ~0.34–0.38 lt at all scalings tried (no rescue)  

---

## How to append a run

1. Toggle only the intended knobs in `Lorenz.h` `config::`; rebuild `Lorenz`.  
2. Note command, Release/Debug, and full FSF triple (`FULL_STATE_FEEDBACK`, seed, scaling).  
3. Paste a new **Run N** section: aggregate table first, per-seed if useful, orbit notes if they move.  
4. One-line verdict; update the queue checklist.  
5. Do **not** rewrite earlier rows — correct only if a paste error is found, and mark the edit.

---

## Related

| Doc | Role |
|-----|------|
| [`README.md`](README.md) | Protocol, ports, scoring, CLI cost model |
| [`JanusCursor.md`](JanusCursor.md) | Janus cursor geometry |

| [`docs/LorenzFreeRun.md`](../../docs/LorenzFreeRun.md) | Historical A(x) vs tanh (stale harness) |
