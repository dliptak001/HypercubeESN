# Lorenz — experiment tracking (Janus harness)

Live result log for the **current** half-anchored free-run harness in this
folder. Protocol and scoring live in [`README.md`](README.md). Knobs in
[`Lorenz.h`](Lorenz.h) `config::`.

This is **not** the historical unassisted A(x)-vs-tanh campaign
([`docs/LorenzFreeRun.md`](../../docs/LorenzFreeRun.md)) — different I/O,
geometry, and free-run policy. Do not mix numbers across those docs.

---

## How to read this log

| Column | Meaning |
|--------|---------|
| **#** | Run id (append only; never renumber) |
| **FSF** | OFF, or ON with `fsf_seed` / `fsf_scaling` |
| **Survey** | `NUM_THREADS × NUM_RUNS` (`Lorenz.exe T R`) |
| **VPT mean*** | Mean of per-ESN-seed VPT means (lt), unless noted |
| **RMSE mean*** | Mean of per-ESN-seed free-run RMSE means |
| **VPT max** | Best single free-run in the survey (lt) |

\* Pooled over ESN seeds when the survey has more than one trial.

**Protocol defaults for recent rows** (unless a row overrides):

| Knob | Value |
|------|--------|
| Architecture | Conv(1→8, TANH) → MaxPool → Linear(8192→3) · 24683 params |
| Reservoir | DIM=11 (N=2048), M=`HISTORY_DEPTH`=24, leak=1, SR target 0.99 |
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

## Queue / next experiments

- [ ] Larger A/B: same seeds, `Lorenz.exe 4 50` (or more) both FSF OFF and ON  
- [ ] `FSF_SCALING` sweep at fixed `FSF_SEED` (e.g. 0.001 / 0.005 / 0.01 / 0.02)  
- [ ] Alternate `FSF_SEED` at best scaling  
- [ ] Record whether hard orbit `373116…` ever improves under any FSF setting  

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
| [`JanusCursor.md`](JanusCursor.md) | Dual-cursor geometry |
| [`docs/full_state_linear_feedback.md`](../../docs/full_state_linear_feedback.md) | FSF mechanism |
| [`docs/LorenzFreeRun.md`](../../docs/LorenzFreeRun.md) | Historical A(x) vs tanh (stale harness) |
