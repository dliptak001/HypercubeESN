# Memory Capacity — Linear Short-Term Memory Diagnostic

## What this example measures

Memory Capacity (MC) is the standard Jaeger (2001) probe of a reservoir's
**linear short-term memory**: how many past inputs can be linearly reconstructed
from the present reservoir state. The reservoir is driven by white noise `u(t)`,
and for each lag `k` a linear readout is fit to reproduce `u(t-k)` from the state
`x(t)`. The squared correlation `r²(k)` between target and reconstruction is the
fraction of lag-`k` information the state retains; summing over lags gives

```
MC = Σ_k r²(k)
```

Unlike NARMA (which couples memory *and* nonlinear mixing), MC isolates memory:
the targets are the raw delayed inputs, so a high MC means the state is a rich
linear record of recent history. The reservoir is never trained or modified —
only its raw state is read. No HCNN, no ESN coupling.

## The two configs

The measurement is built around a clean split between two kinds of settings:

- **`MCConfig`** — the *experiment*: drive length (`t_warmup`, `t_collect`), lag
  range (`k_max`), ridge (`ridge_lambda`), train/test split (`train_frac`),
  feature cap, drive seed (`input_seed`), and the early-stop streak. It fixes a
  **meter** and never changes within a sweep.
- **`ReservoirConfig`** — the *operating point*: `spectral_radius`, `leak_rate`,
  `history_depth`, `seed`, `input_scaling`. It is **passed into `Measure()`** per
  cell. There is no parallel "extended parameter list" — the operating point is
  exactly a `ReservoirConfig`, the same struct the rest of the library uses.

This is why a sweep is cheap and comparable: the meter generates **one** white-noise
drive up front and reuses it for every operating point, so every cell scores the
byte-identical task and differences are attributable to the reservoir alone.

## The measurement (`MemoryCapacityMeter::Measure`)

```
white noise u(t) ──> Reservoir ──> state matrix X ──> ridge readout per lag k
   [-1,+1]            (fixed)        T rows × F cols      r²(k) on held-out rows
```

1. **Drive & collect.** Feed `u(t)` for `t_warmup + t_collect` steps. Discard the
   warmup and the first `k_max` rows (so every lag target is a valid post-warmup
   input), then record the state into `X` — `T = t_collect - k_max` rows of `F`
   features (`F = min(N, feature_cap)`). (`T` is the row count here; `M` denotes
   the delay depth in the Results sweep below.)
2. **Fit on train, score on test.** Build the regularized train Gram
   `XᵀX + λI` over the first `train_frac` rows, Cholesky-factor it **once**, then
   for each lag `k` solve for the readout weights and score `r²(k)` on the
   held-out rows. Held-out scoring matters: in-sample R² on `T ~ a few × F`
   samples inflates the headline by a roughly lag-independent margin.
3. **Metric.** `r²(k)` is squared **Pearson** correlation, not regression
   `R² = 1 - SS_res/SS_tot`. They agree only with an intercept; the linear readout
   has none, so the Pearson form is the canonical MC metric and stays in `[0, 1]`.
4. **Early stop (sweeps only).** The memory function decays, so once `r²(k)` holds
   below `early_stop_thresh` for `early_stop_patience` consecutive lags the loop
   stops — remaining lags only add noise floor. A full-curve run
   (`MeasureOptions{early_stop=false}`) ignores this and computes every lag.

`Measure()` is `const` and allocates all working buffers locally, so the same
meter can be measured concurrently from many threads — which is what `RunSweep`
relies on.

## The pieces

| File | Role |
|------|------|
| `MCLinalg.h` | Pure double-precision kernels: cache-blocked Gram build, in-place Cholesky, triangular solve, `XᵀY`. No state, no I/O — the math. |
| `MemoryCapacity.h` | The engine: `MCConfig`, `MCResult`, `MemoryCapacityMeter` (constructed with `dim`), and the parallel `RunSweep`. |
| `MemoryCapacity.cpp` | The driver: four run modes + table formatting. `main()` picks a mode. |

## Run modes (`MemoryCapacity.cpp`)

Each mode builds `ReservoirConfig`s from a base operating point and fans them out.

- **`RunDetailed`** — one operating point, the full per-lag `r²(k)` table plus the
  headline `Total MC` and the last lag crossing `r² > {0.5, 0.1, 0.01}`.
- **`RunGridSweep`** — an `sr × leak × history_depth` grid. Prints an ordered
  results table and one `TotalMC` grid (rows = sr, cols = leak) per depth. The
  base op-point supplies the fixed `seed` and `input_scaling`.
- **`RunSeedSurvey`** — hold the op-point fixed, sweep the reservoir `seed` over an
  inclusive range. Reports per-seed MC, a summary, and an SR-band-filtered
  followup + top-5 (the per-seed rescale lands each realization at a slightly
  different realized SR; the band keeps only those at the same operating point).
- **`RunDepthProbe`** — per-lag `r²(k)` curves for several depths side-by-side.
  Discriminates whether a `Total-MC` dip at some `M` is real dynamics (`r²(1)`
  intact, smoothly faster-decaying tail) or an indexing/injection glitch (a
  discontinuity, or an across-the-board depression hitting the lag-1 term).

## Reading the results

- **`TotalMC`** — `Σ_k r²(k)`. Theoretical ceiling is `F` (one unit of capacity
  per linearly independent state dimension); real reservoirs land far below it.
- **`k>.5 / k>.1 / k>.01`** — the last lag whose reconstruction still clears that
  `r²`. These are the memory *horizons*: how far back the state still carries
  recoverable signal at strong / moderate / faint fidelity.
- **`realSR`** — the realized post-rescale spectral radius. Memory generally grows
  as `sr → 1⁻` (longer fading memory) and depth × radius is super-multiplicative,
  but pushing `sr` past the echo-state boundary breaks reproducibility.

## Results — DIM 11, spectral-radius × delay-depth sweep

A single authoritative `RunGridSweep` at **DIM 11** (N = F = 2048 — full state,
below the 8192 feature cap), `input_scaling = 0.06` (weak drive, the
memory/margin regime), `leak_rate = 1.0`, `warmup = 2000`, `collect = 15000`,
`Kmax = 2000`, `ridge = 1e-4`. The grid crosses four spectral radii — two
contractive (0.90, 0.95), the echo-state edge (1.00), and one supercritical
(1.10) — with the delay depth `M` (`history_depth`) at {1, 2, 4, 8, 16, 32, 64}.
`realSR` landed within 0.003 of every target, so the spread below is the
operating point, not rescale drift.

`TotalMC` by depth (rows) and spectral radius (columns). With `leak_rate` fixed
at 1.0, `RunGridSweep`'s per-depth `sr × leak` grids each reduce to one column;
they are re-pivoted here into a single depth × sr table:

| M \ sr |      0.90 |       0.95 |       1.00 |       1.10 |
|-------:|----------:|-----------:|-----------:|-----------:|
| 1      |     36.50 |      44.96 |      54.06 |      50.84 |
| 2      |     36.94 |      45.33 |      56.80 |      85.05 |
| 4      |     56.42 |      80.64 |     127.96 | **128.41** |
| 8      |     65.72 |      97.76 |     183.04 |      86.44 |
| 16     | **75.44** | **158.11** |     374.07 |       4.76 |
| 32     |     75.15 |     152.34 |     837.21 |       0.01 |
| 64     |     64.00 |     150.92 | **922.34** |       0.00 |

Memory horizons (last lag clearing the threshold) at sr = 1.00, the echo-state
edge, showing how far the state actually reaches:

| sr   | M  | TotalMC | k>.5 | k>.1 | k>.01 |
|------|---:|--------:|-----:|-----:|------:|
| 1.00 |  8 |  183.04 |  184 |  209 |   240 |
| 1.00 | 16 |  374.07 |  374 |  430 |   496 |
| 1.00 | 32 |  837.21 |  843 |  988 |  1143 |
| 1.00 | 64 |  922.34 |  961 | 1562 |  2000 |

What the run shows, at a glance:

- **Depth pays, scaled by radius.** Capacity peaks around M ≈ 16 at the
  contractive radii (sr 0.95 plateaus there, sr 0.90 turns back down) but never
  stops climbing at the echo-state edge (sr = 1.00).
  **Depth × radius is super-multiplicative** — deep delay lines only cash in once
  the recurrent memory is long enough to fill them.
- **The first real gain is M = 2 → 4.** M = 1 and M = 2 are nearly flat.
- **Supercriticality (sr = 1.10) is a knife edge** — it wins at shallow depth but
  collapses to near zero by M ≥ 16 as the dynamics lose the echo-state property.
  Usable window: M ≈ 2–4.
- **The top number (922) is an undercount.** `TotalMC` sums `r²(k)` over lags,
  but the sweep only measures out to lag `Kmax = 2000`. At sr = 1.00, M = 64 the
  state still had recoverable memory at that last lag, so the sum was cut off
  before the memory decayed — the true capacity is **above 922**. M = 32 decays
  inside the window, so its 837 is complete. Every cell still sits far below the
  theoretical F = 2048 ceiling.
