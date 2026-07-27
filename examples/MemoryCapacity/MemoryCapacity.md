# Memory Capacity — Linear Short-Term Memory Diagnostic

Quick reference for the HypercubeESN MC diagnostic. **Engine + how to run** first;
**lookup tables** are archived reference campaigns (not auto-synced to whatever
`main()` currently has checked in).

---

## What this measures

Memory Capacity (MC) is the Jaeger (2001) probe of a reservoir's **linear
short-term memory**: how many past inputs can be linearly reconstructed from the
present state. Drive with white noise `u(t) ~ U[-1,+1]`; for each lag `k` fit a
ridge readout to `u(t-k)` from state `x(t)`. Squared **Pearson** correlation
`r²(k)` on **held-out** rows is the retained lag-`k` fraction; the headline is

```
MC = Σ_k r²(k)
```

Unlike NARMA (memory + nonlinear mixing), MC isolates memory. No HCNN, no ESN
coupling — only `Reservoir` state (newest delay-line slice / `Outputs()`).

| Symbol | Meaning |
|--------|---------|
| **N** | Reservoir size: `N = 2^DIM` (DIM in **[5, 16]**). |
| **F** | MC features: `F = min(N, feature_cap)` (default cap 8192). Full-state → **`F = N`**. Theoretical MC ceiling is `F`. |
| **M** (tables) | `history_depth` — delay-line depth, any integer in **[1, 64]**. |
| **M_usable** (banner) | Collected state rows = `t_collect - k_max` (train + test). **Not** history_depth. |
| **TotalMC** | `Σ_k r²(k)` over scored lags. |
| **MC/F** | Utilization `TotalMC / F`. |
| **k>.5 / .1 / .01** | Last lag with `r²` above that floor. |
| **realSR** | Post-rescale spectral radius. |

**Tail marks** (grid / seed printers):

| Mark | Meaning |
|------|---------|
| `*` | Open tail — last scored `r²` still ≥ early-stop floor → **TotalMC is a lower bound**. Raise `k_max` / warmup / collect. |
| `e` | Early-stop — sub-threshold streak; curve decayed; sum complete for that cell. |
| blank | Full `k_max` scored and tail closed. |
| `n/a` | Cell OOM or train Gram not PD (raise ridge / lower workers). |

---

## Code layout

| Piece | Role |
|-------|------|
| `MCLinalg.h` | Double Gram / Cholesky / solve kernels (no I/O). |
| `MemoryCapacity.h` | `MCConfig`, `MCResult`, `MemoryCapacityMeter::Measure`, parallel `RunSweep`. |
| `MemoryCapacity.cpp` | Four run modes + tables. Edit **`main()`** to pick mode / axes. |

### Two configs

- **`MCConfig`** — the **meter** (fixed for a sweep): `t_warmup`, `t_collect`,
  `k_max`, `ridge_lambda`, `train_frac`, `feature_cap`, `input_seed`, early-stop
  knobs. One white-noise drive is generated at meter construction and **reused**
  for every cell (byte-identical task).
- **`ReservoirConfig`** — the **op-point** passed into `Measure()` per cell: `sr`,
  `leak`, `history_depth`, `seed`, `input_scaling`, FSF. `Measure()` forces
  `dim` / `num_inputs=1` / `verbose=false` so the feature layout cannot desync.

Library defaults on `MCConfig` (header): warmup **2000**, collect **15000**,
`k_max` **2000**, ridge **1e-4**, train_frac **0.7**. Constraint:
`k_max ≤ t_warmup` and `k_max < t_collect`.

### Run modes (`MemoryCapacity.cpp`)

| Mode | What it does |
|------|----------------|
| **`RunDetailed`** | One op-point; early_stop **off**; summary + sparse `r²(k)` dump. |
| **`RunGridSweep`** | Parallel `sr × leak × history_depth` via `RunSweep`. Ordered cell table + auto pivot: **M×sr** if leak is a singleton, **M×leak** if sr is a singleton, else one **sr×leak** grid per M. |
| **`RunSeedSurvey`** | Inclusive seed range `[start, end]` at fixed op-point; mean/median/std; optional realSR band filter + top-5. |
| **`RunDepthProbe`** | Full curves for several depths side-by-side (early_stop off). |

Progress on **stderr**; tables on **stdout**. Concurrent workers self-schedule
from an atomic counter; optional RAM budget caps worker count. Banner prints
live meter + layout. Edit `main()` for DIM / seed / grid axes.

---

## Reference campaign — multi-DIM TotalMC grids

Archived single-seed `RunGridSweep` results. **FSF off.** Every cell
early-stopped (`e`); no open-tail undercount. To reproduce: DIM 5…12 in turn,
seed **47397376**, sr `{0.9, 0.95, 1.0}`, leak `{1.0}`,
M `{1,2,4,8,16,32,40,48,56,64}`, same extended meter as the table below.

| Knob | Value |
|------|--------|
| DIM | 5 … 12 (one grid each; `N = F = 2^DIM`) |
| Spectral radius | 0.90 · 0.95 · 1.00 |
| Leak | **1.0** (singleton → M × sr pivot) |
| `history_depth` M | 1, 2, 4, 8, 16, 32, 40, 48, 56, 64 |
| `input_scaling` | 0.06 |
| Reservoir seed | **47397376** |
| Meter | warmup **4000** · collect **25000** · Kmax **4000** · ridge **1e-4** · train_frac **0.7** · `input_seed` **0xc0ffee** |
| FSF | **OFF** |

Seed dependence is not in these tables — treat peaks as **indicative**. Use
`RunSeedSurvey` (or Appendix B) before locking an op-point for other tasks.

For **leak ≠ 1** at DIM 10, see [Appendix A](#appendix-a--dim-10-leak--sr--m).
For **seed variance** at M = 30–34 / sr = 1.00, see
[Appendix B](#appendix-b--dim-10-seed-survey-m--3034).

---

## At a glance — peak TotalMC by DIM

Max cell in each grid (and the M / sr that hit it). Prefer **sr = 1.00** rows
when memory length matters; contractive radii plateau earlier.

| DIM | N = F | Peak TotalMC | at M | at sr | MC/F |
|----:|------:|-------------:|-----:|------:|-----:|
| 5 | 32 | 30.09 | 48 | 0.95 | 0.94 |
| 6 | 64 | 61.76 | 32 | 0.95 | 0.97 |
| 7 | 128 | **124.64** | 64 | 1.00 | 0.97 |
| 8 | 256 | **249.44** | 40 | 1.00 | 0.97 |
| 9 | 512 | **495.00** | 64 | 1.00 | 0.97 |
| 10 | 1024 | **819.29** | 40 | 1.00 | 0.80 |
| 11 | 2048 | **993.28** | 64 | 1.00 | 0.48 |
| 12 | 4096 | **1383.41** | 64 | 1.00 | 0.34 |

Notes from the grids:

- **Depth × radius is super-multiplicative.** At sr = 0.90 capacity often plateaus
  by M ≈ 16–32; at sr = 1.00 it keeps climbing (especially DIM ≥ 9).
- **Non-power-of-two M matter.** M ∈ {40, 48, 56} often beat or match nearby
  powers of two (e.g. DIM 10: M=40 → 819 vs M=32 → 623 / M=64 → 567 at sr=1.00;
  DIM 11: M=48–64 dominate).
- **Not monotone in M.** Several DIMs dip at M=48 then recover (seed / geometry);
  do not assume deeper is always better without checking the table.
- **Small DIM saturates.** DIM 5–6 sit near MC/F ≈ 1; larger DIM has more headroom
  but lower utilization at this op-point.

---

## TotalMC grids (M × sr)

Leak = 1.0, is = 0.06, seed = 47397376. Bold = best cell in that DIM table.

### DIM 5 · N = 32

| M \ sr | 0.90 | 0.95 | 1.00 |
|-------:|-----:|-----:|-----:|
| 1 | 18.83 | 19.38 | 18.70 |
| 2 | 22.34 | 24.29 | 22.26 |
| 4 | 29.47 | 30.08 | 27.52 |
| 8 | 29.75 | 29.82 | 27.95 |
| 16 | 29.92 | 29.83 | 28.53 |
| 32 | 29.76 | 29.95 | 26.55 |
| 40 | 29.75 | 30.05 | 29.11 |
| 48 | 28.85 | **30.09** | 24.88 |
| 56 | 25.98 | 29.99 | 25.44 |
| 64 | 23.86 | 29.91 | 26.66 |

### DIM 6 · N = 64

| M \ sr | 0.90 | 0.95 | 1.00 |
|-------:|-----:|-----:|-----:|
| 1 | 20.96 | 23.23 | 23.15 |
| 2 | 27.52 | 31.12 | 33.20 |
| 4 | 37.67 | 45.34 | 48.14 |
| 8 | 39.36 | 49.64 | 58.70 |
| 16 | 49.27 | 61.55 | 59.60 |
| 32 | 50.93 | **61.76** | 59.46 |
| 40 | 44.89 | 61.55 | 59.71 |
| 48 | 45.75 | 61.59 | 42.70 |
| 56 | 47.00 | 61.65 | 59.55 |
| 64 | 40.90 | 61.54 | 40.24 |

### DIM 7 · N = 128

| M \ sr | 0.90 | 0.95 | 1.00 |
|-------:|-----:|-----:|-----:|
| 1 | 22.34 | 24.24 | 24.65 |
| 2 | 35.07 | 41.39 | 45.10 |
| 4 | 49.91 | 67.92 | 78.83 |
| 8 | 55.12 | 81.98 | 93.46 |
| 16 | 59.29 | 90.48 | 122.91 |
| 32 | 61.33 | 104.08 | 121.21 |
| 40 | 53.68 | 103.13 | 122.34 |
| 48 | 47.86 | 105.96 | 108.56 |
| 56 | 55.34 | 106.68 | 124.62 |
| 64 | 54.62 | 106.53 | **124.64** |

### DIM 8 · N = 256

| M \ sr | 0.90 | 0.95 | 1.00 |
|-------:|-----:|-----:|-----:|
| 1 | 30.04 | 34.05 | 35.74 |
| 2 | 39.37 | 49.70 | 59.37 |
| 4 | 46.92 | 63.31 | 86.76 |
| 8 | 60.28 | 91.28 | 158.25 |
| 16 | 66.58 | 108.63 | 233.33 |
| 32 | 63.71 | 121.43 | 243.35 |
| 40 | 68.91 | 123.07 | **249.44** |
| 48 | 51.03 | 130.70 | 168.82 |
| 56 | 55.92 | 127.95 | 248.86 |
| 64 | 61.47 | 125.86 | 169.56 |

### DIM 9 · N = 512

| M \ sr | 0.90 | 0.95 | 1.00 |
|-------:|-----:|-----:|-----:|
| 1 | 37.33 | 43.69 | 45.16 |
| 2 | 43.28 | 54.38 | 65.15 |
| 4 | 51.68 | 69.96 | 95.52 |
| 8 | 65.89 | 110.94 | 213.56 |
| 16 | 69.96 | 127.05 | 324.50 |
| 32 | 67.91 | 135.18 | 477.04 |
| 40 | 77.10 | 149.55 | 481.27 |
| 48 | 49.62 | 147.02 | 309.77 |
| 56 | 56.01 | 143.77 | 424.23 |
| 64 | 63.76 | 132.22 | **495.00** |

### DIM 10 · N = 1024

| M \ sr | 0.90 | 0.95 | 1.00 |
|-------:|-----:|-----:|-----:|
| 1 | 35.93 | 42.98 | 49.36 |
| 2 | 50.08 | 63.46 | 78.43 |
| 4 | 43.39 | 54.40 | 72.57 |
| 8 | 71.58 | 117.84 | 225.87 |
| 16 | 75.09 | 122.94 | 285.13 |
| 32 | 74.37 | 157.72 | 623.22 |
| 40 | 78.99 | 149.15 | **819.29** |
| 48 | 73.12 | 158.69 | 554.54 |
| 56 | 56.26 | 156.23 | 725.92 |
| 64 | 63.99 | 149.58 | 566.92 |

### DIM 11 · N = 2048

| M \ sr | 0.90 | 0.95 | 1.00 |
|-------:|-----:|-----:|-----:|
| 1 | 37.06 | 45.57 | 54.74 |
| 2 | 48.96 | 62.63 | 78.93 |
| 4 | 55.85 | 74.90 | 108.23 |
| 8 | 69.78 | 106.15 | 207.41 |
| 16 | 75.40 | 119.18 | 257.84 |
| 32 | 79.88 | 150.05 | 779.48 |
| 40 | 79.65 | 149.93 | 591.12 |
| 48 | 74.03 | 164.95 | 904.21 |
| 56 | 57.34 | 162.10 | 954.07 |
| 64 | 64.00 | 161.70 | **993.28** |

### DIM 12 · N = 4096

| M \ sr | 0.90 | 0.95 | 1.00 |
|-------:|-----:|-----:|-----:|
| 1 | 36.65 | 44.54 | 54.32 |
| 2 | 46.45 | 59.81 | 79.88 |
| 4 | 66.04 | 98.97 | 145.26 |
| 8 | 73.74 | 117.07 | 230.21 |
| 16 | 75.38 | 123.31 | 322.10 |
| 32 | 85.52 | 144.80 | 514.89 |
| 40 | 80.25 | 165.03 | 1045.04 |
| 48 | 88.98 | 175.15 | 1346.06 |
| 56 | 59.50 | 170.18 | 1169.33 |
| 64 | 64.00 | 171.42 | **1383.41** |

---

## How to re-run

1. Edit `MemoryCapacity.cpp` `main()`:
   - **Meter:** `mccfg.k_max`, `t_warmup` (≥ k_max), `t_collect` (> k_max).
   - **Base op-point:** `DIM`, `seed`, `input_scaling`, FSF knobs.
   - **Mode:** uncomment exactly one of `RunDetailed` / `RunGridSweep` /
     `RunSeedSurvey` / `RunDepthProbe`.
   - **Grid axes:** `sr`, `leak`, `history_depth` vectors (`M` any in **[1, 64]**).
2. Build Release (CLion owns `cmake-build-release`; do not reconfigure generators).
3. Run:

```
cmake-build-release\MemoryCapacity.exe
```

Progress on stderr; tables on stdout. For seed variance at fixed (DIM, M, sr):
`RunSeedSurvey`. For lag-shape diagnostics: `RunDepthProbe`.

---

## Appendix A — DIM 10 leak × sr × M

Same meter and seed family as the main tables (FSF off, is = 0.06, seed =
47397376, warmup 4000 / collect 25000 / Kmax 4000 / ridge 1e-4). **DIM = 10**
only (`N = F = 1024`). Grid: 3 sr × 5 leak × 10 M = **150 cells**; all
early-stopped (`e`), none open-tail.

| Axis | Values |
|------|--------|
| sr | 0.90 · 0.95 · 1.00 |
| leak | 0.60 · 0.70 · 0.80 · 0.90 · **1.00** |
| M | 1, 2, 4, 8, 16, 32, 40, 48, 56, 64 |

The **leak = 1.00** column matches the DIM 10 main-grid row for each M (that
campaign held leak fixed at 1.0). Bold = best cell **within that M block**.

**Reading tip.** At shallow M, higher leak almost always wins. At deep M and
sr = 1.00, mid/high leak still dominates through M = 40, but at M = 48 and
M = 64 the peak shifts off leak = 1.00 (leak 0.90 beats 1.00). Do not assume
leak = 1 is optimal for every depth.

### Peak by M (DIM 10)

| M | Peak TotalMC | at sr | at leak |
|--:|-------------:|------:|--------:|
| 1 | 49.36 | 1.00 | 1.00 |
| 2 | 78.43 | 1.00 | 1.00 |
| 4 | 72.57 | 1.00 | 1.00 |
| 8 | 225.87 | 1.00 | 1.00 |
| 16 | 285.13 | 1.00 | 1.00 |
| 32 | 623.22 | 1.00 | 1.00 |
| 40 | **819.29** | 1.00 | 1.00 |
| 48 | 621.07 | 1.00 | 0.90 |
| 56 | 725.92 | 1.00 | 1.00 |
| 64 | 684.51 | 1.00 | 0.90 |

Overall peak for this appendix: **819.29** (M = 40, sr = 1.00, leak = 1.00).

### M = 1

| sr \ leak | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
|----------:|-----:|-----:|-----:|-----:|-----:|
| 0.90 | 17.19 | 19.31 | 22.31 | 27.21 | 35.93 |
| 0.95 | 19.67 | 22.19 | 25.94 | 32.13 | 42.98 |
| 1.00 | 22.76 | 25.81 | 30.48 | 37.99 | **49.36** |

### M = 2

| sr \ leak | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
|----------:|-----:|-----:|-----:|-----:|-----:|
| 0.90 | 23.79 | 27.36 | 31.90 | 38.60 | 50.08 |
| 0.95 | 27.89 | 32.25 | 38.06 | 46.98 | 63.46 |
| 1.00 | 32.18 | 37.72 | 45.30 | 57.16 | **78.43** |

### M = 4

| sr \ leak | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
|----------:|-----:|-----:|-----:|-----:|-----:|
| 0.90 | 24.73 | 27.78 | 31.41 | 36.20 | 43.39 |
| 0.95 | 29.36 | 33.34 | 38.12 | 44.62 | 54.40 |
| 1.00 | 35.34 | 40.70 | 47.67 | 57.44 | **72.57** |

### M = 8

| sr \ leak | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
|----------:|-----:|-----:|-----:|-----:|-----:|
| 0.90 | 41.34 | 46.53 | 52.49 | 60.27 | 71.58 |
| 0.95 | 57.77 | 66.93 | 78.24 | 93.57 | 117.84 |
| 1.00 | 85.27 | 102.57 | 125.73 | 161.05 | **225.87** |

### M = 16

| sr \ leak | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
|----------:|-----:|-----:|-----:|-----:|-----:|
| 0.90 | 50.30 | 55.29 | 60.61 | 67.32 | 75.09 |
| 0.95 | 72.30 | 81.21 | 91.39 | 104.56 | 122.94 |
| 1.00 | 123.51 | 146.46 | 176.01 | 217.22 | **285.13** |

### M = 32

| sr \ leak | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
|----------:|-----:|-----:|-----:|-----:|-----:|
| 0.90 | 54.38 | 60.60 | 64.77 | 68.38 | 74.37 |
| 0.95 | 104.00 | 114.84 | 126.79 | 141.16 | 157.72 |
| 1.00 | 256.39 | 307.32 | 373.77 | 467.69 | **623.22** |

### M = 40

| sr \ leak | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
|----------:|-----:|-----:|-----:|-----:|-----:|
| 0.90 | 54.01 | 60.34 | 68.64 | 75.99 | 78.99 |
| 0.95 | 103.66 | 114.15 | 125.15 | 137.11 | 149.15 |
| 1.00 | 351.21 | 421.86 | 511.22 | 639.40 | **819.29** |

### M = 48

| sr \ leak | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
|----------:|-----:|-----:|-----:|-----:|-----:|
| 0.90 | 50.33 | 52.26 | 56.02 | 63.20 | 73.12 |
| 0.95 | 112.95 | 122.37 | 134.20 | 146.26 | 158.69 |
| 1.00 | 382.93 | 452.86 | 518.02 | **621.07** | 554.54 |

### M = 56

| sr \ leak | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
|----------:|-----:|-----:|-----:|-----:|-----:|
| 0.90 | 55.38 | 55.90 | 56.01 | 56.07 | 56.26 |
| 0.95 | 114.66 | 123.59 | 132.98 | 145.97 | 156.23 |
| 1.00 | 390.28 | 467.13 | 563.69 | 694.10 | **725.92** |

### M = 64

| sr \ leak | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
|----------:|-----:|-----:|-----:|-----:|-----:|
| 0.90 | 58.86 | 62.58 | 63.70 | 63.95 | 63.99 |
| 0.95 | 110.19 | 122.52 | 130.59 | 137.98 | 149.58 |
| 1.00 | 499.15 | 579.49 | 625.09 | **684.51** | 566.92 |

---

## Appendix B — DIM 10 seed survey (M = 30–34)

Seed-to-seed variance at **DIM = 10** (`N = F = 1024`), **sr = 1.00**,
**is = 0.06**, FSF off. Fine M grid around the deep-memory knee (not the coarse
pow2 / M∈{40,48,…} grid of the main tables). Leak near full (0.95 · 0.98 · 1.00).
Ten reservoir seeds of the form `73896+k` × `(k+2)` for k = 0…9 (same family as
the NARMA multi-seed pools). All cells early-stopped.

| Axis | Values |
|------|--------|
| DIM | 10 |
| sr | **1.00** (fixed) |
| leak | 0.95 · 0.98 · 1.00 |
| M | 30 · 31 · 32 · 33 · 34 |
| Seeds | 10 (table below) |

**Note from the run log:** “M = 30 appears to be the best choice” as a coarse
pick in this band. Per-seed peaks (below) actually land most often at **M = 31
or 33**, not 30 — M = 30 is solid but rarely the max. Prefer the peak-by-seed
table when locking an op-point.

**Qual labels** (from the source log, per seed overall): GREAT · GOOD · POOR.

### Peak by seed (best cell in that seed’s 5 × 3 grid)

| Seed expr | Seed | Qual | Peak TotalMC | at M | at leak |
|-----------|-----:|:----:|-------------:|-----:|--------:|
| 73896×2 | 147792 | GREAT | 696.23 | 33 | 1.00 |
| 73897×3 | 221691 | GREAT | **785.23** | 31 | 1.00 |
| 73898×4 | 295592 | GREAT | 674.80 | 33 | 0.98 |
| 73899×5 | 369495 | GOOD | 626.32 | 33 | 1.00 |
| 73900×6 | 443400 | GREAT | 729.77 | 33 | 1.00 |
| 73901×7 | 517307 | GREAT | 739.65 | 34 | 1.00 |
| 73902×8 | 591216 | GOOD | 651.26 | 34 | 0.98 |
| 73903×9 | 665127 | GOOD | 669.60 | 31 | 1.00 |
| 73904×10 | 739040 | POOR | 705.36 | 34 | 1.00 |
| 73905×11 | 812955 | GREAT | 774.68 | 33 | 1.00 |

Across these ten seeds at sr = 1.00: best peak **785.23** (seed 221691, M = 31,
leak = 1.00); weakest peak **626.32** (seed 369495, M = 33, leak = 1.00). Spread
is large — single-seed main tables can sit anywhere in this band.

### Per-seed grids (M × leak)

#### Seed 147792 (73896×2) · GREAT

| M \ leak | 0.95 | 0.98 | 1.00 |
|---------:|-----:|-----:|-----:|
| 30 | 509.84 | 548.35 | 577.39 |
| 31 | 598.72 | 643.71 | 693.75 |
| 32 | 428.55 | 456.98 | 488.60 |
| 33 | 595.63 | 642.84 | **696.23** |
| 34 | 543.15 | 584.15 | 631.16 |

#### Seed 221691 (73897×3) · GREAT

| M \ leak | 0.95 | 0.98 | 1.00 |
|---------:|-----:|-----:|-----:|
| 30 | 518.46 | 561.16 | 609.22 |
| 31 | 660.24 | 718.82 | **785.23** |
| 32 | 558.88 | 603.52 | 651.99 |
| 33 | 692.71 | 737.69 | 768.18 |
| 34 | 656.01 | 688.22 | 728.18 |

#### Seed 295592 (73898×4) · GREAT

| M \ leak | 0.95 | 0.98 | 1.00 |
|---------:|-----:|-----:|-----:|
| 30 | 513.05 | 552.79 | 600.44 |
| 31 | 528.76 | 570.75 | 616.54 |
| 32 | 484.33 | 519.68 | 559.73 |
| 33 | 625.69 | **674.80** | 672.66 |
| 34 | 420.22 | 447.81 | 478.01 |

#### Seed 369495 (73899×5) · GOOD

| M \ leak | 0.95 | 0.98 | 1.00 |
|---------:|-----:|-----:|-----:|
| 30 | 421.28 | 449.79 | 482.28 |
| 31 | 451.24 | 482.46 | 517.71 |
| 32 | 487.91 | 524.02 | 564.49 |
| 33 | 591.88 | 612.08 | **626.32** |
| 34 | 448.44 | 479.10 | 513.29 |

#### Seed 443400 (73900×6) · GREAT

| M \ leak | 0.95 | 0.98 | 1.00 |
|---------:|-----:|-----:|-----:|
| 30 | 481.92 | 518.48 | 561.72 |
| 31 | 579.59 | 631.88 | 670.04 |
| 32 | 498.17 | 529.99 | 569.55 |
| 33 | 645.39 | 699.61 | **729.77** |
| 34 | 514.84 | 552.35 | 594.85 |

#### Seed 517307 (73901×7) · GREAT

| M \ leak | 0.95 | 0.98 | 1.00 |
|---------:|-----:|-----:|-----:|
| 30 | 440.41 | 471.85 | 506.00 |
| 31 | 487.02 | 522.36 | 563.10 |
| 32 | 409.53 | 436.22 | 466.08 |
| 33 | 562.76 | 605.00 | 653.64 |
| 34 | 649.97 | 705.47 | **739.65** |

#### Seed 591216 (73902×8) · GOOD

| M \ leak | 0.95 | 0.98 | 1.00 |
|---------:|-----:|-----:|-----:|
| 30 | 448.49 | 480.42 | 516.30 |
| 31 | 534.89 | 577.57 | 626.91 |
| 32 | 472.44 | 505.54 | 542.95 |
| 33 | 529.31 | 568.39 | 613.68 |
| 34 | 603.37 | **651.26** | 621.68 |

#### Seed 665127 (73903×9) · GOOD

| M \ leak | 0.95 | 0.98 | 1.00 |
|---------:|-----:|-----:|-----:|
| 30 | 484.74 | 521.33 | 563.15 |
| 31 | 566.82 | 614.76 | **669.60** |
| 32 | 547.33 | 590.99 | 640.40 |
| 33 | 517.78 | 557.72 | 601.61 |
| 34 | 471.61 | 504.43 | 538.95 |

#### Seed 739040 (73904×10) · POOR

| M \ leak | 0.95 | 0.98 | 1.00 |
|---------:|-----:|-----:|-----:|
| 30 | 558.47 | 604.24 | 631.65 |
| 31 | 531.34 | 572.91 | 621.37 |
| 32 | 466.55 | 499.60 | 534.09 |
| 33 | 405.44 | 431.58 | 460.34 |
| 34 | 598.88 | 647.45 | **705.36** |

#### Seed 812955 (73905×11) · GREAT

| M \ leak | 0.95 | 0.98 | 1.00 |
|---------:|-----:|-----:|-----:|
| 30 | 471.60 | 506.17 | 546.15 |
| 31 | 582.27 | 630.47 | 665.95 |
| 32 | 507.02 | 545.62 | 590.02 |
| 33 | 680.79 | 740.67 | **774.68** |
| 34 | 426.21 | 454.90 | 485.76 |
