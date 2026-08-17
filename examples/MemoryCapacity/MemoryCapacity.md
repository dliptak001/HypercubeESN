# Memory Capacity — Linear Short-Term Memory Diagnostic

Jaeger (2001) **linear short-term memory** probe for HypercubeESN. Unlike NARMA
(memory + nonlinear mixing), MC isolates memory only: ridge on reservoir state,
no HCNN / ESN coupling.

**Full M × sr grids, leak tables, and seed surveys:** [MemoryCapacity_grids.md](MemoryCapacity_grids.md)
(archived reference campaign — not auto-synced to `main()`).

Storefront summary (tunable TotalMC ~30 → 1400+) lives in the project /
[Python README](../../python/README.md) headline section.

---

## What this measures

Drive with white noise `u(t) ~ U[-1,+1]`; for each lag `k` fit a ridge readout to
`u(t-k)` from state `x(t)` (newest delay-line slice / `Outputs()`). Squared
**Pearson** correlation `r²(k)` on **held-out** rows is the retained lag-`k`
fraction; the headline is

```
MC = Σ_k r²(k)     (= TotalMC over scored lags)
```

Theoretical ceiling is **F** (feature count), not “unlimited”: full-state MC uses
F = N = 2<sup>dim</sup> (unless `feature_cap` truncates).

| Symbol | Meaning |
|--------|---------|
| **N** | Reservoir size: N = 2<sup>dim</sup> (dim in **[5, 16]**). |
| **F** | MC features: `F = min(N, feature_cap)` (default cap 8192). Full-state → **F = N**. Ceiling for TotalMC is **F**. |
| **M** (tables) | `history_depth` — delay-line depth **[1, 64]**. |
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

## At a glance — peak TotalMC by dim

Max cell in each archived dim grid ([full tables](MemoryCapacity_grids.md)). Prefer
**sr = 1.00** when memory length matters; contractive radii plateau earlier.
**Single-seed (indicative)** — seed **47397376**, is = 0.06, leak = 1.0, extended
meter (warmup 4000 / collect 25000 / Kmax 4000).

| dim | N = F | Peak TotalMC | at M | at sr | MC/F |
|----:|------:|-------------:|-----:|------:|-----:|
| 5 | 32 | 30.09 | 48 | 0.95 | 0.94 |
| 6 | 64 | 61.76 | 32 | 0.95 | 0.97 |
| 7 | 128 | **124.64** | 64 | 1.00 | 0.97 |
| 8 | 256 | **249.44** | 40 | 1.00 | 0.97 |
| 9 | 512 | **495.00** | 64 | 1.00 | 0.97 |
| 10 | 1024 | **819.29** | 40 | 1.00 | 0.80 |
| 11 | 2048 | **993.28** | 64 | 1.00 | 0.48 |
| 12 | 4096 | **1383.41** | 64 | 1.00 | 0.34 |

**Takeaways**

- **Tunable range:** peak TotalMC from about **30** (dim 5) to **1400+** (dim 12)
  by changing dim, M, and spectral radius — the storefront “~30 → 1400+” line.
- **Depth × radius is super-multiplicative.** At sr = 0.90 capacity often plateaus
  by M ≈ 16–32; at sr = 1.00 it keeps climbing (especially dim ≥ 9).
- **Non-power-of-two M matter.** M ∈ {40, 48, 56} often beat or match nearby
  powers of two (e.g. dim 10: M=40 → 819 vs M=32 → 623 / M=64 → 567 at sr=1.00).
- **Not monotone in M.** Deeper is not always better without checking the table.
- **Small dim saturates** near MC/F ≈ 1; larger dim has more TotalMC headroom but
  lower utilization at this op-point.
- **Seed variance is real** (dim 10 band peaks ~626–785 across 10 seeds — see
  [Appendix B](MemoryCapacity_grids.md#appendix-b--dim-10-seed-survey-m--3034)).

---

## How to run

1. Edit the **`campaign`** block at the top of `MemoryCapacity.cpp`:
   - **`kMode`** — `Detailed` | `GridSweep` | `SeedSurvey` | `DepthProbe`.
   - **Meter** — `MakeMeterConfig()` (`k_max`, `t_warmup` ≥ k_max, `t_collect` > k_max).
   - **Base op-point** — `MakeBaseReservoirConfig()` (`dim`, `seed`, `input_scaling`, …).
   - **Sweep axes** — only the section for the active mode matters:
     - Grid: `GridSpectralRadii` / `GridLeakRates` / `GridHistoryDepths` (M in **[1, 64]**).
     - Seed survey: `kSeedSurveyStart` … `kSeedSurveyEnd`.
     - Depth probe: `kDepthProbeSr` / `Leak` / `Kmax` / `DepthProbeDepths()`.
2. Build Release (CLion owns `cmake-build-release`; do not reconfigure generators).
3. Run:

```text
cmake-build-release\MemoryCapacity.exe
```

Progress on **stderr**; tables on **stdout**. For seed variance at fixed
(dim, M, sr): `RunSeedSurvey`. For lag-shape diagnostics: `RunDepthProbe`.

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
  `leak`, `history_depth`, `seed`, `input_scaling`. `Measure()` forces
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

Concurrent workers self-schedule from an atomic counter; optional RAM budget
caps worker count. Banner prints live meter + layout.

---