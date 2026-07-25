# Memory Capacity — Linear Short-Term Memory Diagnostic

Quick reference for HypercubeESN memory-capacity (MC) grids. Methodology first;
lookup tables below.

---

## What this measures

Memory Capacity (MC) is the Jaeger (2001) probe of a reservoir's **linear
short-term memory**: how many past inputs can be linearly reconstructed from the
present state. Drive with white noise `u(t)`; for each lag `k` fit a linear
readout to `u(t-k)` from `x(t)`. Squared Pearson correlation `r²(k)` is the
retained lag-`k` fraction; the headline is

```
MC = Σ_k r²(k)
```

Unlike NARMA (memory + nonlinear mixing), MC isolates memory. The reservoir is
not trained — only its raw state is read. No HCNN, no ESN coupling.

| Symbol | Meaning |
|--------|---------|
| **N** | Reservoir state size: `N = 2^DIM` hypercube nodes. |
| **F** | Feature count used by the MC readout: `F = min(N, feature_cap)`. Full-state runs (default here) have no cap hit, so **`F = N = 2^DIM`**. Theoretical MC ceiling is `F` (one unit per independent feature). |
| **M** | `history_depth` (delay-line depth). |
| **TotalMC** | `Σ_k r²(k)` over scored lags. Ceiling is `F`. |
| **MC/F** | Capacity utilization: `TotalMC / F`. |
| **k>.5 / .1 / .01** | Last lag with `r²` above that floor (memory horizons). |
| **realSR** | Realized spectral radius after rescale. |

**Tail marks** (live printer; all cells in the tables below closed early-stop):

| Mark | Meaning |
|------|---------|
| `*` | Open tail — `r²` still live at last lag → **TotalMC is a lower bound**. Raise `k_max` / warmup / collect. |
| `e` | Early-stop — curve decayed; sum is complete for that cell. |
| blank | Full `k_max` scored and tail closed. |

---

## Setup

| Piece | Role |
|-------|------|
| `MCConfig` | The **meter**: warmup, collect, `k_max`, ridge, train split, drive seed. Fixed for a sweep. |
| `ReservoirConfig` | The **op-point**: `sr`, leak, `M`, seed, `input_scaling`, FSF. Passed into `Measure()` per cell. |
| `MCLinalg.h` | Double-precision Gram / Cholesky / solve kernels. |
| `MemoryCapacity.h` | `MemoryCapacityMeter`, `RunSweep`. |
| `MemoryCapacity.cpp` | Driver modes + table formatting. Edit `main()` to pick mode / grids. |

**Modes:** `RunDetailed` (one op-point + sparse lag dump) · `RunGridSweep`
(M × sr × leak, prints M×sr pivot when leak is singleton) · `RunSeedSurvey` ·
`RunDepthProbe` (per-lag curves across depths).

One white-noise drive is generated once per meter and reused for every cell, so
differences are reservoir-only.

---

## Protocol (tables below)

Single-seed `RunGridSweep` campaign. **FSF off.** Every cell early-stopped (`e`)
— curves closed inside `Kmax`; no open-tail undercount.

| Knob | Value |
|------|--------|
| DIM | 5 … 12 (one grid each; `N = F = 2^DIM`) |
| Spectral radius | 0.90 · 0.95 · 1.00 |
| Leak | **1.0** (singleton → M × sr pivot) |
| `history_depth` M | 1, 2, 4, 8, 16, 32, 40, 48, 56, 64 |
| `input_scaling` | 0.06 |
| Reservoir seed | 47397376 |
| Meter | warmup **4000** · collect **25000** · Kmax **4000** · ridge **1e-4** · train_frac **0.7** · `input_seed` **0xc0ffee** |
| FSF | **OFF** |

There is seed dependence not represented in these single-seed tables. Treat
peaks as **indicative**, not multi-seed means. Re-run with `RunSeedSurvey` at a
chosen (DIM, M, sr) before locking an op-point for other tasks.

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

Edit `MemoryCapacity.cpp` `main()`:

1. Set `DIM`, meter knobs (`k_max`, warmup, collect), and base op-point
   (`seed`, `input_scaling`, FSF).
2. Enable `RunGridSweep` with the M / sr / leak vectors you want
   (`history_depth` any integer in **[1, 64]** — not restricted to powers of two).
3. Build Release and run `MemoryCapacity.exe`. Progress on stderr; tables on stdout.

```
cmake-build-release\MemoryCapacity.exe
```

For seed variance at a fixed (DIM, M, sr): comment out the grid, enable
`RunSeedSurvey`. For lag-shape diagnostics: `RunDepthProbe`.
