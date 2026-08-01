# Lorenz — free-run on Lorenz-63

Closed-loop free-run: train a HypercubeESN online on a Lorenz-63 orbit, then
generate with self-feedback on the **input bank** (external feedback off).

Stack:

```text
Cursor  →  LorenzAttractor  →  LorenzDatastream  →  Lorenz (ESN train / free-run)
```

> **Note:** This example was reworked from a dual-cursor (half-anchored) harness
> to a single forward cursor. Treat storefront numbers as provisional until a
> fresh survey lands.

---

## 1. Literature context — vanilla ESN free-run on Lorenz-63

Ballpark for **standard (vanilla) Echo State Network** free-run on Lorenz-63,
scored versus Lyapunov time. Unassisted closed loop after teacher-forced
training. Do not quote these ranges as HypercubeESN results until the example
survey is re-run with the same free-run seating.

| Class | Valid prediction horizon (Lyapunov times) |
|-------|-------------------------------------------|
| Conventional / baseline ESNs | **~4–8 LT** |
| Well-tuned (N = 100–500, careful SR / scaling) | **~10–15 LT** |
| Extreme optimized / noiseless claims | **>30 LT** (definition- and solver-sensitive) |

**Papers:** [Doan et al.](https://arxiv.org/abs/1906.11122) · [Hurley et al.](https://arxiv.org/abs/2508.06730) · local PDFs under [`reference/`](reference/)

---

## 2. Pipeline

```text
 LorenzAttractor (RK4 Lorenz-63; σ, ρ, β, dt as configured)
        │
        ▼
 LorenzDatastream  — integrate once, normalize → float S[·] ≈ [-1,1]; is a Cursor
        │
        │  input port: fixed 4-in [x, y, z, x·z]
        │    real in train/warmup; prediction in free-run
        ▼
 ESN  — fixed hypercube reservoir + online HCNN readout (3 outputs: x, y, z)
        │  external feedback: off
        ├─ Train()   multi-epoch teacher-forced sweeps (horizon-1, prequential)
        └─ FreeRun() warmup → generative self-feedback; VPT + RMSE + re-lock proxies
```

---

## 3. Forward cursor

System-agnostic index walker (`Cursor.h`). Lorenz only maps indices into the
normalized stream. Details: [`Cursor.md`](Cursor.md).

```text
stream:  0 ======================= span ....... end
              train section             free-run runway
```

| Signal | Meaning |
|--------|---------|
| **`Reset()`** | Index = 0 |
| **`Step()`** | Index += 1 |
| **`OOB()`** | `index > span` — generative / eval region |

Default layout (`config::`): train window `[0, TRAINING_WINDOW_SIZE]`;
`STREAM_LENGTH = TRAINING_WINDOW_SIZE + FREE_RUN_WINDOW_SIZE`.

---

## 4. Drive and targets

Fixed drive (`kNumDriveChannels = 4` in `Lorenz.h`): **`[x, y, z, x·z]`**
(ODE bilinear in y-dot). Must divide `N = 2^DIM` (always true for legal DIM).

```text
 drive (4):    (x, y, z, x*z)   FillDrive + INPUT_SCALE_CH[4]
 targets (3):  (x, y, z)        ExtractTargets
```

- Products use the same-step `(x,y,z)` (normalized). Free-run rebuilds `x*z` from predictions.
- Global `INPUT_SCALING` (reservoir) plus locked per-channel `INPUT_SCALE_CH[]`
  (`constexpr` soft z/xz: `{1, 1, 0.9, 0.7}`). Applied in `FillDrive` train + free-run.
- Load/save readout must match the layout **and** channel gains used at train time.

---

## 5. Training (`Lorenz::Train`)

Per epoch:

1. **`RebuildDatastream`** — new orbit; `Reset()` cursor.
2. **Warmup** — teacher-forced `ReservoirStep(drive, nullptr)`; no readout update.
3. **Train sweep** while `!OOB()`:
   - Predict at current reservoir state **before** inject; target = current sample.
   - Prequential train RMSE; `TrainStep` → `ReservoirStep(drive, nullptr)` → `Step()`.

---

## 6. Free-run (`Lorenz::FreeRun`)

Single multi-IC challenge path (no train-orbit replay modes):

| | Orbit | Warmup | Generative scores |
|--|--------|--------|-------------------|
| Free-run | New IC (remix), or fixed orbit seed / fixed attractor IC | Last W of train on that orbit | From `span+1` (eval runway) |

Warmup length: `WARMUP_STEPS` (shared with train; override free-run via `warmup_steps`;
`0` = default). Teacher-forced open-loop on the edge of the train section, then
generative scoring past span.

Generative loop: for up to `FREE_RUN_WINDOW_SIZE` steps —

- `Predict` → pack prediction as input drive → `ReservoirStep(drive, nullptr)`
- Score vs true `S[index]` (normalized channel-RMS)
- Primary metrics: **VPT**, **duty**, **VPT×duty**, **RMSE** (θ = `VPT_THRESHOLD`)
- Per-step `locked` is still written to trace CSVs for plots (not aggregated)

**Claim discipline:** free-run numbers are multi-IC generalization (remix or fixed
IC/orbit), not in-sample train-window generative score.

| Field | Meaning |
|-------|---------|
| `vpt_steps` / `vpt_lt` | First step with channel-RMS `err > θ` (Lyapunov times) |
| `duty` | Fraction of steps with `err ≤ θ` |
| `vpt_x_duty` | `VPT_lt × duty` — first-hold length scaled by time-in-lock |
| `rmse` | Free-run RMSE over scored steps |

### Aggregate freerun stats (top 10% of ICs)

Across freeruns with different initial conditions, the four primary metrics keep
only the **top 10%** of values **per metric** (`keep = max(1, ceil(n/10))`;
e.g. 1000 freeruns → 100). The rest are **discarded on purpose**.

| Direction | Metrics |
|-----------|---------|
| Higher is better (keep largest 10%) | VPT, duty, VPT×duty |
| Lower is better (keep smallest 10%) | free-run RMSE |

Rationale: some Lorenz ICs are hard for any autonomous model; mean-of-all is
dominated by those weak orbits. Storefront and campaign numbers answer
**how well the system can perform on the better starts**, not how poorly the
worst orbits do. With 1000 freeruns, top 10% is still a solid sample. Filtering
is **independent per metric**.

Survey roll-ups are **mean of trial-means**, where each trial mean already used
that trial’s top-10% freeruns. CSV metadata records
`freerun_metrics=VPT,duty,VPT*duty,RMSE` and
`freerun_pool=top_10pct_per_metric`. Always state this when quoting aggregates.

---

## 7. Configuration (`config::` in `Lorenz.h`)

| Group | Controls |
|-------|----------|
| Diagnostics | `ENABLE_PRINTF` (verbose); `ENABLE_PROGRESS` (stderr heartbeats; off for quiet overnight) |
| Reservoir | dim, seed, `SPECTRAL_RADIUS` (reassignable), `INPUT_SCALING`, `INPUT_SCALE_CH[]` (constexpr), leak, history depth |
| Readout | online Adam schedule, epochs, slices, pooling |
| Stream | train span + freerun runway for **Train**; free-run stores only wash + runway (burn-in discarded) |
| Stage | `WARMUP_STEPS` (train and free-run) |
| Free-run | Edge warmup (`WARMUP_STEPS`) then `FREE_RUN_WINDOW_SIZE` past span |
| Export | `SAVE_TRAINED_WEIGHTS` (off) → `lorenz_seed{S}_D{DIM}_M{M}_in{Nin}` HCNW + arch |
| Load | `LOAD_TRAINED_WEIGHTS` (off) + `LOAD_WEIGHTS_STEM` — skip train, free-run only |
| Score | VPT threshold, Lyapunov exponent |

No runtime config file — edit constants and rebuild.

---

## 8. Building and running

```text
cmake --build <build-dir> --target Lorenz
```

Release preferred. MinGW on `PATH` when running the exe. **No CLI knobs** —
edit `main.cpp` to call a campaign, rebuild, run `Lorenz.exe`.

```cpp
// examples/Lorenz/main.cpp — edit stages, rebuild, run Lorenz.exe
// Pipeline: SeedSweep → Train → OrbitSweep → FreeRun
```

### Campaign pipeline (keepers only)

```text
SeedSweep  →  Train  →  OrbitSweep  →  FreeRun
 multi-seed    save      rank orbits     plot one IC
 search        weights   (load-only)     (load-only)
```

**Artifact tree** (`config::RUNS_DIR` = `C:\HypercubeESN\results`; models under `C:\HypercubeESN\models`):

| Subdir | Campaigns |
|--------|-----------|
| `traces/` | `FreeRun` plottable CSVs |
| `surveys/` | `SeedSweep`, `OrbitSweep` leaderboards |

**Shared reporting:** banners `=== HypercubeESN: Lorenz / Name ===`, tags
`[train]` / `[freerun]` / `[par-seed-sweep]` / `[par-orbit-sweep]`, freerun
scores as `VPT / duty / VPT*duty / RMSE`, and `[tag] wrote path (bytes)` +
`[tag] done wall time: …`.

**`SeedSweep`** — overnight parallel seed search. Always trains in
memory (**no weight save/load**; refuses `SAVE_TRAINED_WEIGHTS` /
`LOAD_TRAINED_WEIGHTS`). Requires HCNN `Lorenz::kReadoutNumThreads == 1`.
ESN seeds from `base_esn_seed` via SplitMix64. Top-10% freerun pool means;
report under `RUNS_DIR/surveys/par_seed_sweep_*.csv|txt`. Cherry-pick with
Train / OrbitSweep / FreeRun.

**`Train`** — train-only: remixed orbits for `epochs`, save to `weights_stem`
or `DefaultWeightStem(esn, dim, M)`.

**`OrbitSweep`** — load-only; parallel one freerun per Mix64 orbit; TXT report
(top 100 + bottom 10 by VxD; top-k includes IC xyz). Requires existing weights.

**`FreeRun`** — load-only; one attractor IC; plottable CSV under
`RUNS_DIR/traces/seed{esn}_ic{x}_{y}_{z}.csv`. Optional SR/IS after `esn_seed`.

| Function | Role |
|----------|------|
| `SeedSweep(...)` | Parallel multi-seed train+freerun search |
| `Train(...)` | Fit one seed; save weights |
| `OrbitSweep(...)` | Load weights; rank orbits (TXT) |
| `FreeRun(...)` | Load weights; plot one IC |
| `DefaultWeightStem(esn, dim, M)` | Shared weight path under `MODEL_SAVE_DIR` |

First arg is always reservoir **DIM** (`N = 2^DIM`, range 5–16); restored on exit
(along with M / SR when a campaign reassigns them). Protocol and knobs live in
`Lorenz.h` `config::`. Signatures in `Campaigns.h`. Progress on **stderr**;
reports on **stdout**.

---

## 9. Related

| Doc | Role |
|-----|------|
| [`Cursor.md`](Cursor.md) | Forward cursor API |
| [`docs/ReservoirFeedbackMechanism.md`](../../docs/ReservoirFeedbackMechanism.md) | Optional external-feedback port (not used here) |
| [`docs/LorenzFreeRun.md`](../../docs/LorenzFreeRun.md) | Historical A(x) vs tanh campaign (different harness) |
