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
| Reservoir | dim, seed, `SPECTRAL_RADIUS` (reassignable; SrAB), `INPUT_SCALING`, `INPUT_SCALE_CH[]` (constexpr), leak, history depth |
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
// examples/Lorenz/main.cpp
int main()
{
    // return Campaign_SeedSurvey(/*dim=*/11, /*threads=*/0, /*runs=*/50);
    // return Campaign_Trace(/*dim=*/11, /*seed=*/21978990, /*max_freeruns=*/30);
    // return Campaign_SpectralRadiusAB(/*dim=*/12, /*history_depth=*/12,
    //                                  /*sr_a=*/0.95f, /*sr_b=*/0.99f,
    //                                  /*num_threads=*/0, /*num_runs=*/50);
    // Load weights + freerun one attractor IC (CSV under HypercubeESNRuns):
    // return FreeRun(/*dim=*/12, /*history_depth=*/18, /*esn_seed=*/221978990,
    //                /*ic_x=*/0.43, /*ic_y=*/0.30, /*ic_z=*/0.64);
    // optional stem: ..., R"(C:\HypercubeESN\models\lorenz_seed..._D12_M18_in4)");
}
```

### `Train` / `FreeRunSurvey` / `FreeRun` / `SeedSweep` (campaign pipeline)

```text
Train  →  FreeRunSurvey  →  FreeRun
 weights     rank orbits       plot one IC

SeedSweep: for each esn_seed → Train? + FreeRunSurvey → rank seeds by mean VPT×duty
```

**`Train`** — train-only (no freerun): remixed orbits for `epochs` from remix base
`target_orbit`, then save readout to `weights_stem` (or default under
`MODEL_SAVE_DIR`). Distinct from member `Lorenz::Train`.

```cpp
Train(/*dim=*/12, /*history_depth=*/18, /*esn_seed=*/221978990,
      /*target_orbit=*/9333312947715283458ull, /*epochs=*/400,
      /*weights_stem=*/R"(C:\HypercubeESN\models\lorenz_seed221978990_D12_M18_in4)");
```

**Artifact tree** (`config::RUNS_DIR` = `C:\HypercubeESNRuns\results`):

| Subdir | Campaigns |
|--------|-----------|
| `traces/` | `FreeRun`, `Campaign_Trace` freerun CSVs |
| `surveys/` | `FreeRunSurvey`, `SeedSweep` leaderboards |
| `campaigns/` | `Campaign_SeedSurvey`, M-sweep, SrAB (`RESULTS_DIR`) |

**Shared reporting:** banners `=== HypercubeESN: Lorenz / Name ===`, tags
`[train]` / `[freerun]` / `[freerun-survey]` / `[seed-sweep]` / `[trace]` /
`[survey]` / `[SrAB]`, freerun scores as
`VPT / duty / VPT*duty / RMSE`, and `[tag] wrote path (bytes)` +
`[tag] done wall time: …`.

**`FreeRunSurvey`** — load-only middle step: free-run `num_runs` remixed orbits
(remix from `orbit_seed`), aggregate top-10% VPT / duty / VPT×duty / RMSE, print
**top_k** by VPT×duty with IC triples and a ready-to-paste `FreeRun(...)` line.
Leaderboard: `RUNS_DIR/surveys/survey_seed{S}_D{D}_M{M}_n{N}.csv`.

**`FreeRun`** — load-only (no train): free-run **one** attractor IC for a plottable
trace. Writes: `RUNS_DIR/traces/seed{esn}_ic{x}_{y}_{z}.csv`.

**`SeedSweep`** — loop ESN seeds: optional `Train` then `FreeRunSurvey`; rank by
mean VPT×duty. Stems `{MODEL_SAVE_DIR}/lorenz_seed{S}_D{dim}_M{M}`. Ranking
CSV + `.partial.csv` under `RUNS_DIR/surveys/`. Optional trailing overrides
(restored on exit): `spectral_radius` / `input_scaling` (>0 set, 0 = keep config).
Channel gains are fixed (`config::INPUT_SCALE_CH`). Stems omit SR/IS/drive_ch —
banner records them.

**`ParallelSeedSweep`** — overnight parallel seed search. Always trains in
memory (**no weight save/load**; refuses `SAVE_TRAINED_WEIGHTS` /
`LOAD_TRAINED_WEIGHTS`). Requires HCNN `Lorenz::kReadoutNumThreads == 1`
(no nested pools). ESN seeds from `base_esn_seed` via SplitMix64 substreams
(`Mix64(base ^ FNV*(i+1))`, not `base+i`). One `base_orbit_seed` is the shared
**remix root** for train and freerun (independent remix streams).
`num_threads` workers over `num_seeds` jobs; capped to `hardware_concurrency`
and `num_seeds`. Header + mutexed stderr heartbeats only; final report on
stdout and `RUNS_DIR/surveys/par_seed_sweep_*.csv|txt` with full table plus
top_k by mean VPT, duty, and VPT×duty (top-10% freerun pool). Same dynamics
overrides as SeedSweep.

| Function | Role |
|----------|------|
| `Campaign_SeedSurvey(dim, threads, runs, ...)` | Multi-seed train + free-run report (`threads=0` => HW concurrency) |
| `ParallelSeedSweep(dim, M, base_esn, num_seeds, threads, epochs, freeruns, ...)` | Parallel train+freerun seed search; no lasting weight I/O; multi-metric ranking report |
| `ParallelOrbitSweep(dim, M, esn, base_orbit, num_orbits, threads, epochs, ...)` | Train one seed once; parallel one-freerun-per-orbit ranking |
| `Campaign_Trace(dim, esn_seed, max_freeruns, target_orbit, ...)` | One seed + CSV under `{RESULTS_DIR}/traces/` (absolute; CWD-safe). `target_orbit≠0` = fixed orbit, every step printed + CSV; plot with `plot_freerun_overlay.py` |
| `Campaign_HistoryDepthSweep(dim, {M...}, threads, runs, ...)` | Sequential surveys per M + **code-computed roll-up table** |
| `Campaign_SpectralRadiusAB(dim, M, sr_a, sr_b, threads, runs, ...)` | A/B spectral radius at fixed dim/M; matched seeds; train per arm |

First arg is always reservoir **DIM** (`N = 2^DIM`, range 5–16); restored on exit
(along with M / SR when a campaign reassigns them). Protocol,
epochs, load/save, etc. live in `Lorenz.h` `config::`. Campaign signatures are in
`Campaigns.h`. Progress on **stderr**; reports on **stdout**.

**`Campaign_SpectralRadiusAB`** — two `Campaign_SeedSurvey` arms with
`config::SPECTRAL_RADIUS` set to `sr_a` then `sr_b` (must be finite and > 0).
Matched dim/M/seeds/drive. Refuses if `LOAD_TRAINED_WEIGHTS` is on.
If `SAVE_TRAINED_WEIGHTS` is on, default stems omit SR (arm B can overwrite arm A);
prefer save off for pure A/B, or use distinct stems. Restores DIM, M, and SR on exit.
Example: DIM12 M12, contractive vs house default:

```cpp
Campaign_SpectralRadiusAB(/*dim=*/12, /*history_depth=*/12,
                          /*sr_a=*/0.95f, /*sr_b=*/0.99f,
                          /*num_threads=*/0, /*num_runs=*/50);
```

**Results files** (under `RESULTS_DIR` = `C:\HypercubeESNRuns\results\campaigns\`,
created if needed):

| Job | Files |
|-----|--------|
| Survey | `Survey_YYYYMMDD_HHMMSS_M{M}.csv` + `.txt` (metadata + one aggregate row) |
| M-sweep | `Msweep_YYYYMMDD_HHMMSS.csv` + `.txt` (metadata + all M rows, deltas, code picks) |
| SR A/B | `SrAB_YYYYMMDD_HHMMSS_D{dim}_M{M}.csv` + `.txt` (both arms + deltas B−A + code picks) |

CSV metrics are mean-of-trial-means (code-computed). Metadata comments stamp
dim, N, M, epochs, θ, SR, drive_ch, seeds, etc.

---

## 9. Related

| Doc | Role |
|-----|------|
| [`Cursor.md`](Cursor.md) | Forward cursor API |
| [`docs/ReservoirFeedbackMechanism.md`](../../docs/ReservoirFeedbackMechanism.md) | Optional external-feedback port (not used here) |
| [`docs/LorenzFreeRun.md`](../../docs/LorenzFreeRun.md) | Historical A(x) vs tanh campaign (different harness) |
