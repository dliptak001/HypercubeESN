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
survey is re-run under the current protocol.

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
 LorenzDatastream  — integrate once, midpoint-offset + shared-scale → float S[·] ≈ [-1,1]
        │            inherits Cursor (forward index over train window)
        │
        │  input port (4): [x, y, z, x·z]  real in train/washout; prediction in free-run
        ▼
 ESN  — fixed hypercube reservoir + online HCNN readout (3 outputs: x, y, z)
        │  external feedback: off
        ├─ Train()   multi-epoch teacher-forced sweeps (horizon-1, prequential)
        └─ FreeRun() washout → generative self-feedback; VPT + RMSE + re-lock proxies
```

---

## 3. Forward cursor

System-agnostic index walker (`Cursor.h`). Lorenz only maps indices into the
normalized stream. Details: [`Cursor.md`](Cursor.md).

```text
stream:  0 = lb ======================= ub ....... end
              training / washout           free-run runway
```

| Signal | Meaning |
|--------|---------|
| **`Reset()`** | Index = train start (`lb`) |
| **`Step()`** | Index += 1 |
| **`OOB()`** | Index past train upper edge — generative / eval region |

Default layout (`config::`): train span = `TRAINING_WINDOW_SIZE` from index 0;
`STREAM_LENGTH = TRAINING_WINDOW_SIZE + FREE_RUN_WINDOW_SIZE`.

---

## 4. Drive and targets

```text
 input port (4):  [ x, y, z, x·z ]   ExtractDriveReal / ExtractDrivePredicted
 targets    (3):  (x, y, z)          ExtractTargets
```

- Fourth channel is the product of that step's `x` and `z` (after normalization).
- Free-run rebuilds the product from predicted channels — no denorm/renorm bug.
- Gain: `input_scaling` only.

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

1. **Washout** — teacher-forced sweep of the training window (no readout updates).
2. **Generative** — for up to `FREE_RUN_WINDOW_SIZE` steps:
   - `Predict` → pack prediction as input drive → `ReservoirStep(drive, nullptr)`
   - Score vs true `S[index]` (normalized channel-RMS)
   - VPT, free-run RMSE, duty / n_relock / n_unlock / meanLock (θ = `VPT_THRESHOLD`)

| Field | Meaning |
|-------|---------|
| `vpt_steps` / `vpt_lt` | First step with channel-RMS `err > θ` (Lyapunov times) |
| `rmse` | Free-run RMSE over scored steps |
| `duty` | Fraction of steps with `err ≤ θ` |
| `n_relock` | Unlocked→locked after a prior unlock |

---

## 7. Configuration (`config::` in `Lorenz.h`)

| Group | Controls |
|-------|----------|
| Diagnostics | `ENABLE_PRINTF` |
| Reservoir | dim, seed, SR, `INPUT_SCALING`, leak, history depth |
| Readout | online Adam schedule, epochs, slices, pooling |
| Stream | train span, free-run window, stream length, dt |
| Score | VPT threshold, Lyapunov exponent |

No runtime config file — edit constants and rebuild.

---

## 8. Building and running

```text
cmake --build <build-dir> --target Lorenz
```

Release preferred. MinGW on `PATH` when running the exe:

```text
Lorenz.exe [NUM_THREADS] [NUM_RUNS]
```

| Arg | Meaning |
|-----|---------|
| `NUM_THREADS` | Parallel ESN-seed trials (default: hardware concurrency) |
| `NUM_RUNS` | Free-runs per trial after training (default: 50) |

Progress on **stderr**; final report on stdout. Optional:

```text
Lorenz.exe --trace <esn_seed> [max_freeruns] [target_orbit_seed]
```

writes per-step CSV under `examples/Lorenz/traces/`.

---

## 9. Related

| Doc | Role |
|-----|------|
| [`Cursor.md`](Cursor.md) | Forward cursor API |
| [`docs/ReservoirFeedbackMechanism.md`](../../docs/ReservoirFeedbackMechanism.md) | Optional external-feedback port (not used here) |
| [`docs/LorenzFreeRun.md`](../../docs/LorenzFreeRun.md) | Historical A(x) vs tanh campaign (different harness) |
