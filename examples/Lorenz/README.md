# Lorenz — Janus half-anchored free-run on Lorenz-63

Closed-loop experiment: train a HypercubeESN online on a Lorenz-63 orbit with a
**Janus dual-cursor** presentation, then free-run with **self-feedback on the
future port** while the past port stays anchored to real history.

This is **assisted / half-anchored** free-run — continuous partial observation
on the past — **not** classical unassisted autonomous generation (no true drive
at all). Report VPT and RMSE with that distinction stated.

**Live knobs** live in `config::` in [`Lorenz.h`](Lorenz.h). This README describes
mechanisms and protocols; it does **not** pin current numerical defaults.
Experiment logs and op-point snapshots go in [`TRACKING.md`](TRACKING.md).

| Layer | Files |
|-------|--------|
| Index motion | [`JanusCursor.h`](JanusCursor.h) · [`JanusCursor.md`](JanusCursor.md) |
| Orbit + normalize | [`LorenzAttractor.h`](LorenzAttractor.h), [`LorenzDatastream.{h,cpp}`](LorenzDatastream.h) |
| Ports, train, free-run, survey | [`Lorenz.{h,cpp}`](Lorenz.h) — all knobs in `config::` |
| Result log | [`TRACKING.md`](TRACKING.md) |
| CMake target | `Lorenz` ← `Lorenz.cpp` + `LorenzDatastream.cpp` |

---

## 1. Pipeline at a glance

```text
 LorenzAttractor (RK4 Lorenz-63; σ, ρ, β, dt as configured)
        │
        ▼
 LorenzDatastream  — integrate once, midpoint-offset + shared-scale → float S[·] ≈ [-1,1]
        │            inherits JanusCursor (past @ ub←, future @ lb→)
        │
        │  input port              (4): past   [x, y, z, x·z]  always real history
        │  external-feedback port  (4): future [x, y, z, x·z]  real in train; prediction in free-run
        ▼
 ESN  — fixed hypercube reservoir + online HCNN readout (3 outputs: x, y, z)
        │
        ├─ Train()   multi-epoch teacher-forced sweeps (horizon-1, prequential)
        └─ FreeRun() washout → generative self-feedback; VPT + free-run RMSE
```

---

## 2. Janus dual-cursor concept

The harness is built around a **pair of counter-moving indices** over one forward
orbit. The index machinery is system-agnostic (`JanusCursor`); Lorenz only maps
those indices into a normalized sample stream.

### What the two cursors are for

| Cursor | Role in the experiment | Motion |
|--------|------------------------|--------|
| **Past** | **Anchor** — always real history on the input port | Starts at `ub`, each `Step()` decrements |
| **Future** | **Horizon** — teacher in train; self-prediction in free-run | Starts at `lb`, each `Step()` increments |

`Reset()` seats them at **opposite edges** of a shared window. Each `Step()` walks
them toward and then past each other (one-way pass; no reflection).

```text
stream:  0 ....... lb =========== center =========== ub ....... N
                    ^future                          ^past       (after Reset)
         future ──────────────────>          <────────────── past  (each Step)
```

### Window geometry (owner-defined)

```text
lb = center − span/2     // integer division
ub = center + span/2
```

| Region | Indices | Role |
|--------|---------|------|
| Past free-run runway | `[0, lb)` | Anchor history the past cursor may enter after leaving the window |
| Training / washout window | `[lb, ub]` | One epoch or washout sweep |
| Prediction / eval runway | `(ub, stream end]` | Future is generative; held-out truth for scoring |

`JanusCursor` does **not** store samples or know stream length `N`. Bounds and
runways are the owner’s job (`LorenzDatastream` validates window vs. stream at
construction). Integer half-span: even `span` → `ub − lb == span`; odd `span` →
`ub − lb == span − 1`.

In this example, `config::` sets:

- `span` ← `TRAINING_WINDOW_SIZE`
- past / future runways ← `FREE_RUN_WINDOW_SIZE` (symmetric layout via
  `CURSOR_CENTER_INDEX` and `STREAM_LENGTH`)
- total integrated length ← `STREAM_LENGTH` (storage is `stream_length + 1`
  samples: seed state + one per RK4 step)

Exact formulas are next to those constants in `Lorenz.h`.

### Public signals (asymmetry is intentional)

| Signal | Meaning |
|--------|---------|
| **`OOB()`** | **Future only:** `future_index > ub` — generative tail has begun |
| Past below `lb` | Allowed; real history for anchoring, **not** a public error |
| Past `< 0` | Owner may throw (`LorenzDatastream::Step` / `States`) |
| **`Distance()`** | `(future − past) / span` — about `−1` at `Reset`, `~0` at center crossing, about `+1` at the mirror extreme (exact ±1 when `span` is even) |
| **`AtStartPosition()`** | Past at `ub` only (proxy for “just Reset”) |

Training and washout loops typically stop when `OOB()` becomes true. Free-run
continues past that edge: future drive switches to the model’s prediction while
truth for scoring is still `S[future_index]` on the eval runway. The past cursor
keeps walking into `[0, lb)` until the anchor runway or step budget ends.

Full API and nested-cursor details: [`JanusCursor.md`](JanusCursor.md).

### Why opposite-ends matter (lag curriculum)

For most of a training pass the past and future samples are **far apart** on the
attractor (up to roughly `span` samples; many Lyapunov times at the ends for
typical windows). They only share phase near the **center crossing**.

| Role | Port | Training | Free-run (generative) |
|------|------|----------|------------------------|
| Teacher of the map | Feedback / future | Real future history → target `S[f]` | Own prediction (closed loop) |
| Stabilizer / tether | Input / past | Present, often long-lag | Always real history |

Consequence:

- The learnable one-step signal is primarily on the **feedback** port
  (`S[f−1] → S[f]` style under teacher forcing).
- The past is a **free-run stabilizer**, not the main training teacher.
- Zeroing **feedback** scaling during train is ill-posed (climatological floor) —
  not a clean “observer floor.” To ablate the anchor, dose **input** scaling
  (or zero the past channels) at free-run time; do not zero feedback during train.

---

## 3. Stream ownership and orbits

```text
 array index n:   0 ······· lb ····· center ····· ub ········· stream end
                  │         │           │           │            │
                 seed   train edge   midpoint   train edge   eval / free-run
                 T=0       (lb)       (center)      (ub)         tail
```

**Integration.** `LorenzAttractor` advances Lorenz-63 with classical RK4
(`dx/dt = σ(y−x)`, etc.). `LorenzDatastream::Build` stores the seed state plus one
sample per step.

**Normalization** (once per orbit, after RK4; storage is `float` ≈ `[-1, 1]`):

1. Scan **this** raw orbit for per-channel min/max (full stream, including eval tail).
2. Per-channel midpoint offset (drops `z`’s large positive DC onto zero).
3. **One shared scale** = max of the three half-ranges (zero scale → `1.0`).
4. Store `(v − c_v) / scale` per channel.

Shared scale preserves relative amplitudes. Free-run scoring is in these
**normalized units**.

**Fresh orbits.** Each `Train` epoch and each `FreeRun` call **rebuilds** a
datastream (`RebuildDatastream`): the orbit seed is remixed (SplitMix-style) and a
new random initial condition is drawn. Successive free-runs in a survey therefore
share a **fixed sequence of held-out orbits** when trials start from the same
`orbit_seed`; the ESN seed is the independent variable across parallel trials.

Note: `config::INITIAL_LORENZ_STATE` is a banner/default-style constant — the
live train/free-run orbits come from the remix path above, not that fixed IC.

**Cursor → samples.** `LorenzDatastream` inherits `JanusCursor`. `States()` /
`Step()` return `{Distance, past sample by value, future* }`. When the future
cursor is OOB, `future` is **`nullptr`** (never an out-of-window address). Callers
that need truth past `ub` (free-run scoring) read `GetDataStream()[future_index]`
directly.

---

## 4. Eight-channel drive and targets

```text
 input port    (4):  past   [ x_p, y_p, z_p, x_p·z_p ]   ExtractPast
 feedback port (4):  future [ x_f, y_f, z_f, x_f·z_f ]   ExtractFutureReal
                                                         or ExtractFuturePredicted
 targets       (3):  future (x, y, z) only                ExtractTargets
```

- Linear channels come from the normalized stream (or the 3-D prediction).
- The 4th channel is always the product of that block’s current `x` and `z`
  (derived feature, never a readout target). Formed **after** normalization so
  free-run rebuild is a multiply of predicted channels — no denorm/renorm product bug.
- Gains are independent: input scaling on past, external-feedback scaling on future
  (separate weight realizations on the two reservoir ports).
- Readout sees delay-line slices only (`readout_slices` blocks of N).

---

## 5. Training protocol (`Lorenz::Train`)

Per epoch:

1. **`RebuildDatastream`** — new orbit; `Reset()` cursors (past@`ub`, future@`lb`).
2. **Warmup** (`RESERVOIR_WARMUP_STEPS`): teacher-forced `ReservoirStep(past, future_real)`
   only; no readout update. Advances cursors into the window (shortens the
   subsequent train sweep if warmup is large relative to span).
3. **Train sweep** while `!OOB()`:
   - **Horizon-1 alignment:** `Predict` at reservoir state `x(t)` **before** injecting
     this step’s drive; target = current future sample `S[f]` (about to be injected),
     not `S[f+1]`.
   - **Prequential** (test-then-train): pre-update prediction vs that target for the
     epoch RMSE line.
   - `TrainStep` → `ReservoirStep(past, future_real)` → cursor `Step()`.
4. Report prequential train RMSE (3 channels × train steps) when printf is enabled.
   LR from `LrProfile` (cosine anneal from `LEARNING_RATE` toward `LEARNING_RATE_MIN`
   by 75% of epochs, then hold).

One epoch = one **one-way** cursor pass. Multi-epoch = outer loop + `Reset` (and a
fresh orbit each epoch).

---

## 6. Free-run protocol (`Lorenz::FreeRun`)

Self-contained for **cursor phase** (readout weights are whatever `Train` left):

### Stage 1 — anchored washout

- `Reset()`; while `!OOB()`: teacher-forced `ReservoirStep(past, future_real)`; no
  readout updates.
- Leaves the reservoir at the window edge, warm and in-distribution.

### Stage 2 — generative rollout

For up to `FREE_RUN_WINDOW_SIZE` steps (or until eval/anchor runway ends):

1. `f = Indices().second` — held-out truth index for this step’s score  
2. `Predict(outputs)` — estimate of `S[f]` at current reservoir state  
3. Past real → input port; `ExtractFuturePredicted(outputs)` → feedback port  
4. `ReservoirStep(past, future)`  
5. Score vs true `S[f]` (normalized channel-RMS); accumulate RMSE; record VPT  
6. Stop if past would leave the seed, stream ends, or step budget hit; else `Step()`

**What free-run measures.** Phase-tracking under continuous partial observation
(always-real past) + self-feedback on the future. Not Pathak-style pure free-run
without stating the anchor.

### Metrics (`FreeRunResult`)

| Field | Meaning |
|-------|---------|
| `valid` | False if zero generative steps scored (excluded from survey stats) |
| `vpt_steps` | First step whose channel-RMS error exceeds `VPT_THRESHOLD` (0 = never) |
| `vpt_lt` | That horizon in Lyapunov times; if never crossed, window floor (lower bound) |
| `rmse` | Free-run RMSE over scored steps (normalized units, all 3 channels) |
| `crossed` | Whether the threshold was ever crossed |
| `steps` | Generative steps actually scored |
| `row` | Pre-formatted table line for surveys |

Conversion: `steps_per_lt = 1 / (LYAPUNOV_EXPONENT · DT)` (canonical Lorenz-63
λ_max and the configured RK4 `DT`).

---

## 7. Configuration surface (`config::` in `Lorenz.h`)

Edit constants there and rebuild the `Lorenz` target. There is no runtime config
file. Groups of interest:

| Group | What it controls |
|-------|------------------|
| Diagnostics | `ENABLE_PRINTF` — banner, per-epoch train lines, free-run traces |
| Reservoir | dim, seed, spectral radius, input / leak / history depth |
| Ports | input vs external-feedback channel counts (fixed 4+4 in harness) and their scalings |
| Readout | online Adam LR schedule, epochs, delay-line slices, pooling / layers |
| Stream / Janus | training window (`span`), free-run window, center, stream length, dt |
| Stage | reservoir warmup steps before train updates |
| Score | VPT threshold (normalized channel-RMS), Lyapunov exponent for lt conversion |

Treat `Lorenz.h` as the source of truth for **current** numbers. Snapshot op-points
and survey outcomes in [`TRACKING.md`](TRACKING.md), not here.

---

## 8. Building and running

**Build** (CLion owns `cmake-build-*` on this machine — do not reconfigure those
dirs with a foreign generator):

```text
cmake --build <build-dir> --target Lorenz
```

Release preferred for numerics (`-O3 -ffast-math`).

**Run** — MinGW runtime on `PATH` (e.g. CLion’s bundled MinGW), then:

```text
Lorenz.exe [NUM_THREADS] [NUM_RUNS]
```

| Arg | Default behavior | Meaning |
|-----|------------------|---------|
| `NUM_THREADS` | hardware concurrency | Parallel trials; each trains one ESN on `base_seed + t` |
| `NUM_RUNS` | CLI default in `main` | Free-runs **per trial** after that training (not free-run *length*) |

`FREE_RUN_WINDOW_SIZE` (generative steps scored per free-run) is a **compile-time**
knob in `config::`. CLI `NUM_RUNS` is how many independent free-runs (with fresh
orbits) to average — orthogonal to window length.

### Cost model (why large `NUM_RUNS` is heavy)

Per trial, order of magnitude:

1. **Train:** `EPOCHS` × ~training-window online steps (each Predict + TrainStep +
   ReservoirStep), minus warmup consumed inside the window.
2. **Free-run × NUM_RUNS:** each free-run **rebuilds an orbit**, **re-washes the full
   training window**, then scores up to `FREE_RUN_WINDOW_SIZE` generative steps.

Large `NUM_THREADS × NUM_RUNS` can mean tens of millions of reservoir steps with
**little final stdout** until every thread finishes. Progress lines go to **stderr**
(`[seed …] free-run k/N`). Prefer modest `NUM_RUNS` for day-to-day checks; reserve
huge surveys for overnight.

### What `main` does today

Multi-thread **seed survey**:

- Shared starting `orbit_seed` across trials → same sequence of held-out orbits;
  ESN seed is the only independent variable.
- **One level of parallelism:** outer `jthread`s run trials; each trial’s HCNN is
  forced single-threaded so the survey does not nest a full worker pool inside
  every ESN.
- `main` forces `ENABLE_PRINTF = false` for the survey; worker exceptions are
  caught per trial; each trial returns a report string (aggregate VPT / RMSE +
  top-10) printed in seed order after join.
- Completion `Beep` on Windows.

For a **single interactive run** (config banner + per-epoch train RMSE + free-run
trace): leave or set `config::ENABLE_PRINTF = true`, call
`Lorenz(seed, orbit_seed).Train()` then `FreeRun(true)` from a slim `main`, or
temporarily replace the survey body.

---

## 9. What this example is (and is not)

**Is:**

- Dual-cursor, opposite-ends lag curriculum on one forward Lorenz orbit
- Online (single-sample) HCNN readout training with prequential monitoring
- Half-anchored closed loop: past always teacher, future becomes student past `ub`
- A harness for long rollouts with a restoring tether and multi-seed surveys

**Is not:**

- Unassisted Pathak-style free-run VPT without an anchor (do not claim that)
- A frozen hyperparameter sheet — numbers live in `Lorenz.h` / `TRACKING.md`

---

## 10. Related docs

| Doc | Role |
|-----|------|
| [`JanusCursor.md`](JanusCursor.md) | Exact cursor API and index geometry |
| [`docs/reservoir_feedback_mechanism.md`](../../docs/reservoir_feedback_mechanism.md) | External-feedback **port** (mechanism; this harness supplies the policy) |
| [`TRACKING.md`](TRACKING.md) | Live experiment log (op-points, surveys; historical FSF A/B retained) |
| [`docs/LorenzFreeRun.md`](../../docs/LorenzFreeRun.md) | Historical A(x) vs tanh free-run campaign (stale harness) |
| Project [`docs/README.md`](../../docs/README.md) | Library-wide reading order |
