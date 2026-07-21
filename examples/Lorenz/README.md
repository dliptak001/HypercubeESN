# Lorenz — Janus half-anchored free-run on Lorenz-63

Closed-loop experiment: train a HypercubeESN online on a Lorenz-63 orbit with a
**Janus dual-cursor** presentation, then free-run with **self-feedback on the
future port** while the past port stays anchored to real history.

This is **assisted / half-anchored** free-run — continuous partial observation
on the past — **not** classical unassisted autonomous generation (no true drive
at all). Report VPT and RMSE with that distinction stated.

| Layer | Files |
|-------|--------|
| Index motion | [`JanusCursor.h`](JanusCursor.h) · [`JanusCursor.md`](JanusCursor.md) |
| Orbit + normalize | [`LorenzAttractor.h`](LorenzAttractor.h), [`LorenzDatastream.{h,cpp}`](LorenzDatastream.h) |
| Ports, train, free-run, survey | [`Lorenz.{h,cpp}`](Lorenz.h) — all knobs in `config::` (incl. FSF A/B) |
| CMake target | `Lorenz` ← `Lorenz.cpp` + `LorenzDatastream.cpp` |

---

## 0. Full-state feedback A/B

In [`Lorenz.h`](Lorenz.h) `config::`:

| Knob | Default | Role |
|------|---------|------|
| `FULL_STATE_FEEDBACK` | `false` | Construction-only FSF port |
| `FSF_SEED` / `FSF_SCALING` | `1` / `0.5` | `B_fsf` weight scale (not V) |
| `FSF_SET_GAIN` / `FSF_GAIN_SCALE` | `false` / `0.05` | Optional default isotropic V after construct |

Independent of the **external-feedback** future port. Other examples share
[`FsfAbSwitch.h`](../FsfAbSwitch.h); Lorenz keeps knobs in `config::` for the
survey harness. Log line: `FSF A/B: ON|OFF …`.

## 1. Pipeline at a glance

```text
 LorenzAttractor (RK4, σ=10, ρ=28, β=8/3, dt=0.02)
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

## 2. Stream geometry (defaults)

From `config::` in `Lorenz.h`:

```text
 TRAINING_WINDOW_SIZE  = span = 20000
 FREE_RUN_WINDOW_SIZE  = E    = 2000
 CURSOR_CENTER_INDEX   = E + span/2 = 12000
 STREAM_LENGTH         = 2E + span  = 24000
```

```text
 array index n:   0 ······· lb ····· center ····· ub ········· stream end
                  │         │           │           │            │
                 seed   train edge   midpoint   train edge   eval / free-run
                 T=0       (lb)       (center)      (ub)         tail

 lb = 2000,  center = 12000,  ub = 22000
```

| Region | Indices | Role |
|--------|---------|------|
| Past free-run runway | `[0, lb)` | Anchor history the past cursor walks into after leaving the window |
| Training window | `[lb, ub]` | One epoch / washout sweep |
| Prediction / eval runway | `(ub, stream end]` | Future is generative; held-out truth for scoring |

`Build()` stores `stream_length + 1` samples (seed state + one sample per RK4
step). Cursor index math is documented in [`JanusCursor.md`](JanusCursor.md).

Each `Train` epoch and each `FreeRun` call **rebuilds** a fresh orbit
(`RebuildDatastream`) so successive free-runs share a fixed orbit-seed sequence
while the ESN seed is the independent variable in surveys.

---

## 3. Eight-channel drive and targets

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
- Gains are independent: `INPUT_SCALING` on past, `FEEDBACK_SCALING` on future
  (separate weight realizations on the two reservoir ports).
- Optional readout aux (`AUX_INPUT_DIM == 3`): normalized past `(x,y,z)` into the
  readout only; reservoir ports unchanged. Default is off (`0`).

### Lag curriculum (why opposite-ends matter)

For most of a training pass the past and future samples are **far apart** on the
attractor (up to ~`span` samples; hundreds of Lyapunov times at the ends with
defaults). They only share phase near the **center crossing**.

| Role | Port | Training | Free-run (generative) |
|------|------|----------|------------------------|
| Teacher of the map | Feedback / future | Real future history → target `S[f]` | Own prediction (closed loop) |
| Stabilizer / tether | Input / past | Present, often long-lag | Always real history |

Consequence:

- The learnable one-step signal is primarily on the **feedback** port
  (`S[f−1] → S[f]` style under teacher forcing).
- The past is a **free-run stabilizer**, not the main training teacher.
- `FEEDBACK_SCALING = 0` makes training ill-posed (climatological floor) — it is
  **not** a clean “observer floor.” To ablate the anchor, dose `INPUT_SCALING`
  (or zero the past channels) at free-run time; do not zero feedback during train.

---

## 4. Normalization

`LorenzDatastream::Normalize` runs **once** after RK4; storage is `float` ≈ `[-1, 1]`.

1. Scan **this** raw orbit for per-channel min/max (full stream, including eval tail).
2. Per-channel **midpoint offset** `c_v = (v_max + v_min) / 2` (drops `z`’s DC ~+24 onto zero).
3. **One shared scale** = max of the three half-ranges; zero scale → `1.0`.
4. Store `(v − c_v) / scale` per channel.

Shared scale preserves relative amplitudes (widest channel reaches ±1; others use
less). Extremes are stream-dependent, not hardcoded. Free-run scoring is in these
**normalized units**.

---

## 5. Training protocol (`Lorenz::Train`)

Per epoch `i = 0 .. EPOCHS−1`:

1. **`RebuildDatastream`** — new orbit from the mixed orbit seed; `Reset()` cursors
   (past@`ub`, future@`lb`).
2. **Warmup** (`RESERVOIR_WARMUP_STEPS`): teacher-forced `ReservoirStep(past, future_real)`
   only; no readout update.
3. **Train sweep** while `!OOB()`:
   - **Horizon-1 alignment:** `Predict` at reservoir state `x(t)` **before** injecting
     this step’s drive; target = current future sample `S[f]` (about to be injected),
     not `S[f+1]`.
   - **Prequential** (test-then-train): pre-update prediction vs that target for the
     epoch RMSE line.
   - Optional **exposure-bias remedies** on the future **linear** channels only
     (teacher target stays real `S[f]`) — see [§7](#7-exposure-bias-remedies).
   - Re-derive `future[3] = x·z`, then `TrainStep` → `ReservoirStep` → cursor `Step()`.
4. Report prequential train RMSE (3 channels × train steps). LR from `LrProfile`
   (cosine anneal from `LEARNING_RATE` to `LEARNING_RATE_MIN` by 75% of epochs, then hold).

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
| `vpt_steps` | First step whose channel-RMS error exceeds `VPT_THRESHOLD` (0 = never) |
| `vpt_lt` | That horizon in Lyapunov times; if never crossed, window floor (lower bound) |
| `rmse` | Free-run RMSE over scored steps (normalized units, all 3 channels) |
| `crossed` | Whether the threshold was ever crossed |
| `row` | Pre-formatted table line for surveys |

Conversion: `steps_per_lt = 1 / (LYAPUNOV_EXPONENT · DT)` with λ ≈ 0.9056, dt = 0.02
→ **≈ 55.2 steps / Lyapunov time**.

Callout in config (illustrative, single free-run): seed **13649419** —
VPT 347 steps (~6.28 λt), free-run RMSE ~0.43 over 2000 steps
(`free_run_error_seed13649419.png`).

---

## 7. Exposure-bias remedies

Training is teacher-forced on the future port; free-run feeds the model’s own
prediction. That train/test drive mismatch is **exposure bias**. Both remedies act
**only in `Train()`**, **only on the future block** (feedback port). The teacher
target stays real `S[f]`. After any perturbation, `future[3] = x·z` is re-derived
so the block matches free-run product semantics. Washout in `FreeRun` stays clean.

Enable **one at a time** for single-delta ablations. Both default **off**.

| Knobs | Effect |
|-------|--------|
| `TRAIN_FUTURE_NOISE` (2a) | Zero-mean Gaussian std on future `x,y,z`. `0` disables. Starting bracket ~`1e-3` … few `1e-2`. |
| `SCHEDULED_SAMPLING_CEILING` (2b) | Probability ceiling of replacing future `x,y,z` with the model’s fresh prediction; linear ramp `0 → ceiling` across epochs. `0` disables. Starting ceiling ~`0.25` … `0.5`. |
| `TRAIN_EXPOSURE_RNG_SEED` | Dedicated RNG; toggling remedies never perturbs the reservoir seed. |

---

## 8. Default model knobs (`config::`)

| Group | Knob | Default |
|-------|------|---------|
| Reservoir | `DIM` | 11 (N = 2048) |
| | `SEED` | 13649419 |
| | `SPECTRAL_RADIUS` | 0.99 |
| | `INPUT_SCALING` | 0.005 |
| | `FEEDBACK_SCALING` | 0.04 |
| | `LEAK_RATE` | 1.0 |
| | `HISTORY_DEPTH` | 24 |
| Readout | online Adam, 3 outputs | |
| | `LEARNING_RATE` → `LEARNING_RATE_MIN` | 4e-5 → 2e-6 |
| | `EPOCHS` | 100 |
| | `READOUT_SLICES` / `AUX_INPUT_DIM` / pooling | 1 / 0 / on |
| Data | `TRAINING_WINDOW_SIZE` / free-run / warmup | 20000 / 2000 / 1000 |
| Score | `VPT_THRESHOLD` | 0.3 (normalized channel-RMS) |

Edit constants in `Lorenz.h`; rebuild the `Lorenz` target. There is no runtime
config file.

---

## 9. Building and running

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

| Arg | Default | Meaning |
|-----|---------|---------|
| `NUM_THREADS` | `hardware_concurrency` | Parallel trials; each trains one ESN on `seed + t` |
| `NUM_RUNS` | 2000 | Free-runs per trial after that training |

`main` currently runs a **multi-thread seed survey**:

- Shared starting `orbit_seed` across trials → same sequence of held-out orbits;
  ESN seed is the only independent variable.
- **One level of parallelism:** outer `jthread`s run trials; each trial’s HCNN is
  forced single-threaded (`readout.num_threads = 1`) so the survey does not nest
  a full worker pool inside every ESN.
- Worker exceptions are caught and reported in that trial’s slot (no process abort).
- Per-run live printf is silenced (`ENABLE_PRINTF = false`); each trial returns a
  report string (aggregate VPT / RMSE stats + top-10 leaderboards) printed in seed
  order after all threads join.
- Completion `Beep` on Windows.

For a **single interactive run** (config banner + per-epoch train RMSE + free-run
trace): set `config::ENABLE_PRINTF = true`, call `Lorenz(seed, orbit_seed).Train()`
then `FreeRun(true)` from a slim `main`, or temporarily replace the survey body.
Do not launch long surveys without intending the wall-clock cost (DIM 11 × 100
epochs × many free-runs is heavy).

---

## 10. What this example is (and is not)

**Is:**

- Dual-cursor, opposite-ends lag curriculum on one forward Lorenz orbit
- Online (single-sample) HCNN readout training with prequential monitoring
- Half-anchored closed loop: past always teacher, future becomes student past `ub`
- A harness for long rollouts with a restoring tether and multi-seed surveys

**Is not:**

- Unassisted Pathak-style free-run VPT without an anchor (do not claim that)

---

## 11. Related docs

| Doc | Role |
|-----|------|
| [`JanusCursor.md`](JanusCursor.md) | Exact cursor API and index geometry |
| [`docs/reservoir_feedback_mechanism.md`](../../docs/reservoir_feedback_mechanism.md) | External-feedback **port** (mechanism; this harness supplies the policy) |
| [`docs/full_state_linear_feedback.md`](../../docs/full_state_linear_feedback.md) | Internal full-state feedback (not used by this example) |
| [`docs/LorenzFreeRun.md`](../../docs/LorenzFreeRun.md) | Historical A(x) vs tanh free-run campaign (stale harness) |
| Project [`docs/README.md`](../../docs/README.md) | Library-wide reading order |
