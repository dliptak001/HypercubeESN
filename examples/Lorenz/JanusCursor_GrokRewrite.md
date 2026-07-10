# Janus Cursor — code-faithful reference

> **Authority:** this document describes the method **as implemented**.
> Source of truth is the code, not the older `JanusCursor.md` (out of date).
>
> | Layer | Files |
> |-------|--------|
> | Index motion | `JanusCursor.h` |
> | Orbit + normalization | `LorenzDatastream.{h,cpp}` |
> | Ports, train, free-run, scoring | `Lorenz.{h,cpp}`, `config::` in `Lorenz.h` |

**Method vs. instantiation.** Cursor geometry is system-agnostic: any array that
supports `value(index)` can sit under the dual indices. This module instantiates
the method on **Lorenz-63**. Read the *geometry and protocol* as generic; read
Lorenz channels, RK4, and Lyapunov scoring as the worked example.

---

## 0. One-paragraph statement

We drive an ESN with **two cursors** over a single **precomputed, positive-time**
trajectory `S[·]`. On each pass they start at **opposite ends** of a training
window and walk toward (then past) each other: past decrements from the right
edge, future increments from the left edge. Each cursor contributes a 4-vector
`(x, y, z, x·z)`, so the drive is **8-D**, split across two reservoir ports:

- **input port** ← past block (always real history — the **anchor**)
- **feedback port** ← future block (real in training; **model prediction** once
  the future cursor leaves the window)

Training is multi-epoch: each epoch `Reset()`s the cursors and sweeps the window
once, teacher-forced, learning a horizon-1 map to the **future** sample. Free-run
is self-contained: `Reset()`, re-sweep the window teacher-forced (washout), then
close the loop on the feedback port while the past cursor keeps reading real
history. Half-anchored free-run is **assisted** closed-loop prediction, not
unassisted autonomous free-run.

---

## 1. Stream geometry (one array, three regions)

Everything is positive time. At setup, Lorenz-63 is integrated **once** with
fixed-step RK4 from a seed state; cursors only **read indices**.

Default `config::` layout (`Lorenz.h`):

```
 TRAINING_WINDOW_SIZE  = span = 20000
 FREE_RUN_WINDOW_SIZE  = E    = 2000
 CURSOR_CENTER_INDEX   = E + span/2 = 12000
 STREAM_LENGTH         = 2E + span  = 24000
```

```
 array index n:   0 ······· lb ····· center ····· ub ········· stream end
                  │         │           │           │            │
                 seed   train edge   midpoint   train edge   eval / free-run
                 T=0       (lb)       (center)      (ub)         tail

 lb     = center − span/2
 ub     = center + span/2
 window = [lb, ub]     training / washout window (span samples wide, integer half-span)
```

With the defaults: `lb = 2000`, `center = 12000`, `ub = 22000`.

| Region | Indices | Role |
|--------|---------|------|
| Past free-run runway | `[0, lb)` | Anchor history the past cursor walks into after leaving the window |
| Training window | `[lb, ub]` | One epoch / washout sweep lives here |
| Prediction / eval runway | `(ub, stream end]` | Future cursor is generative; held-out truth for scoring if precomputed |

`Build()` stores `stream_length + 1` samples (seed state plus one sample per
integration step). `LorenzDatastream` rejects configs where the window underruns
index 0 or overruns `stream_length`.

**Offline vs open-ended free-run.** Scoring needs true `S[f]` past `ub`. That
tail exists only when the orbit was integrated past the window (this harness
always does). On live data without a future, the run can still go generative;
only the score is lost. The past anchor remains available down to the seed.

---

## 2. Cursor motion (what `JanusCursor` actually does)

### Construction

```text
JanusCursor(span, center_index)
  lb_ = center_index − span/2
  ub_ = center_index + span/2
  require span > 0 and center_index > span/2
  Reset()
```

### Reset positions (opposite ends)

| Cursor | `Reset()` sets index to | Step direction |
|--------|-------------------------|----------------|
| **Past** | `ub` | `idx −= 1` |
| **Future** | `lb` | `idx += 1` |

They start at opposite edges of the window and walk toward each other, meet at
center, then continue toward the opposite edges.

```text
 k (steps after Reset)     0              ~span/2            ~span
 past index              ub  ──────────►  center  ─────────►  lb
 future index            lb  ──────────►  center  ─────────►  ub
 |future − past|        span               ~0               span
 Distance()              −1                 ~0               +1
```

`Distance()` is the continuous order parameter of this pass:

```text
Distance = (future_index − past_index) / span
```

At start: `(lb − ub) / span = −1`. At the center crossing: `~0`. Near the end of
a full window walk: approaches `+1`.

### Public API (`JanusCursor.h`)

| Method | Behavior |
|--------|----------|
| `Reset()` | Past → `ub`, future → `lb` |
| `Step()` | Advance both; return `{past_idx, future_idx}` |
| `Indices()` | Current pair |
| `NextIndices()` | Peek one step ahead (past−1, future+1) |
| `OOB()` | **Future only:** `future_index > ub` |
| `Distance()` | `(future − past) / span` as `float` |
| `AtStartPosition()` | Past at `ub` (proxy for “just reset”) |

There is **no** reflecting shuttle, **no** `StepBounded` / `StepUnbounded`, and
**no** “both start at center as ±i mirrors.” Multi-epoch presentation is the
caller’s job: `Reset()` + sweep until `OOB()`.

### Out-of-window behavior (split across layers)

- **Future past `ub`:** `JanusCursor::OOB() == true`. `LorenzDatastream::Step()`
  still advances both indices but returns `future == nullptr` (no teacher sample).
- **Past below 0:** `LorenzDatastream::Step()` throws
  (`free-run outran the anchor history`). `FreeRun()` also stops proactively when
  `past_index <= 0` before stepping further.

Past-cursor “below `lb`” is tracked inside the nested class but is **not** what
`JanusCursor::OOB()` reports; the public generative signal is **future overran
`ub`**.

---

## 3. Lag curriculum (why this geometry matters)

Because the cursors start at opposite ends, for most of a training pass the past
and future samples are **far apart on the attractor** — up to `span` samples
(~`span · λ · dt` Lyapunov times; with defaults and `λ≈0.9056`, `dt=0.02`, on
the order of hundreds of λt at the ends). They only share phase near the
**center crossing**.

Consequence, confirmed experimentally (`recovery.md`):

- The **learnable one-step signal** during training lives primarily on the
  **feedback port**: the reservoir sees the future stream’s own recent history
  and must predict the next future sample (`S[f−1] → S[f]` style autoregression
  under teacher forcing).
- The past block is often a **decorrelated** attractor point. It is real, but not
  a tight phase cue for `S[f]` except near center.
- Setting `FEEDBACK_SCALING = 0` makes training ill-posed (climatological floor):
  it is **not** a clean “observer floor.” The past is a **free-run stabilizer**,
  not the main training teacher.

So the honest split of roles is:

| Role | Port / half | Training | Free-run (generative) |
|------|-------------|----------|------------------------|
| Teacher of the map | Feedback / future | Real `S[f]` history → target `S[f]` | Own prediction (closed loop) |
| Stabilizer / tether | Input / past | Present, often long-lag | Always real history |

---

## 4. Eight-channel drive and targets

### Blocks

```text
 input port    (4):  past   [ x_p, y_p, z_p, x_p·z_p ]   ExtractPast
 feedback port (4):  future [ x_f, y_f, z_f, x_f·z_f ]   ExtractFutureReal
                                                         or ExtractFuturePredicted
```

- Linear channels come from the normalized stream (or from the 3-D prediction).
- The 4th channel is **always** the product of that block’s current `x` and `z`
  (derived feature, never a readout target).
- Targets are **3-D**: future `(x, y, z)` only (`ExtractTargets`).

Optional readout aux (`config::AUX_INPUT_DIM == 3`): normalized past `(x,y,z)`
into the readout only; reservoir ports unchanged. Default aux is off (`0`).

### Gains

```text
 input_scaling     → past block  (config::INPUT_SCALING)
 feedback_scaling  → future block (config::FEEDBACK_SCALING)
```

Independent gains and independent weight realizations on the two ports. Anchor
**dose** is largely `INPUT_SCALING` on the past; generative drive strength is
`FEEDBACK_SCALING` on the future.

---

## 5. Normalization (`LorenzDatastream::Normalize`)

Applied **once** after integration; stored stream is `float` in roughly
`[-1, 1]`.

1. Scan **this** raw orbit for per-channel min/max (full stream, including eval
   tail).
2. Per-channel **midpoint offset**:
   `c_v = (v_max + v_min) / 2` for `x`, `y`, and `z`
   (centers `z`’s DC ~+24 onto zero; also centers residual offset in `x,y`).
3. **One shared scale** = max of the three half-ranges
   `(v_max − v_min) / 2`. Degenerate zero scale → `1.0`.
4. Store `(v − c_v) / scale` per channel.

Shared scale preserves relative amplitudes across axes (widest channel reaches
±1; others use less of the range). Extremes are stream-dependent, not hardcoded.
`x·z` is formed **after** normalization as `x̂·ẑ`, so generative reconstruction
is a multiply of predicted channels — no denorm/renorm product bug class.

Scoring in `FreeRun` is in **these normalized units** (channel-RMS, VPT
threshold, free-run RMSE).

---

## 6. Training protocol (`Lorenz::Train`)

Per epoch `i = 0 .. EPOCHS−1`:

1. **`data_stream_.Reset()`** — past@`ub`, future@`lb`.
2. **Warmup** (`RESERVOIR_WARMUP_STEPS`): teacher-forced `ReservoirStep(past,
   future_real)` only; no readout update; advance cursors each step.
3. **Train sweep** while `!data_stream_.OOB()`:
   - Horizon-1 alignment: `Predict` at state `x(t)` **before** injecting this
     step’s inputs; target = current future sample `S[f]` (the sample about to
     be injected), not `S[f+1]`.
   - Prequential error: pre-update prediction vs that target.
   - Optional exposure remedies (**future linear channels only**; teacher target
     stays real `S[f]`):
     - **2a** `TRAIN_FUTURE_NOISE` — Gaussian noise on future `x,y,z`
     - **2b** `SCHEDULED_SAMPLING_CEILING` — with ramped probability, replace
       future `x,y,z` with the model’s fresh prediction
     - Then re-derive `future[3] = x·z` so the block matches free-run product
       semantics
   - `TrainStep` then `ReservoirStep(past, future)`; cursor `Step()`.
4. Report prequential train RMSE (3 channels × train steps). LR from
   `LrProfile` (cosine anneal to floor by 75% of epochs, then hold).

Multi-epoch = outer loop + `Reset`. The cursor itself does one **one-way** pass
per epoch, not a reflecting triangle wave.

---

## 7. Free-run protocol (`Lorenz::FreeRun`)

Self-contained relative to training state of the **cursors** (reservoir weights
are whatever `Train` left; cursor phase is rebuilt):

### Stage 1 — anchored washout

- `Reset()`
- While `!OOB()`: teacher-forced `ReservoirStep(past, future_real)`; no readout
  updates
- Leaves the reservoir at the window edge, warm and in-distribution, ready to
  go generative on the next conceptual step

### Stage 2 — generative rollout

For up to `FREE_RUN_WINDOW_SIZE` steps (or until eval/anchor runway ends):

1. `f = Indices().second` — held-out truth index for this step’s score  
2. `Predict(outputs)` — model’s estimate of `S[f]` at current reservoir state  
3. `ExtractPast` → input port (real)  
4. `ExtractFuturePredicted(outputs)` → feedback port (closed loop)  
5. `ReservoirStep(past, future)`  
6. Score `outputs` vs true `S[f]` (normalized); accumulate RMSE; VPT = first step
   whose channel-RMS error exceeds `VPT_THRESHOLD`  
7. Stop if past would leave the seed, stream ends, or step budget hit; else
   `Step()` (future may already be past `ub`; past keeps walking left through
   real history)

**What free-run measures.** Phase-tracking skill under **continuous partial
observation** (always-real past on the input port) and **self-feedback** on the
future port. It is **not** classical unassisted free-run (no true drive at all).
Report it as half-anchored / assisted free-run. Anchor ablation
(`INPUT_SCALING` dose, or zeroing past channels) is the right way to isolate how
much the tether contributes; do not treat `FEEDBACK_SCALING = 0` as that
ablation during **training** (see §3).

---

## 8. Code map

```text
JanusCursor
  PastCursor / FutureCursor   opposite-ends walk, shared [lb, ub]
  Step / Reset / OOB / Distance

LorenzDatastream : JanusCursor
  Build()        RK4 integrate once
  Normalize()    midpoint offsets + shared scale → float stream
  States()       (Distance, S[past], &S[future])
  Step()         advance cursors; nullptr future if OOB; throw if past < 0

Lorenz
  ExtractPast / ExtractFutureReal / ExtractFuturePredicted / ExtractTargets
  Train()        Reset → warmup → teacher-forced epoch sweeps
  FreeRun()      Reset → washout sweep → generative self-feedback + score
```

---

## 9. Default numeric sketch (illustrative)

With current `config::` defaults:

| Quantity | Value |
|----------|-------|
| span / window | 20000 |
| center | 12000 |
| lb, ub | 2000, 22000 |
| past runway `[0, lb)` | 2000 samples |
| eval tail past ub | stream through index 24000 (`STREAM_LENGTH+1` samples in the vector) |
| free-run budget | 2000 generative steps |
| dt | 0.02 |
| λ (scoring) | 0.9056 → ~55.2 steps / Lyapunov time |

Exact envelopes for normalization depend on the integrated orbit for the chosen
initial condition.

---

## 10. What this method is (and is not)

**Is:**

- Dual-cursor presentation of one forward stream
- Opposite-ends lag curriculum during each pass
- Asymmetric closed loop: past always teacher, future becomes student past `ub`
- A harness for long rollouts with a restoring tether and re-lock behavior

**Is not:**

- Center-mirror `±i` free-run from a shared present (older doc; not this code)
- Reflecting multi-sweep physics inside the cursor
- Seamless “end training at center → continue free-run without Reset”
- A claim of pure autonomous generative skill equal to Pathak-style free-run VPT
  without stating the anchor

---

## 11. Related notes in-tree

- `README.md` — exposure bias, identity burden, mitigation knobs  
- `recovery.md` — re-lock is the anchor; why `FEEDBACK=0` training fails  
- `past_head.md` — past diagnostic head decoupled from future error (removed)  
- `JanusCursor.md` — **historical / out of date**; prefer this file + the sources
  in the header table
