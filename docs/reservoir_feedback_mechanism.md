# Reservoir external-feedback port

> Status: **landed** — optional second drive port on `Reservoir` / `ESN`.  
> Scope: **mechanism only** (buffers, weights, inject, clear). **Your app**
> decides what floats (if any) go on the port each step — the library never
> invents them.  
> Full reservoir picture: [Reservoir.md](Reservoir.md). Pipeline vocabulary:
> [CPP_SDK.md](CPP_SDK.md).

---

## What this port is

An optional, **caller-owned** drive path that mirrors the task **input** port:

- Same geometry: dim-neighbor XOR gather into each vertex.
- Own weight block and scale (`external_feedback_scaling`).
- Staged each step, **consumed and zeroed** by `Step()`.
- **Outside** spectral-radius rescale (like input; unlike the recurrent block).

It is an *input into the reservoir*, not a second path to task output **y**.
Only the readout emits **y**. There is **no** auto-wired loop from `y` back into
the reservoir. If you want closed-loop drive, **your code** packs a length-D
vector each step and passes it into `ReservoirStep` (or `InjectExternalFeedback`).

```
  Your app                          HypercubeESN
  ────────                          ────────────
  task u(t)  ─────────────────────▶ │
                                    │  Reservoir ──▶ slices ──▶ readout ──▶ y
  optional fb(t) ─────────────────▶ │
       ▲                            │
       │                            │
       │   only if YOU write it:    │
       │   e.g. pack(y(t−1)),       │
       │   or teacher-forced truth  │
       └────────────────────────────┘  (not built into Step)
```

Examples of what **your app** might put in `fb(t)`:

| Choice | Typical use |
|--------|-------------|
| Omit / `nullptr` | Open loop — port unused this step (or D = 0) |
| True / scheduled values from data | Teacher forcing during train |
| Packed last prediction `y(t−1)` | Free-run / generative closed loop |
| Anything else | Custom controllers, auxiliaries, … |

The reservoir only sees D floats (or zeros). It does not know which row of that
table you used.

**Not** full-state linear feedback (FSF) or any auto-wired self-loop inside the
core. FSF was removed; this port is the general substrate that remains.

---

## Config (`ReservoirConfig`)

| Field | Default | Rule |
|-------|---------|------|
| `num_external_feedback_channels` (D) | `0` | **0** = path off (no buffer, no weights). Else **[1, N]** with N = 2<sup>dim</sup>. |
| `external_feedback_scaling` | `0.5` | Used only if D > 0. Weights drawn U(−1,1) then × `scaling / √dim`. |

D **need not** divide N (unlike `num_inputs`, which must divide N).

```cpp
ReservoirConfig cfg;
cfg.dim = 10;
cfg.num_external_feedback_channels = 4;   // enable port
cfg.external_feedback_scaling      = 0.04f;
auto r = Reservoir::Create(cfg);
```

On ESN:

```cpp
ESNConfig esn_cfg;
esn_cfg.reservoir.num_external_feedback_channels = 4;
esn_cfg.reservoir.external_feedback_scaling      = 0.04f;
ESN esn(esn_cfg);
// esn.NumExternalFeedbackChannels() == 4
```

---

## Weight layout and gather

When D > 0, `vtx_weight_` is:

```
[ input: N × dim ]
[ external feedback: N × dim ]   // this port
[ recurrent: N × M × dim ]       // SR-rescaled only
```

| | Input | External feedback | Recurrent |
|--|-------|-------------------|-----------|
| Allocated if | always | D > 0 | always |
| Init scale | `input_scaling / √dim` | `external_feedback_scaling / √dim` | `1/√(dim·M)` then SR |
| In SR rescale? | No | **No** | Yes |
| RNG substream | Input (2) | ExternalFeedback (3) | Recurrent (1) |

Per vertex in `UpdateState` (when D > 0), same pattern as input:

```
for i = 0 .. dim-1:
    s += ext_fb[v XOR (1<<i)] * W_ext[v][i]
```

then recurrent gather, `tanh(s) + bias`, leak blend. See [Reservoir.md](Reservoir.md)
for the full timestep.

**Stability:** placing the port outside SR rescale means closed-loop behavior is
**not** guaranteed by the open-loop spectral-radius target. Survey
`external_feedback_scaling` (and whatever rule you use to fill `fb`) separately.

---

## Channel layout (block broadcast)

`InjectExternalFeedback(channel, value)` writes `value` onto a contiguous
vertex block:

```
block = floor(N / D)
channel c  →  vertices [c · block, (c+1) · block)
```

- If D does not divide N, the **tail** vertices `D·block .. N-1` stay at 0 as
  *sources* of the staged field.
- Those tail vertices still **receive** drive via the neighbor gather when a
  neighbor lies inside a driven block.

Vector form requires `count == D` and non-null `values` when `count > 0`.

---

## Per-step contract

```
// Reservoir (low level)
InjectInput(...)                 // task channels
InjectExternalFeedback(...)     // optional; only if D > 0
Step()                           // update, age delay line, zero staged drives
// read Outputs() / SliceAt(age)
```

- Staged external feedback is **consumed and cleared** every `Step()` (same as
  input). Re-stage every timestep you want the drive.
- Skipping injection for a step leaves zeros on that port for that step.
- **Snapshots:** `TakeSnapshot` / `RestoreSnapshot` do **not** include staged
  drive buffers (empty between steps). Weights and bias are not in snapshots.
- **`Clear`:** zeros state, history, and staged drives; keeps weights and bias.

---

## Injection API

```cpp
void Reservoir::InjectExternalFeedback(size_t channel, float value);
void Reservoir::InjectExternalFeedback(const float* values, size_t count);
// count must equal D; throws if D == 0 and channel out of range / count mismatch
```

Throws `std::invalid_argument` if:

- channel ≥ D (or D == 0 for the channel API path used with invalid channel),
- vector `count != D`,
- `values == nullptr` with `count > 0`.

---

## ESN seam

```cpp
void ESN::ReservoirStep(const float* inputs,
                        const float* external_feedback = nullptr);
```

| Argument | Meaning |
|----------|---------|
| `inputs` | Always required: `NumInputs()` floats for this timestep |
| `external_feedback` | `nullptr` → skip port. Non-null → exactly `NumExternalFeedbackChannels()` floats |

Throws if `external_feedback != nullptr` while D = 0 (no silent no-op).

Order inside `ReservoirStep`: stage ext-fb (if any) → stage all inputs →
`Reservoir::Step()`.

### Paths that do **not** inject external feedback

| API | Ext-fb |
|-----|--------|
| `ReservoirWarmup` | No — open-loop task input only |
| `ReservoirRun` | No — open-loop collect |
| `ReservoirStep(..., nullptr)` | Explicit skip |
| `ReservoirStep(..., ptr)` | Yes, when D > 0 |

Closed-loop free-run / teacher forcing must call **`ReservoirStep`** (or the raw
`Reservoir` inject API) each step with a vector **your app** fills. Batch
warmup/run are intentionally open-loop.

### Example: teacher force then free-run (sketch)

```cpp
// D configured at construction
float fb[4];

// Teacher force: your app loads truth into fb, then steps
esn.ReservoirStep(u_t, fb_real_t);

// Free-run: your app packs last prediction into fb, then steps
auto y = esn.Predict();
// … pack y into fb_pred …
esn.ReservoirStep(u_t, fb_pred);
```

What goes in `fb_*` is **your** rule, not something ESN selects. A production
example is the Lorenz free-run harness: past block on the **input** port, future
block on this **external-feedback** port (real during train, predicted in
free-run) — see `examples/Lorenz/`.

---

## Python bindings

External feedback is **not** exposed on the Python `ESN` surface today
(constructor has no D / scaling knobs; drive APIs are open-loop). Use the C++
API for closed-loop work, or extend the bindings later.

---

## What this document is not

| Claim | Reality |
|-------|---------|
| Auto closed-loop inside `Step` | No — **your app** stages values (or passes `nullptr`) |
| Built-in “use last y as feedback” | No — you pack and pass `fb` if you want that |
| Path from reservoir to **y** | No — readout only |
| SR guarantees closed-loop stability | No — port is outside rescale |
| Replacement for recurrent history depth M | No — orthogonal drive port |
| FSF / linear state feedback controller | Removed; this is a generic drive substrate |

---

## Related

| Document / code | Role |
|-----------------|------|
| [Reservoir.md](Reservoir.md) | Full drive ports, timestep, SR |
| [CPP_SDK.md](CPP_SDK.md) | `ReservoirConfig` / `ReservoirStep` API |
| `Reservoir.h` / `Reservoir.cpp` | Implementation |
| `ESN.h` / `ESN.cpp` | `ReservoirStep` seam |
| `examples/Lorenz/` | Policy example (Janus cursor free-run) |
| `main.cpp` | Snapshot / drive tests including ext-fb |
