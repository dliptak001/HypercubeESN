# HypercubeESN Python SDK

Python bindings for the HypercubeESN C++ library: a fixed Boolean-hypercube
reservoir plus a trainable HypercubeCNN readout, exposed as one `ESN` class.

Deep dives on the C++ core: [Reservoir.md](Reservoir.md) · [Readout.md](Readout.md)
· [CPP_SDK.md](CPP_SDK.md).

Package version: **2.0.0** (`hypercube_esn.__version__`).

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Pipeline vocabulary](#pipeline-vocabulary)
- [API reference](#api-reference)
- [Input data layout](#input-data-layout)
- [Data types](#data-types)
- [Error handling](#error-handling)
- [Model persistence](#model-persistence)
- [Limitations](#limitations)
- [Dependencies](#dependencies)

## Installation

### From PyPI (recommended)

Pre-built wheels for Python 3.10–3.14 on Windows (x64), Linux (x86_64, aarch64),
and macOS (x86_64, arm64):

```bash
pip install hypercube-esn
```

Import as `import hypercube_esn as he` (package name on PyPI is `hypercube-esn`).

### From source

Requirements: Python 3.10+, C++23 compiler (GCC 13+, Clang 17+, MSVC 2022+),
CMake 3.20+, scikit-build-core, pybind11, NumPy.

```bash
git clone https://github.com/dliptak001/HypercubeESN.git
cd HypercubeESN/python
pip install .
```

On Windows with MinGW, put the toolchain on `PATH` and set the generator:

```powershell
pip install scikit-build-core pybind11 numpy
$env:PATH = "C:\path\to\mingw\bin;" + $env:PATH
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_MAKE_PROGRAM = "C:\path\to\ninja.exe"
$env:CC = "C:\path\to\mingw\bin\gcc.exe"
$env:CXX = "C:\path\to\mingw\bin\g++.exe"
pip install . --no-build-isolation
```

### Running tests

```bash
pip install ".[test]"
pytest python/tests/
```

## Quick start

### Simple (recommended)

```python
import numpy as np
import hypercube_esn as he

signal = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)

esn = he.ESN(dim=7, seed=73895)
esn.fit(signal, warmup=200)       # warmup, run, train in one call

print(f"R² = {esn.r2():.6f}")     # held-out test R²
print(f"NRMSE = {esn.nrmse():.6f}")
```

### Explicit (full control)

```python
import numpy as np
import hypercube_esn as he

signal = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)

esn = he.ESN(dim=7, seed=73895)
esn.reservoir_warmup(signal[:200])
esn.reservoir_run(signal[200:-1])

targets = signal[201:]
esn.train(targets[:1400])

r2 = esn.r2(targets, start=1400)  # count defaults to all remaining
print(f"R² = {r2:.6f}")
```

---

## Pipeline vocabulary

Same story as the C++ core:

```
  inputs [+ optional external feedback]
    │
    ▼
  Reservoir (fixed)
    │
    ▼
  pack B ages (N each)  ──▶  HCNN readout (trained) ──▶ y
```

| Term | Meaning |
|------|---------|
| **N** | Reservoir neurons = 2<sup>dim</sup> (`reservoir_neuron_count`) |
| **M** | `history_depth` — delay-line depth used by the **recurrent** gather |
| **B** | Delay-line ages packed into the readout (`readout_slices`). Power of two, 1 ≤ B ≤ M. Default 1. |
| **Reservoir state** | Newest published slice (`copy_reservoir_state`) — N floats |
| **Readout input** | What the HCNN trains/predicts on (`copy_readout_input`) — B·N floats |

Only the readout is trained. The reservoir is frozen after construction.

---

## API reference

### The `reservoir_hypercube_dimension` parameter

Controls reservoir hypercube size. N = 2<sup>dim</sup>. Supported: **5–16**.

| dim  | Neurons   | Typical use |
|------|-----------|-------------|
| 5    | 32        | Fast prototyping, embedded |
| 6    | 64        | Light benchmarks |
| 7    | 128       | Standard benchmarks |
| 8    | 256       | Production, complex tasks |
| 9–16 | 512–65536 | Research, high-capacity tasks |

---

### ESN

Owns the full Reservoir → Readout pipeline.

```python
import hypercube_esn as he

esn = he.ESN(dim=7)  # or reservoir_hypercube_dimension=7
esn = he.ESN(dim=7, leak_rate=0.3, history_depth=8, readout_slices=2)

# High-level
esn.fit(signal, warmup=200)
esn.r2(); esn.nrmse()

# Low-level
esn.reservoir_warmup(inputs)
esn.reservoir_run(inputs)
esn.reservoir_run(inputs, clear_recorded=True)
esn.train(targets)

# Predict / score
esn.predict()
esn.predict_from_recorded(t)
esn.predictions()
esn.r2(targets, start=1400)
esn.nrmse(targets, start, count)
esn.accuracy(labels, start, count)

# State
esn.collected_states()
esn.copy_reservoir_state()
```

---

#### Construction

```python
ESN(
    dim=7,                         # or reservoir_hypercube_dimension=7
    *,
    seed=73895,
    spectral_radius=0.99,
    input_scaling=0.5,
    leak_rate=1.0,
    num_inputs=1,
    history_depth=16,
    verbose=False,
    num_external_feedback_channels=0,
    external_feedback_scaling=0.5,
    bias_scaling=0.02,
    readout_slices=1,              # B; power of two ≤ history_depth
    readout_num_outputs=1,
    readout_task="regression",
    readout_num_layers=1,          # house default; 0 = auto min(dim-2, 2)
    readout_conv_channels=16,
    readout_epochs=200,
    readout_batch_size=32,
    readout_lr_max=0.0015,
    readout_lr_min_frac=0.01,
    readout_lr_decay_epochs=0,
    readout_weight_decay=0.0,
    readout_momentum=0.0,
    readout_activation="tanh",
    readout_seed=42,
    readout_num_threads=0,
    readout_restore_best_epoch=True,
    readout_best_epoch_holdout_frac=0.0,
)
```

Reservoir weights are drawn and spectral-radius-rescaled at construction. The
HCNN is built eagerly from the `readout_*` kwargs and is ready before the first
`train` / `train_step`.

**Still C++-only** (not yet in Python): `use_pooling` / pool type / batch-norm /
optimizer choice / channel growth (C++ struct defaults: pooling on, Adam, …).

Closed-loop: set `num_external_feedback_channels=D>0`, then
`esn.reservoir_step(u_t, fb_t)` each step.

##### Reservoir parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` / `reservoir_hypercube_dimension` | `int` | — | Hypercube dim **[5, 16]**. N = 2<sup>dim</sup>. Prefer `dim=`. |
| `seed` | `int` | `73895` | Master reservoir seed (SplitMix64 substreams). Screen per task. |
| `spectral_radius` | `float` | `0.99` | Target ρ for the **recurrent** block. |
| `input_scaling` | `float` | `0.5` | Input weights × `input_scaling / √dim`. Local construction only. |
| `leak_rate` | `float` | `1.0` | Leaky integrator; 1.0 = full replacement. |
| `num_inputs` | `int` | `1` | Channels; must divide N. Channel k drives block `k·N/K`. |
| `history_depth` | `int` | `16` | Delay-line depth **M ∈ [1, 64]** for the **recurrent** gather — not readout B. |
| `verbose` | `bool` | `False` | Construction banner. |
| `num_external_feedback_channels` | `int` | `0` | D closed-loop channels; 0 = off. |
| `external_feedback_scaling` | `float` | `0.5` | Ext-fb weight scale (like input). |
| `bias_scaling` | `float` | `0.02` | Per-neuron bias after tanh; 0 disables. |
| `readout_slices` | `int` | `1` | B ages for the HCNN (power of two, ≤ M). |

##### Readout (HCNN) parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `readout_num_outputs` | `int` | `1` | Regression targets or class count. |
| `readout_task` | `str` | `"regression"` | `"regression"` or `"classification"`. |
| `readout_num_layers` | `int` | `1` | Conv(+Pool) stages. Default **1**. **`0` = auto** `min(dim−2, 2)`. |
| `readout_conv_channels` | `int` | `16` | First-layer channels (then × channel growth in C++ default stack). |
| `readout_epochs` | `int` | `200` | Batch-train epochs. |
| `readout_batch_size` | `int` | `32` | Mini-batch size. |
| `readout_lr_max` | `float` | `0.0015` | Cosine peak. Keep ≤ ~0.005 to avoid NaN. |
| `readout_lr_min_frac` | `float` | `0.01` | Floor = `lr_max * lr_min_frac`. |
| `readout_lr_decay_epochs` | `int` | `0` | Cosine horizon; `0` = use `readout_epochs`. |
| `readout_weight_decay` | `float` | `0.0` | L2 weight decay. |
| `readout_momentum` | `float` | `0.0` | SGD momentum (C++ default optimizer is Adam; momentum applies if SGD is selected in C++). |
| `readout_activation` | `str` | `"tanh"` | `"tanh"`, `"relu"`, `"leaky_relu"`, or `"none"`. |
| `readout_seed` | `int` | `42` | HCNN weight init seed. |
| `readout_num_threads` | `int` | `0` | HCNN workers: `0` auto, `1` single-threaded (multi-ESN hosts), `N` workers. |
| `readout_restore_best_epoch` | `bool` | `True` | Restore best epoch after batch `train` (min MSE / max accuracy). |
| `readout_best_epoch_holdout_frac` | `float` | `0.0` | Tail hold-out for best-epoch scoring; `0` = score full train set. |

---

#### High-level pipeline

##### `fit(inputs, targets=None, *, warmup=200, train_size=None, train_frac=None, horizon=1) → ESN`

Warmup → run → train with a train/test split. Stores targets so `r2()` /
`nrmse()` / `accuracy()` work with no arguments. Returns `self`.

**Auto-target** (`targets=None`, single-input only): next-step (or
`horizon`-step) targets from the signal.

```python
esn.fit(signal, warmup=200)
esn.fit(signal, warmup=200, train_size=1400)
esn.fit(signal, warmup=200, horizon=5)
```

**Explicit-target** (any `num_inputs`): one target row per collected state.
Required for multi-input and classification. `horizon` is ignored.

```python
esn.fit(inputs, targets=ch0[201:], warmup=200)
esn.fit(signal, targets=labels, warmup=200)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputs` | `ndarray` | — | `(steps,)` or `(steps, num_inputs)` |
| `targets` | `ndarray` | `None` | One target per collected state (explicit mode) |
| `warmup` | `int` | `200` | Transient washout steps |
| `train_size` | `int` | `None` | Train count; exclusive with `train_frac` |
| `train_frac` | `float` | `None` | Train fraction; default **0.7** if neither set |
| `horizon` | `int` | `1` | Auto-target shift; ignored with explicit targets |

```python
esn.fit(signal, warmup=200)
print(esn.r2())
print(esn.train_size, esn.test_size)
```

---

#### Low-level driving

##### `reservoir_warmup(inputs)`

Drive without recording. Shape `(num_steps,)` or `(num_steps, num_inputs)`.
Converted to float32.

##### `reservoir_run(inputs, *, clear_recorded=False)`

Drive and **append** recorded readout inputs. `clear_recorded=True` discards
prior rows and any cached `fit()` targets first (live reservoir and trained
readout stay).

##### `reservoir_clear()`

Zero reservoir dynamics. Recorded rows and readout weights are preserved.

---

#### Training

##### `train(targets)`

Batch-fit the HCNN on the prefix of collected rows.

- **Regression:** `(train_size,)` or flat `(train_size * num_outputs,)`.
- **Classification:** `(train_size,)` float class indices.

Sample count is `len(targets) // num_outputs`. Raises if that exceeds
`num_collected_states` or if length is not a multiple of `num_outputs`.

A second `train()` **continues** from current weights (same as C++). Construct a
new `ESN` for a fresh random init. Architecture / schedule changes also require a
new instance.

---

#### Prediction and evaluation

##### `predict() → ndarray`

Live prediction from the current reservoir / readout input. Shape
`(num_outputs,)`. Softmax is **not** applied (classification returns logits).

##### `predict_from_recorded(timestep) → ndarray`

One recorded row. Shape `(num_outputs,)`.

##### `predictions() → ndarray`

All recorded rows: `(num_collected_states, num_outputs)`.

##### `predict_from_state(state) → ndarray`

Forward a caller-supplied vector of length **`reservoir_neuron_count`** (equals
readout input width while Python keeps B = 1). Shape `(num_outputs,)`.

##### `r2` / `nrmse` / `accuracy`

```python
esn.r2()                              # after fit(): test window
esn.r2(targets)                       # all collected
esn.r2(targets, start=1400)           # from 1400 to end
esn.r2(targets, start=0, count=1400)
```

Same argument conventions for `nrmse` and `accuracy`.

- **Targets** are index-aligned with collected states — pass the full array and
  use `start` / `count`. Do **not** pre-slice targets.

```python
esn.r2(targets, start=1400)   # correct
esn.r2(targets[1400:])        # wrong alignment
```

- **R²:** average of per-output coefficients of determination.
- **NRMSE:** RMSE / std(target), averaged over outputs (C++ semantics).
- **Accuracy:** multi-class = argmax match; single-output thresholds logit at 0
  (labels typically in {−1, +1} for the binary head).

---

#### State access

##### `collected_states() → ndarray`

Shape `(num_collected_states, reservoir_neuron_count)` while B = 1 (each row is
the recorded readout input).

##### `copy_reservoir_state() → ndarray`

Newest slice: `(reservoir_neuron_count,)`. With B = 1 this is also the row shape
for `train_step_batch` / `predict_from_state`.

---

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `reservoir_hypercube_dimension` | `int` | Reservoir dim |
| `reservoir_neuron_count` | `int` | N = 2<sup>dim</sup> |
| `num_collected_states` | `int` | Rows from `reservoir_run` |
| `num_outputs` | `int` | Readout width |
| `num_inputs` | `int` | Input channels |
| `history_depth` | `int` | M (recurrent delay line) |
| `seed` | `int` | Reservoir master seed |
| `spectral_radius` | `float` | Configured SR **target** |
| `leak_rate` | `float` | Leak coefficient |
| `input_scaling` | `float` | Input drive scale |
| `verbose` | `bool` | Construction banner flag |
| `train_size` | `int \| None` | From `fit()`, else `None` |
| `test_size` | `int \| None` | From `fit()`, else `None` |
| `readout_best_epoch` | `int` | 1-based best epoch after restore, else 0 |

---

#### Streaming / online training

| Method | Description |
|--------|-------------|
| `reservoir_warmup(inputs)` | Settle before online steps |
| `train_step(target, lr, weight_decay=0.0)` | One step on the live state. Regression: `(num_outputs,)`; classification: one class index |
| `train_step_batch(states, targets, lr, weight_decay=0.0)` | Mini-batch. `states`: `(count, reservoir_neuron_count)` (B = 1). Targets: `(count, num_outputs)` or `(count,)` class indices |
| `copy_reservoir_state()` | Newest slice for batch accumulation |
| `predict()` | Live prediction |

`readout_epochs` is ignored in online mode — the caller owns the schedule
(e.g. cosine on `lr`).

---

#### HCNN-native export

| Method | Description |
|--------|-------------|
| `save_readout_hcnn_model(path_stem)` | Write `stem.hcnw` + `stem.arch.json` |
| `load_readout_hcnn_model(path_stem, *, mode="eval")` | Load; `mode` is `"eval"` or `"resume_train"` |
| `readout_arch_summary()` | Human-readable stack + parameter counts |

---

## Input data layout

Row-major, C-contiguous float32 preferred.

**Single-input** (`num_inputs=1`):

```python
inputs = signal[200:400]  # shape (200,)
```

**Multi-input** (`num_inputs=K`):

```python
inputs = np.column_stack([ch1, ch2, ch3])  # shape (num_steps, 3)
```

Flattened internally as `[step0_ch0, step0_ch1, …, step1_ch0, …]`. Channel k
drives vertices `[k·N/K, (k+1)·N/K)`.

Any numeric dtype is converted to C-contiguous float32 at the boundary.

---

## Data types

The C++ core is **float32** throughout (weights, states, features, readout).
Inputs and targets are converted automatically. Pre-cast hot paths if you want
to skip conversion:

```python
signal = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)
```

---

## Error handling

- **`ValueError`** — dim outside 5–16; bad `train_size`; targets length not a
  multiple of `num_outputs`; input size not divisible by `num_inputs`; invalid
  `readout_activation`; missing targets when evaluating without `fit()`; etc.
- **`IndexError` / range errors** — `predict_from_recorded` past the buffer;
  `r2` / `nrmse` / `accuracy` window past `num_collected_states`.

Validation runs at the Python / pybind boundary before C++ work.

---

## Model persistence

Reservoir weights are deterministic from config + seed; pickle stores config and
the trained readout. Files are compact (typically under 1 MB).

##### `esn.save(path)` / `ESN.load(path)`

Standard pickle:

```python
esn.fit(signal, warmup=200)
esn.save("model.pkl")

loaded = he.ESN.load("model.pkl")
loaded.reservoir_warmup(new_signal[:200])
loaded.reservoir_run(new_signal[200:])
preds = loaded.predictions()
```

Also works with `pickle.dumps` / `pickle.loads`. Persistence format version is
internal (`_PERSISTENCE_VERSION`); newer pickles on older installs raise a clear
upgrade error.

| Saved | Not saved |
|-------|-----------|
| Constructor parameters (reservoir + `readout_*`) | Collected states |
| Trained readout weights | `fit()` targets and train/test split |

Restored ESNs have zero collected states — re-drive before recorded prediction
APIs.

Portable HCNN-only export (no full ESN pickle): `save_readout_hcnn_model` /
`load_readout_hcnn_model` (architecture must match).

---

## Limitations

- **No scikit-learn estimator protocol.** The ESN is a temporal pipeline (order
  matters, warmup required, states accumulate). Row-shuffled CV would destroy
  that structure.
- **HCNN shape knobs still partial in Python.** Pooling / BN / optimizer /
  channel growth stay at C++ defaults (pooling on, Adam). Multi-slice B,
  external feedback, and bias_scaling **are** exposed.
- **Scoring buffers:** `r2` / `nrmse` / `accuracy` need a targets array that
  covers index **0 through start+count−1** (not a window slice alone).
- **Defaults (2.0):** `verbose=False`, `readout_num_layers=1` (0 = auto),
  `readout_slices=1`, external feedback off. See [CHANGELOG.md](../CHANGELOG.md).

---

## Dependencies

**Runtime:** NumPy ≥ 1.21

**Build:** scikit-build-core ≥ 0.10, pybind11 ≥ 2.13 (wheels use pybind11 ≥ 3.0),
C++23 compiler, CMake 3.20+. HypercubeCNN is vendored in-tree and linked
statically into the extension — no separate install for PyPI wheels or from-source
builds.
