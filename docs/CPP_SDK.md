# HypercubeESN C++ SDK

Static C++ library for reservoir computing on Boolean hypercube graphs: a fixed
`Reservoir` plus a trainable HypercubeCNN `Readout`, wrapped by `ESN`.

Deep dives: [Reservoir.md](Reservoir.md) · [Readout.md](Readout.md).

## Contents

- [What's in the SDK](#whats-in-the-sdk)
- [Building from source](#building-from-source)
- [Using the SDK](#using-the-sdk)
- [Pipeline vocabulary](#pipeline-vocabulary)
- [API Reference](#api-reference)
  - [Hypercube dimension: dim](#hypercube-dimension-dim)
  - [Enums](#enums)
  - [ReservoirConfig](#reservoirconfig)
  - [ReadoutConfig](#readoutconfig)
  - [ESNConfig](#esnconfig)
  - [ESN](#esn)
- [Dependencies](#dependencies)

## What's in the SDK

After installation:

```
<prefix>/
  include/HypercubeESN/
    ESN.h              -- public API (the only header consumers need)
    Reservoir.h        -- included by ESN.h
    Readout.h          -- types used by the ESN API (ReadoutConfig, enums)
  lib/
    libHypercubeESNCore.a
  lib/cmake/HypercubeESN/
    HypercubeESNConfig.cmake
    HypercubeESNTargets.cmake
    HypercubeESNConfigVersion.cmake
```

Include `<HypercubeESN/ESN.h>` (installed) or `"ESN.h"` (FetchContent) and link
`HypercubeESN::HypercubeESNCore` (or `HypercubeESNCore` in FetchContent builds).
`Reservoir.h` / `Readout.h` come along transitively; their public types are part
of the API surface.

The convolutional readout comes from **HypercubeCNN**, vendored at
`third_party/HypercubeCNN`. HypercubeESNCore links it transitively — consumers
do not name it. See [Dependencies](#dependencies).

## Building from source

Requirements: C++23 (GCC 13+, Clang 17+, MSVC 2022+), CMake 4.1+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /path/to/sdk
```

## Using the SDK

### CMake FetchContent (recommended)

```cmake
cmake_minimum_required(VERSION 4.1)
project(MyApp)

set(CMAKE_CXX_STANDARD 23)

include(FetchContent)
FetchContent_Declare(
    HypercubeESN
    GIT_REPOSITORY https://github.com/dliptak001/HypercubeESN.git
    GIT_TAG        v1.4.0   # pin a release tag; check GitHub for latest
)
FetchContent_MakeAvailable(HypercubeESN)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE HypercubeESNCore)
```

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Pin `GIT_TAG` to a release for reproducible builds. Include paths are set
automatically — `#include "ESN.h"`.

HypercubeCNN is vendored in-tree; no sibling checkout or network fetch.

### Installed SDK (`find_package`)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /path/to/sdk
```

```cmake
cmake_minimum_required(VERSION 4.1)
project(MyApp)

set(CMAKE_CXX_STANDARD 23)

find_package(HypercubeESN REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE HypercubeESN::HypercubeESNCore)
```

```bash
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/sdk
cmake --build build
```

### Minimal example

FetchContent-style include (`"ESN.h"`). Installed SDK: `<HypercubeESN/ESN.h>`.

```cpp
#include "ESN.h"
#include <cmath>
#include <vector>
#include <iostream>

int main()
{
    constexpr size_t dim = 7;         // N = 128 (= 2⁷) neurons
    constexpr size_t warmup = 200;
    constexpr size_t collect = 2000;

    std::vector<float> signal(warmup + collect + 1);
    for (size_t t = 0; t < signal.size(); ++t)
        signal[t] = std::sin(0.1f * static_cast<float>(t));

    ESNConfig cfg;
    cfg.reservoir.dim  = dim;              // hypercube dimension (5-16)
    cfg.reservoir.seed = 74119;            // per-task surveyed seed
    cfg.readout.epochs     = 25;
    cfg.readout.batch_size = 128;
    cfg.readout.lr_max     = 0.003f;
    // cfg.readout_slices = 1;             // default: newest reservoir slice only
    ESN esn(cfg);

    esn.ReservoirWarmup(signal.data(), warmup);
    esn.ReservoirRun(signal.data() + warmup, collect);

    std::vector<float> targets(collect);
    for (size_t t = 0; t < collect; ++t)
        targets[t] = signal[warmup + t + 1];  // next-step targets

    size_t train_size = 1400;
    size_t test_size = collect - train_size;

    esn.Train(targets.data(), train_size);

    double r2 = esn.R2(targets.data(), train_size, test_size);
    std::cout << "R2: " << r2 << "\n";

    return 0;
}
```

---

## Pipeline vocabulary

```
  inputs [+ optional ext-fb]
       │
       ▼
  Reservoir (fixed)
       │
       ▼
  SliceAt(0 .. B-1)  ── pack B blocks of N ──▶  HCNN readout (trained) ──▶ y
```

Only the readout emits **y**. External feedback is an *input* into the reservoir
(caller-owned closed-loop drive), not a second path to **y**.

| Term | Meaning |
|------|---------|
| **Timestep** | One `ReservoirStep` |
| **N** | Reservoir neurons = 2<sup>dim</sup> (`ReservoirNeuronCount`) |
| **M** | `history_depth` — delay-line depth the recurrent gather uses |
| **B** | `readout_slices` — power of two, 1 ≤ B ≤ M; ages packed into the readout |
| **Reservoir state** | Newest slice only (`Outputs` / `CopyReservoirState`) — N floats |
| **Readout input** | What the HCNN sees: B blocks of N (`ReadoutInputWidth`) |
| **Open loop** | Task input only |
| **Closed loop** | Also stage `external_feedback` on the reservoir |

**Not thread-safe.** Const predict paths share a scratch buffer — one ESN per
thread.

---

## API Reference

### Hypercube dimension: `dim`

`ReservoirConfig::dim` sets the reservoir hypercube size. N = 2<sup>dim</sup> neurons at
construction. Valid range **[5, 16]** — out of range throws
`std::invalid_argument`. One concrete `Reservoir` / `ESN` type serves every
dimension (no per-dim templates).

| dim   | Neurons     | Typical use |
|-------|-------------|-------------|
| 5     | 32          | Fast prototyping, embedded |
| 6     | 64          | Light benchmarks |
| 7     | 128         | Standard benchmarks |
| 8     | 256         | Production, complex tasks |
| 9–12  | 512–4096    | Research, high-capacity tasks |
| 13–16 | 8192–65536  | Large-scale research |

When `readout_slices = B > 1`, the HCNN start dimension is
`reservoir.dim + log2(B)` (set by ESN — do not set `readout.dim` yourself).

### Enums

Declared in `Readout.h`.

#### `ReadoutTask`

| Value | Description |
|-------|-------------|
| `Regression` | MSE loss. Raw network outputs at inference (no automatic target centering). `num_outputs` = number of targets. |
| `Classification` | Softmax + cross-entropy in the **loss** only. `num_outputs` = number of classes. Targets are float class indices; `Predict` returns raw logits (argmax for the label). |

#### `ReadoutActivation`

Per-Conv activation (`ReadoutConfig::activation`).

| Value | Description |
|-------|-------------|
| `TANH` | Hyperbolic tangent (default) |
| `RELU` | Rectified linear |
| `LEAKY_RELU` | Leaky rectified linear |
| `NONE` | Identity |

#### `ReadoutPoolType`

| Value | Description |
|-------|-------------|
| `Max` | Antipodal max pool (default when pooling is on) |
| `Avg` | Antipodal average pool |

#### `ReadoutOptimizer`

| Value | Description |
|-------|-------------|
| `Adam` | Default |
| `Sgd` | Heavy-ball SGD; uses `momentum` |

#### `ReadoutLoadMode`

| Value | Description |
|-------|-------------|
| `Eval` | Load parameters only (default; safe for inference) |
| `ResumeTrain` | Also reset optimizer moments for continued online training |

---

### ReservoirConfig

Construction-time reservoir parameters. Defaults are a sensible starting point;
production callers set dim, seed, spectral radius, and history depth per task
(surveyed offline).

```cpp
struct ReservoirConfig
{
    size_t   dim             = 10;     // N = 1 << dim; range [5, 16]
    uint64_t seed            = 73895;
    float    spectral_radius = 0.99f;  // target for recurrent block only
    float    leak_rate       = 1.0f;   // (0, 1]
    float    input_scaling   = 0.5f;   // weights × scaling/√dim
    size_t   num_inputs      = 1;      // must divide N
    size_t   history_depth   = 16;     // M in [1, 64]
    bool     verbose         = false;  // construction banner; demos may set true

    size_t   num_external_feedback_channels = 0;  // 0 = off; else [1, N]
    float    external_feedback_scaling      = 0.5f;

    float    bias_scaling    = 0.02f;  // after tanh; 0 disables
};
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dim` | `size_t` | `10` | Hypercube dimension; N = 2<sup>dim</sup>. **[5, 16]**. |
| `seed` | `uint64_t` | `73895` | Master RNG seed (SplitMix64 substreams: recurrent / input / external-feedback / bias / SR probe). Screen per dim/task. |
| `spectral_radius` | `float` | `0.99` | Target ρ of the **recurrent** companion operator (MN×MN when M > 1). Drive ports are outside the rescale. |
| `leak_rate` | `float` | `1.0` | `state = (1 − leak) * old + leak * (tanh(s) + bias)`. **(0, 1]**. |
| `input_scaling` | `float` | `0.5` | Input weights U(−1,1) then × `input_scaling / √dim` (fan-in variance). Local construction, not a universal optimum. Typical O(0.5–3). |
| `num_inputs` | `size_t` | `1` | Input channels; must divide N. Channel k drives `[k·N/K, (k+1)·N/K)`. |
| `history_depth` | `size_t` | `16` | Delay-line depth M **[1, 64]**. Recurrent gather over M published slices. Independent of how many ages the readout packs (B). See [Reservoir.md](Reservoir.md). |
| `verbose` | `bool` | `false` | One construction banner on stdout. |
| `num_external_feedback_channels` | `size_t` | `0` | D external-feedback channels. **0** = path off. Else **[1, N]** (need not divide N). See [reservoir_feedback_mechanism.md](reservoir_feedback_mechanism.md). |
| `external_feedback_scaling` | `float` | `0.5` | Like input; only if D > 0. Outside SR rescale. |
| `bias_scaling` | `float` | `0.02` | Per-neuron bias U(−1,1)×scale, **after** tanh. **0** disables. Survives `Clear`; not in snapshots. |

`GetConfig().spectral_radius` / `ESN::TargetSpectralRadius()` is the **target**.
Post-secant estimate: `Reservoir::GetRealizedSpectralRadius()` /
`ESN::RealizedSpectralRadius()`.

---

### ReadoutConfig

HCNN architecture and training. Under ESN, `dim` is overwritten to
`reservoir.dim + log2(B)` — leave it at 0.

```cpp
struct ReadoutConfig {
    size_t dim           = 0;        // set by ESN — do not set
    int num_outputs      = 1;
    ReadoutTask task     = ReadoutTask::Regression;
    int num_layers       = 1;        // typical; 0 = auto min(dim-2, 2)
    bool use_pooling     = true;
    ReadoutPoolType pool_type = ReadoutPoolType::Max;
    int conv_channels    = 16;
    int channel_growth   = 2;
    bool use_batchnorm   = false;
    ReadoutOptimizer optimizer = ReadoutOptimizer::Adam;
    int epochs           = 200;
    int batch_size       = 32;
    float lr_max         = 0.0015f;  // keep ≤ ~0.005
    float lr_min_frac    = 0.01f;
    int   lr_decay_epochs = 0;       // 0 = use epochs
    float weight_decay   = 0.0f;
    float momentum       = 0.0f;     // SGD only
    unsigned seed        = 42;
    ReadoutActivation activation = ReadoutActivation::TANH;
    size_t num_threads   = 0;        // 0=auto, 1=ST, N=N workers
    bool restore_best_epoch = true;
    float best_epoch_holdout_frac = 0.0f;
};
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dim` | `size_t` | `0` | Features per sample = 2<sup>dim</sup>. **Set by ESN** from reservoir dim + log2(B). |
| `num_outputs` | `int` | `1` | Regression targets or class count. |
| `task` | `ReadoutTask` | `Regression` | Task head. |
| `num_layers` | `int` | `1` | Conv(+Pool) stages. Default **1** (house default for most tasks). `0` → auto `min(dim − 2, 2)`. With pooling: assert `n ≤ dim − 2`. |
| `use_pooling` | `bool` | `true` | Antipodal pool after each conv (mixes every bit, including block-index bits when B > 1). |
| `pool_type` | `ReadoutPoolType` | `Max` | Max or Avg when pooling is on. |
| `conv_channels` | `int` | `16` | First-layer channels. |
| `channel_growth` | `int` | `2` | Multiplier after each stage. |
| `use_batchnorm` | `bool` | `false` | Per-conv BN; grows the weight blob. |
| `optimizer` | `ReadoutOptimizer` | `Adam` | Forwarded to HypercubeCNN. |
| `epochs` | `int` | `200` | Batch-train epochs. Ignored by online `TrainStep*`. |
| `batch_size` | `int` | `32` | Mini-batch size (batch mode). |
| `lr_max` | `float` | `0.0015` | Cosine peak. **Keep ≤ ~0.005** to avoid NaN. |
| `lr_min_frac` | `float` | `0.01` | Floor = `lr_max * lr_min_frac`. |
| `lr_decay_epochs` | `int` | `0` | Cosine horizon; `0` = use `epochs`. |
| `weight_decay` | `float` | `0.0` | L2 weight decay. |
| `momentum` | `float` | `0.0` | SGD only; ignored by Adam. |
| `seed` | `unsigned` | `42` | HCNN weight init seed. |
| `activation` | `ReadoutActivation` | `TANH` | After each Conv. |
| `num_threads` | `size_t` | `0` | HCNN workers: `0` auto, `1` single-threaded (use for multi-ESN hosts), `N` workers. |
| `restore_best_epoch` | `bool` | `true` | Restore best epoch (min MSE / max accuracy) at end of `Train`. |
| `best_epoch_holdout_frac` | `float` | `0.0` | Tail hold-out for scoring; train on prefix. `0` = score full train set. Clamped to [0, 0.5]. |

See [Readout.md](Readout.md) and the vendor pin in
[../third_party/HypercubeCNN/VENDORED.md](../third_party/HypercubeCNN/VENDORED.md).

---

### ESNConfig

```cpp
struct ESNConfig {
    ReservoirConfig reservoir;
    ReadoutConfig   readout;
    // B ages packed into the readout (power of two, 1 ≤ B ≤ history_depth).
    // ESN sets readout.dim = reservoir.dim + log2(B).
    size_t          readout_slices = 1;
};
```

| Field | Description |
|-------|-------------|
| `reservoir` | Fixed dynamical core. |
| `readout` | HCNN architecture + training. Leave `dim` at 0. |
| `readout_slices` | B delay-line ages (newest first). Must be ≥ 1, a **power of two**, and ≤ `reservoir.history_depth`. B = 1 → readout input is one N-vector. B > 1 → multi-block packing on a larger cube (2-bit block map so consecutive ages share an HCNN filter). Widening B does **not** change reservoir dynamics. |

---

### ESN

Complete pipeline: Reservoir → pack B slices → Readout. Constructed from one
`ESNConfig`. Readout hyperparameters are fixed at construction — no per-call
config overloads on `Train`.

```cpp
ESN esn(cfg);

// Drive
esn.ReservoirStep(inputs, external_feedback /* optional */);
esn.ReservoirWarmup(inputs, num_steps);
esn.ReservoirRun(inputs, num_steps);
esn.ReservoirRun(inputs, num_steps, /*clear_recorded=*/true);
esn.ReservoirClear();

// Batch train / score on recorded readout inputs
esn.Train(targets, train_size);
esn.R2(targets, start, count);
esn.NRMSE(targets, start, count);
esn.Accuracy(labels, start, count);

// Streaming
esn.TrainStep(target, lr, weight_decay);
esn.TrainStepBatch(readout_inputs, targets, count, lr, weight_decay);
esn.CopyReadoutInput(out);      // B×N
esn.CopyReservoirState(out);    // N only

// Predict
esn.Predict();
esn.PredictFromRecorded(timestep);
esn.PredictFromState(readout_input);

// Persist
esn.GetConfig();
esn.GetReadoutState();
esn.SetReadoutState(state, mode);
esn.SaveReadoutHcnnModel(stem);
esn.LoadReadoutHcnnModel(stem, mode);
esn.ReadoutArchSummary();
esn.ReadoutBestEpoch();
```

#### Construction

```cpp
explicit ESN(const ESNConfig& cfg);
```

Builds the reservoir (`Create`) and the HCNN eagerly (`MakeReadoutConfig` fills
`readout.dim`). Both weight sets are ready before the first `Train` / `TrainStep`.

```cpp
ESNConfig cfg;
cfg.reservoir.dim             = 8;
cfg.reservoir.seed            = 74119;
cfg.reservoir.spectral_radius = 0.99f;
cfg.readout_slices            = 1;       // or 2, 4, … ≤ history_depth
cfg.readout.epochs     = 1000;
cfg.readout.batch_size = 512;
cfg.readout.lr_max     = 0.001f;
ESN esn(cfg);
```

---

#### Reservoir driving

##### `ReservoirStep`

Pointer and `std::span` overloads (span form validates lengths):

```cpp
void ReservoirStep(const float* inputs, const float* external_feedback = nullptr);
void ReservoirStep(std::span<const float> inputs,
                   std::span<const float> external_feedback = {});
```

One timestep: stage task `inputs` (`NumInputs()` floats), optionally stage
**external** feedback (`NumExternalFeedbackChannels()` floats, or `nullptr` to
skip), then `Reservoir::Step`. No learning.

Throws if `external_feedback` is non-null when D = 0.

##### `ReservoirWarmup`

```cpp
void ReservoirWarmup(const float* inputs, size_t num_steps);
```

Drive without recording (wash out zero initial state). Layout: `num_steps ×
NumInputs()` row-major. **No** external feedback — use `ReservoirStep` if needed.
Typical warmup: 100–500 steps.

Values are **not** clamped; pass already-bounded signals.

##### `ReservoirRun`

```cpp
void ReservoirRun(const float* inputs, size_t num_steps, bool clear_recorded = false);
```

Drive and **append** each assembled readout input (B×N) to the internal buffer
for `Train` / metrics. Same input layout as warmup. No external feedback.

`clear_recorded = true` discards prior rows first (live reservoir and readout
weights untouched).

##### `ReservoirClear`

```cpp
void ReservoirClear();
```

Zero reservoir dynamics (state + history). Recorded rows and readout weights are
preserved.

---

#### Batch training

##### `Train`

```cpp
void Train(const float* targets, size_t train_size);
```

Fit the HCNN on recorded timesteps `[0, train_size)`. A second call **continues**
from current weights — construct a new ESN for a fresh init.

- **Regression:** `train_size × NumOutputs()` floats, row-major.
- **Classification:** `train_size` floats (class indices).

Throws if `train_size > NumCollectedStates()`.

---

#### Streaming training

CNN is built at construction — no separate init. Warm up the reservoir, then
interleave `ReservoirStep` with `TrainStep` / `Predict`. `epochs` is ignored;
loop length is the caller's.

##### `TrainStep`

```cpp
void TrainStep(const float* target, float lr, float weight_decay = 0.0f);
```

One gradient step on the **current** readout input (assembled after your last
drive). Regression: `NumOutputs()` floats. Classification: one class-index float.

##### `TrainStepBatch`

```cpp
void TrainStepBatch(const float* readout_inputs, const float* targets, size_t count,
                    float lr, float weight_decay = 0.0f);
```

Mini-batch of caller-supplied **readout inputs** (`count × ReadoutInputWidth()`).
Assemble rows with `CopyReadoutInput` (not `CopyReservoirState`, unless B = 1).

##### `CopyReadoutInput` / `CopyReservoirState`

```cpp
void CopyReadoutInput(float* out) const;     // ReadoutInputWidth() = B×N
void CopyReservoirState(float* out) const;   // N (newest slice only)
```

---

#### Prediction and evaluation

##### Recorded window

```cpp
std::vector<float> PredictFromRecorded(size_t timestep) const;
double R2(const float* targets, size_t start, size_t count) const;
double NRMSE(const float* targets, size_t start, size_t count) const;
double Accuracy(const float* labels, size_t start, size_t count) const;
```

- **Targets** must cover `[0, start+count)` — pass the full array; methods index
  from `start`. Do not pre-slice.
- **R²:** average of per-output coefficients of determination. 1.0 = perfect.
- **NRMSE:** mean over outputs of RMSE / std(target). 0 = perfect. Degenerate
  target variance → +inf on that output.
- **Accuracy:** multi-class argmax; single-output thresholds the logit at 0.

##### Live / caller-supplied

```cpp
std::vector<float> Predict() const;                    // assemble live, then forward
void Predict(float* out) const;
std::vector<float> PredictFromState(const float* readout_input) const;
void PredictFromState(const float* readout_input, float* out) const;
```

`PredictFromState` never reads the reservoir — pass a `ReadoutInputWidth()` buffer
(e.g. from `CopyReadoutInput`). Softmax is **not** applied; classification returns
logits.

---

#### State access and accessors

```cpp
std::vector<float> CollectedStates() const;  // T × ReadoutInputWidth(), row-major
```

| Method | Returns |
|--------|---------|
| `NumCollectedStates()` | Rows recorded by `ReservoirRun` |
| `NumInputs()` | Input channels per timestep |
| `NumOutputs()` | Readout width (targets or classes) |
| `NumExternalFeedbackChannels()` | D (0 = no ext-fb port) |
| `ReservoirHypercubeDimension()` | `cfg.reservoir.dim` |
| `ReservoirNeuronCount()` | N = 2<sup>dim</sup> |
| `ReadoutInputWidth()` | B × N |
| `ReadoutBlockCount()` | B |
| `ReadoutBlockOf(slot)` | Physical block index for logical age `slot` |
| `GetConfig()` | `ESNConfig` with derived `readout.dim` filled |

---

#### Readout persistence

Reservoir weights are deterministic from config + seed. Persist `GetConfig()` and
the readout.

**`ESN::ReadoutState`**

| Field | Description |
|-------|-------------|
| `weights` | Opaque `vector<double>` (unversioned HCNN blob). Round-trip only. |
| `is_trained` | True if the network exists (true after construction — not “has seen data”). |

| Method | Description |
|--------|-------------|
| `GetReadoutState()` | Snapshot weights |
| `SetReadoutState(state, mode=Eval)` | Inject into the live net. No-op if `!is_trained` |
| `ReadoutBestEpoch()` | 1-based best epoch after last batch `Train` with restore, else 0 |
| `SaveReadoutHcnnModel(stem)` | Portable `stem.hcnw` + `stem.arch.json` |
| `LoadReadoutHcnnModel(stem, mode=Eval)` | Load after arch sidecar validation |
| `ReadoutArchSummary()` | Human-readable stack + parameter counts |

```cpp
ESNConfig cfg   = esn.GetConfig();
auto      state = esn.GetReadoutState();
// serialize cfg + state …

ESN restored(cfg);
restored.SetReadoutState(state);
```

Standalone reservoir: `Reservoir::Create(cfg)` / `GetConfig()` /
`GetRealizedSpectralRadius()`.

---

## Dependencies

**HypercubeCNN** — hypercube convolutional stack used by `Readout`.

- Vendored read-only snapshot at `third_party/HypercubeCNN` (see `VENDORED.md`).
- Built transitively via `add_subdirectory`; offline and version-pinned.
- Linked through HypercubeESNCore — consumers only link HypercubeESNCore.
- Public HypercubeESN surface is ESN / Reservoir / Readout types; full HCNN API
  is not re-exported (`hcnn::HCNN` is PIMPL'd inside `Readout`).

No other external dependencies beyond the C++ standard library.
