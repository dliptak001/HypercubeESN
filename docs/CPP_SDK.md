# HypercubeESN C++ SDK

Static C++ library for reservoir computing on Boolean hypercube graphs.

## Contents

- [What's in the SDK](#whats-in-the-sdk)
- [Building from source](#building-from-source)
- [Using the SDK](#using-the-sdk)
  - [CMake FetchContent (recommended)](#cmake-fetchcontent-recommended)
  - [Installed SDK (find_package)](#installed-sdk-find_package)
- [API Reference](#api-reference)
  - [Hypercube dimension: dim](#hypercube-dimension-dim)
  - [Enums](#enums)
  - [ReservoirConfig](#reservoirconfig)
  - [ReadoutConfig](#readoutconfig)
  - [ESN](#esn)
- [Dependencies](#dependencies)

## What's in the SDK

After installation, the SDK contains:

```
<prefix>/
  include/HypercubeESN/
    ESN.h              -- The public API (the only header consumers include)
    Reservoir.h        -- Internal: included by ESN.h
    Readout.h      -- Transitive: types used by the ESN API (ReadoutConfig, ReadoutTask)
  lib/
    libHypercubeESNCore.a
  lib/cmake/HypercubeESN/
    HypercubeESNConfig.cmake
    HypercubeESNTargets.cmake
    HypercubeESNConfigVersion.cmake
```

Consumers include `<HypercubeESN/ESN.h>` (installed SDK) or `"ESN.h"` (FetchContent) and link against `HypercubeESN::HypercubeESNCore`. The other headers are present because ESN.h includes them; there is no need to include them directly, but their public types (`ReadoutConfig`, `ReadoutTask`) are part of the API surface.

The SDK depends on a second static library — **HypercubeCNN** — that provides the convolutional readout. It is vendored in-tree at `third_party/HypercubeCNN`, so no sibling checkout or network fetch is needed. HypercubeESNCore transitively links to it, so consumers don't need to reference it explicitly. See [Dependencies](#dependencies).

## Building from source

Requirements: C++23 compiler (GCC 13+, Clang 17+, MSVC 2022+), CMake 4.1+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /path/to/sdk
```

## Using the SDK

### CMake FetchContent (recommended)

The simplest way to use HypercubeESN in a CMake project. No installation, no
manual downloads -- CMake pulls the source from GitHub and builds it alongside
your project.

```cmake
cmake_minimum_required(VERSION 4.1)
project(MyApp)

set(CMAKE_CXX_STANDARD 23)

include(FetchContent)
FetchContent_Declare(
    HypercubeESN
    GIT_REPOSITORY https://github.com/dliptak001/HypercubeESN.git
    GIT_TAG        v1.0.0
)
FetchContent_MakeAvailable(HypercubeESN)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE HypercubeESNCore)
```

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Pin `GIT_TAG` to a release tag (e.g., `v1.0.0`) for reproducible builds.
Include paths are set automatically -- just `#include "ESN.h"`.

**Note:** HypercubeCNN is vendored in-tree at `third_party/HypercubeCNN`, so it
is fetched along with HypercubeESN — no sibling checkout or extra configuration.
See [Dependencies](#dependencies).

### Installed SDK (find_package)

If you prefer to install the library once and link against it:

```bash
# Build and install
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

Configure with the SDK path:

```bash
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/sdk
cmake --build build
```

### Minimal example

This example uses FetchContent-style includes (`"ESN.h"`). For an installed SDK,
use `<HypercubeESN/ESN.h>` instead.

```cpp
#include "ESN.h"
#include <cmath>
#include <vector>
#include <iostream>

int main()
{
    constexpr size_t DIM = 7;         // 2^7 = 128 neurons
    constexpr size_t warmup = 200;
    constexpr size_t collect = 2000;

    // Generate a sine wave
    std::vector<float> signal(warmup + collect + 1);
    for (size_t t = 0; t < signal.size(); ++t)
        signal[t] = std::sin(0.1f * static_cast<float>(t));

    // ESN config bundles reservoir + readout. Start from the struct
    // defaults, set the surveyed reservoir values + readout hyperparameters.
    ESNConfig cfg;
    cfg.reservoir.dim  = DIM;              // hypercube dimension (5-16)
    cfg.reservoir.seed = 74119;            // per-task surveyed seed
    cfg.readout.epochs     = 25;
    cfg.readout.batch_size = 128;
    cfg.readout.lr_max     = 0.003f;
    ESN esn(cfg);

    // Drive and train
    esn.Warmup(signal.data(), warmup);
    esn.Run(signal.data() + warmup, collect);

    std::vector<float> targets(collect);
    for (size_t t = 0; t < collect; ++t)
        targets[t] = signal[warmup + t + 1];  // predict next value

    size_t train_size = 1400;
    size_t test_size = collect - train_size;

    esn.Train(targets.data(), train_size);

    double r2 = esn.R2(targets.data(), train_size, test_size);
    std::cout << "R2: " << r2 << "\n";

    return 0;
}
```

---

## API Reference

### Hypercube dimension: `dim`

`dim` is a runtime `ReservoirConfig` field controlling the hypercube dimension. The reservoir has N = 2^dim neurons, sized at construction. Valid range is **dim 5-16**, enforced by `Reservoir::Create` (out-of-range throws `std::invalid_argument`). A single concrete `Reservoir`/`ESN` type serves every dimension — there are no longer per-DIM template instantiations.

| dim   | Neurons     | Typical use |
|-------|-------------|-------------|
| 5     | 32          | Fast prototyping, embedded |
| 6     | 64          | Light benchmarks |
| 7     | 128         | Standard benchmarks |
| 8     | 256         | Production, complex tasks |
| 9-12  | 512-4096    | Research, high-capacity tasks |
| 13-16 | 8192-65536  | Large-scale research |

### Enums

#### `ReadoutTask`

Task head for the HCNN readout. Declared in `Readout.h`.

| Value | Description |
|-------|-------------|
| `Regression` | MSE loss, de-centered predictions. `num_outputs` sets the number of regression targets. |
| `Classification` | Softmax + cross-entropy loss. `num_outputs` sets the number of classes. Targets are float class indices; predictions are raw logits (use `argmax`). |

#### `ReadoutActivation`

Activation applied after each Conv layer in the readout's CNN stack. Declared in `Readout.h`; set via `ReadoutConfig::activation`.

| Value | Description |
|-------|-------------|
| `TANH` | Hyperbolic tangent (default). |
| `RELU` | Rectified linear. |
| `LEAKY_RELU` | Leaky rectified linear. |
| `NONE` | Identity (no activation). |

---

### ReservoirConfig

Configuration struct for reservoir construction. The struct defaults are a sensible general starting point; production callers set the dimension, seed, spectral radius, and history depth explicitly per task (surveyed offline), then override any other fields they care about.

```cpp
struct ReservoirConfig
{
    size_t   dim             = 10;     // hypercube dimension; N = 2^dim (5-16)
    uint64_t seed            = 73895;
    float    spectral_radius = 0.99f;
    float    leak_rate       = 1.0f;
    float    input_scaling   = 0.5f;   // DIM-invariant input drive
    size_t   num_inputs      = 1;
    size_t   history_depth   = 16;
    bool     verbose         = true;
};

// Typical:
ReservoirConfig cfg;          // struct defaults
cfg.dim             = 8;      // hypercube dimension (N = 256)
cfg.seed            = 74119;  // per-task surveyed seed
cfg.spectral_radius = 0.99f;  // per-DIM/task
cfg.history_depth   = 16;     // per-task recurrent delay-line depth
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dim` | `size_t` | `10` | Hypercube dimension; the reservoir has N = 2^dim neurons, sized at construction. Must be in [5, 16] — `Reservoir::Create` throws `std::invalid_argument` otherwise. |
| `seed` | `uint64_t` | `73895` | RNG seed for weight initialization. Different seeds produce measurably different performance; screen per DIM/task and set explicitly. |
| `spectral_radius` | `float` | `0.99` | Target spectral radius of the recurrent operator (the MN×MN augmented companion operator when `history_depth` > 1). Controls the echo-state property — how quickly past inputs fade. Tune per DIM/task. |
| `leak_rate` | `float` | `1.0` | Leaky-integrator coefficient. `state = (1 - leak_rate) * old_output + leak_rate * tanh(drive)`. At 1.0, each step fully replaces state; values < 1.0 add explicit temporal carryover. |
| `input_scaling` | `float` | `0.5` | Input drive coefficient. Input weights are drawn U(-1,1) then scaled by `input_scaling / √DIM`; the `1/√DIM` fan-in normalization makes a given value deliver the same `tanh` drive at any DIM (**DIM-invariant by construction** — not the legacy fixed `0.02`, which was a readout-standardization artifact). Task-dependent, typically O(0.5–3). |
| `num_inputs` | `size_t` | `1` | Number of input channels; must divide N evenly. In multi-input mode (K channels), channel k drives the contiguous vertex block `[k*N/K, (k+1)*N/K)`. |
| `history_depth` | `size_t` | `16` | Per-vertex output-history depth M (the recurrent delay line): each `Step` sums over the M most-recent output slices, each with its own weights. Must be in [1, 64]; M = 1 is the legacy single-slice reservoir. See [Reservoir.md](Reservoir.md). |
| `verbose` | `bool` | `true` | Print the per-construction reservoir banner with the seed/leak/input-scaling and spectral-radius rescale (`[Reservoir DIM=… M=… seed=… leak=… in_scale=… SR target=… post=… (secant iters=…)]`). |

> **Note:** `output_fraction` (reservoir->readout subsampling) lives on `ESNConfig`, not `ReservoirConfig` — the reservoir does not consume it. See [ESN](#esn). `float output_fraction = 1.0` — fraction of N vertices used as readout features, in range (0.0, 1.0]; must yield a power-of-2 stride. At 0.5, a stride-selected sub-hypercube of N/2 vertices is passed to the readout. The mapping is lossy: intermediate values round down to the nearest power-of-2 stride (e.g. `0.4` → effectively `0.5`), and values that would yield a non-power-of-2 stride (e.g. `0.3`) throw. Exactly-honored values: `{1.0, 0.5, 0.25, 0.125, 0.0625, ...}`.

---

### ReadoutConfig

Configuration struct for the HCNN readout's architecture and training.

```cpp
struct ReadoutConfig {
    size_t dim           = 0;        // input feature dim = 2^dim; set internally by ESN — do not set
    int num_outputs      = 1;
    ReadoutTask task     = ReadoutTask::Regression;
    int num_layers       = 1;        // Conv+Pool pairs; 0 = auto: min(DIM-2, 2)
    int conv_channels    = 16;       // doubles per layer
    int epochs           = 200;
    int batch_size       = 32;
    float lr_max         = 0.0015f;
    float lr_min_frac    = 0.01f;
    int   lr_decay_epochs = 0;       // 0 = use epochs
    float weight_decay   = 0.0f;
    float momentum       = 0.0f;     // heavy-ball SGD momentum; 0 = plain
    unsigned seed        = 42;
    bool verbose         = false;
    bool verbose_train_acc = false;
    ReadoutActivation activation = ReadoutActivation::TANH;  // per-Conv-layer
};
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dim` | `size_t` | `0` | Input feature dimension (features per sample = 2^dim). **Set internally by the ESN** from `output_fraction` — consumers do not set this; any value here is overwritten at construction. |
| `num_outputs` | `int` | `1` | Number of output neurons. For regression: number of targets. For classification: number of classes. |
| `task` | `ReadoutTask` | `Regression` | Task head. See [ReadoutTask](#readouttask). |
| `num_layers` | `int` | `1` | Number of Conv+Pool pairs. `1` (default) builds a single Conv+Pool layer. `0` auto-computes `min(DIM - 2, 2)`. Each Pool halves the hypercube dimension, capped by `DIM - 2` (HCNNConv requires DIM >= 3). |
| `conv_channels` | `int` | `16` | Base channel count for the first Conv layer. Doubles per layer (16, 32 for a 2-layer stack). |
| `epochs` | `int` | `200` | Training epochs (batch mode). Structured signals saturate at ~25 epochs; chaotic signals need ~2000. Ignored in online mode. |
| `batch_size` | `int` | `32` | Mini-batch size (batch mode). Use 128+ on multi-core CPUs to saturate threading. |
| `lr_max` | `float` | `0.0015` | Peak learning rate for cosine annealing. **Keep <= 0.005 to avoid weight divergence into denormal/NaN territory.** |
| `lr_min_frac` | `float` | `0.01` | Cosine schedule floor as fraction of `lr_max`. Effective `lr_min = lr_max * lr_min_frac`. |
| `lr_decay_epochs` | `int` | `0` | Cosine decay horizon. 0 = use `epochs`. Set > epochs to trace only a prefix of the cosine curve (keeps lr high when shortening a run). |
| `weight_decay` | `float` | `0.0` | L2 weight decay. |
| `momentum` | `float` | `0.0` | Heavy-ball SGD momentum. `0` = plain SGD; `0.9` is typical for a wider CNN head. |
| `seed` | `unsigned` | `42` | Seed for weight initialization. |
| `verbose` | `bool` | `false` | Print per-epoch lr to stdout. |
| `verbose_train_acc` | `bool` | `false` | Also compute and print training accuracy (classification) or MSE (regression) each epoch. Costs one extra forward pass per epoch. |
| `activation` | `ReadoutActivation` | `TANH` | Activation applied after each Conv layer. See [ReadoutActivation](#readoutactivation). |

**Architecture:** the default `num_layers = 1` builds a single Conv+Pool layer (16 channels). Set `num_layers = 0` for auto-sizing — `min(DIM - 2, 2)` Conv+Pool pairs with channels doubling per layer. See [Readout.md](Readout.md) for the full design notes.

---

### ESN

The complete pipeline wrapper: Reservoir -> Readout. Constructed from a single `ESNConfig` that bundles the reservoir and readout configs (the hypercube dimension comes from `cfg.reservoir.dim`); no further config arguments are passed to `Train` / `InitOnline`.

```cpp
struct ESNConfig {
    ReservoirConfig reservoir;
    ReadoutConfig   readout;
    float           output_fraction = 1.0f;  // reservoir->readout subsampling; (0.0, 1.0], power-of-2 stride
};

// Construction (dimension comes from cfg.reservoir.dim)
ESN esn(cfg);

// Reservoir driving
esn.Warmup(inputs, num_steps);
esn.Run(inputs, num_steps);
esn.ClearStates();
esn.ResetReservoirOnly();

// Batch training (readout hyperparameters come from cfg.readout)
esn.Train(targets, train_size);

// Online (streaming) training
esn.InitOnline(warmup_inputs, warmup_count);
esn.TrainLiveStep(target_class, lr, weight_decay);
esn.TrainLiveBatch(states, targets, count, lr, weight_decay);
esn.TrainLiveStepRegression(target, lr, weight_decay);
esn.TrainLiveBatchRegression(states, targets, count, lr, weight_decay);
esn.CopyLiveState(out);

// Prediction & evaluation (collected states)
esn.PredictRaw(timestep);
esn.PredictRaw(timestep, output);
esn.R2(targets, start, count);
esn.NRMSE(targets, start, count);
esn.Accuracy(labels, start, count);

// Prediction (live reservoir state)
esn.PredictLiveRaw();
esn.PredictLiveRaw(output);
esn.PredictFromState(state, output);   // run readout on a caller-supplied state

// State access & persistence
esn.SelectedStates();
esn.NumCollected();
esn.NumOutputs();
esn.NumOutputVerts();
esn.NumInputs();
esn.Dim();                   // hypercube dimension (cfg.reservoir.dim)
esn.Size();                  // reservoir neuron count N = 2^Dim()
esn.GetConfig();             // returns ESNConfig
esn.GetReadoutState();
esn.SetReadoutState(state);
```

---

#### Construction

```cpp
explicit ESN(const ESNConfig& cfg);
```

Creates the reservoir from `cfg.reservoir` and prepares the readout with `cfg.readout`. Reservoir weights are generated and spectral-radius-rescaled at construction time; the HCNN readout itself is built when `Train()` or `InitOnline()` is called.

**Parameters:**
- `cfg` -- Full ESN configuration. See [ReservoirConfig](#reservoirconfig) and [ReadoutConfig](#readoutconfig).

The recommended construction pattern starts from the default-constructed `ESNConfig`, sets the dimension and surveyed reservoir values explicitly, then overrides task-specific readout hyperparameters:

```cpp
ESNConfig cfg;                                    // reservoir/readout defaulted
cfg.reservoir.dim             = 8;                // hypercube dimension (N = 256)
cfg.reservoir.seed            = 74119;            // surveyed seed
cfg.reservoir.spectral_radius = 0.99f;            // per-DIM/task
cfg.readout.epochs     = 1000;                    // task-driven overrides
cfg.readout.batch_size = 512;
cfg.readout.lr_max     = 0.001f;
ESN esn(cfg);
```

---

#### Reservoir Driving

##### `Warmup`

```cpp
void Warmup(const float* inputs, size_t num_steps);
```

Drives the reservoir for `num_steps` timesteps without recording state. Use this to wash out the reservoir's initial transient (zero state) before collecting data for training.

**Parameters:**
- `inputs` -- Pointer to `num_steps * num_inputs` floats, row-major. Each timestep has `num_inputs` consecutive values (one per channel). When `num_inputs == 1` (default), this is simply `num_steps` scalars. Values are **not** clamped — pass already-bounded signals (the `1/√DIM` input normalization sets the `tanh` operating point via `input_scaling`).
- `num_steps` -- Number of timesteps to drive. Typical: 100-500 depending on task.

---

##### `Run`

```cpp
void Run(const float* inputs, size_t num_steps);
```

Drives the reservoir for `num_steps` timesteps, recording the **subsampled** state at each step (`NumOutputVerts()` floats — the stride-selected vertices set by `output_fraction`, not the full N). States are appended to the internal buffer -- multiple `Run()` calls accumulate.

**Parameters:**
- `inputs` -- Pointer to `num_steps * num_inputs` floats, row-major. Same layout as `Warmup()`.
- `num_steps` -- Number of timesteps to drive and record.

---

##### `ClearStates`

```cpp
void ClearStates();
```

Clears all collected states. The reservoir's live internal state is **not** reset -- it retains its current activation. The trained readout is also preserved.

Use this between independent sequences: clear the collected data, then `Warmup()` + `Run()` on a new input sequence without rebuilding the ESN.

---

##### `ResetReservoirOnly`

```cpp
void ResetReservoirOnly();
```

Zeros the reservoir's internal state — `vtx_state_` plus every output-history slice. Recurrent weights, input weights, and all hyperparameters are untouched. Collected states are **not** cleared. The trained readout is preserved.

Use for episodic tasks where each episode starts from a clean slate (e.g., per-sequence reset).

---

#### Batch Training

##### `Train`

```cpp
void Train(const float* targets, size_t train_size);
```

Trains the HCNN readout on the first `train_size` collected states. Readout hyperparameters come from the `cfg.readout` passed to the constructor — there are no per-call config overloads.

**Parameters:**
- `targets` -- Target values. Layout depends on `cfg.readout.task`:
  - **Regression:** `train_size * cfg.readout.num_outputs` floats, row-major.
  - **Classification:** `train_size` floats (float class indices).
- `train_size` -- Number of training samples from collected state index 0.

---

#### Online (Streaming) Training

For applications where data arrives continuously. The reservoir advances one step at a time; the readout is updated via per-sample or mini-batch gradient steps.

##### `InitOnline`

```cpp
void InitOnline(const float* warmup_inputs, size_t warmup_count);
```

Initializes the HCNN readout for online training. Internally calls `Warmup()` on the warmup inputs to drive the reservoir to a representative state, then builds the CNN architecture from `cfg.readout` (passed at construction) and sets the Adam optimizer. After this call the reservoir's live state reflects having processed `warmup_count` steps; `NumCollected()` is unchanged (Warmup does not collect). Call before any `TrainLive*` method.

**Parameters:**
- `warmup_inputs` -- Warmup signal: `warmup_count * num_inputs` floats. These drive the reservoir forward without collecting state.
- `warmup_count` -- Number of warmup timesteps.

`cfg.readout.epochs` is ignored in online mode — the loop epoch count is determined by the caller's training loop.

---

##### `TrainLiveStep` (classification)

```cpp
void TrainLiveStep(float target_class, float lr, float weight_decay);
```

Single-sample online gradient step on the reservoir's current live state. Classification only -- `target_class` is cast to int internally.

---

##### `TrainLiveBatch` (classification)

```cpp
void TrainLiveBatch(const float* states, const int* targets,
                    size_t count, float lr);
void TrainLiveBatch(const float* states, const int* targets,
                    size_t count, float lr, float weight_decay);
```

Mini-batch online gradient step. `states` is `count` rows of `NumOutputVerts()` floats (from `CopyLiveState`). `targets` is `count` int class indices. Parallelized across threads. The no-`weight_decay` overload inherits `cfg.readout.weight_decay`; the explicit form overrides it for this call.

---

##### `TrainLiveStepRegression`

```cpp
void TrainLiveStepRegression(const float* target, float lr,
                             float weight_decay);
```

Single-sample online gradient step on the reservoir's current live state. `target` is `NumOutputs()` floats.

---

##### `TrainLiveBatchRegression`

```cpp
void TrainLiveBatchRegression(const float* states, const float* targets,
                              size_t count, float lr, float weight_decay);
```

Mini-batch online gradient step for regression. `states` is `count` rows of `NumOutputVerts()` floats. `targets` is `count * NumOutputs()` floats (row-major).

---

##### `CopyLiveState`

```cpp
void CopyLiveState(float* out) const;
```

Copies the current subsampled reservoir state into `out` (`NumOutputVerts()` floats). Use to accumulate states for `TrainLiveBatch` / `TrainLiveBatchRegression`.

---

#### Prediction and Evaluation (Collected States)

##### `PredictRaw` (scalar)

```cpp
[[nodiscard]] float PredictRaw(size_t timestep) const;
```

Returns the scalar prediction for a collected timestep. Asserts `NumOutputs() == 1`.

**Parameters:**
- `timestep` -- Index into collected states, in [0, NumCollected()).

---

##### `PredictRaw` (multi-output)

```cpp
void PredictRaw(size_t timestep, float* output) const;
```

Writes `NumOutputs()` floats to `output` for a collected timestep. For regression: de-centered predictions. For classification: raw logits (apply argmax for predicted class).

---

##### `R2`

```cpp
[[nodiscard]] double R2(const float* targets, size_t start, size_t count) const;
```

R-squared on collected timesteps [start, start+count).

**Parameters:**
- `targets` -- Must span timesteps [0, start+count): `(start+count) * NumOutputs()` floats (row-major). Indexed from `targets[start * NumOutputs()]`.
- `start` -- First timestep index.
- `count` -- Number of timesteps to evaluate.

**Returns:** R² averaged across outputs. 1.0 = perfect. Can be negative.

---

##### `NRMSE`

```cpp
[[nodiscard]] double NRMSE(const float* targets, size_t start, size_t count) const;
```

Normalized RMSE on collected timesteps. Same `targets` layout as `R2`.

**Returns:** NRMSE averaged across outputs. 0.0 = perfect. 1.0 = predicts the mean.

---

##### `Accuracy`

```cpp
[[nodiscard]] double Accuracy(const float* labels, size_t start, size_t count) const;
```

Classification accuracy on collected timesteps.

**Parameters:**
- `labels` -- Must span timesteps [0, start+count): `(start+count)` floats (class indices). Indexed from `labels[start]`.
- `start` -- First timestep index.
- `count` -- Number of timesteps to evaluate.

**Returns:** Fraction correct in [0.0, 1.0].

---

#### Prediction (Live Reservoir State)

For streaming inference without collecting states.

##### `PredictLiveRaw` (scalar)

```cpp
[[nodiscard]] float PredictLiveRaw() const;
```

Scalar prediction from the reservoir's current live state. Asserts `NumOutputs() == 1`.

---

##### `PredictLiveRaw` (multi-output)

```cpp
void PredictLiveRaw(float* output) const;
```

Writes `NumOutputs()` floats to `output` from the reservoir's current live state. For autoregressive / streaming inference loops.

---

##### `PredictFromState`

```cpp
void PredictFromState(const float* state, float* output) const;
```

Runs the readout on a caller-supplied state buffer (already subsampled to `NumOutputVerts()` floats, e.g. from `CopyLiveState`), writing `NumOutputs()` floats to `output`. Unlike `PredictLiveRaw`, it does not read the live reservoir — letting the caller modify the readout input (e.g. brand a side channel onto the first few slots) before prediction. This is the prequential predict-then-train primitive used by the streaming examples.

---

#### State Access and Persistence

##### `SelectedStates`

```cpp
[[nodiscard]] std::vector<float> SelectedStates() const;
```

Returns stride-selected vertices from all collected states: `NumCollected() * NumOutputVerts()` floats, row-major.

---

##### Accessors

| Method | Returns | Description |
|--------|---------|-------------|
| `NumCollected()` | `size_t` | Timesteps recorded by `Run()`. |
| `NumOutputs()` | `size_t` | From `cfg.readout.num_outputs` (set at construction). |
| `NumOutputVerts()` | `size_t` | Number of selected vertices M = ceil(N / stride). |
| `NumInputs()` | `size_t` | Number of input channels from config. |
| `Dim()` | `size_t` | Hypercube dimension of the underlying reservoir (`cfg.reservoir.dim`). |
| `Size()` | `size_t` | Reservoir neuron count N = 2^`Dim()`. |
| `GetConfig()` | `ESNConfig` | Full config used to construct this ESN (both reservoir and readout). |

---

##### Readout State Serialization

The ESN exposes its trained readout state for save/restore. The reservoir weights are deterministic from the seed, so only the config (`GetConfig()`) and readout state (`GetReadoutState()`) need to be persisted. On restore, construct a fresh `ESN` from the saved `ESNConfig` (which carries `reservoir.dim`) and call `SetReadoutState` — the readout config travels with the `ESNConfig`, so no separate `SetCNNConfig` step is needed.

**`ReadoutState` struct** (nested in `ESN`):

| Field | Type | Description |
|-------|------|-------------|
| `weights` | `std::vector<double>` | Opaque flattened blob of all conv kernels, biases, and dense-head weights. Round-trip only -- do not interpret. |
| `is_trained` | `bool` | True if the readout has been trained. |

| Method | Description |
|--------|-------------|
| `GetReadoutState()` | Extract trained readout for serialization. |
| `SetReadoutState(state)` | Restore a previously saved readout state. Reconstructs the CNN from stored config + weight blob. |

**Example: save and restore a trained model**

```cpp
// Save
ESNConfig cfg     = esn.GetConfig();         // includes both reservoir and readout
auto      state   = esn.GetReadoutState();
// serialize cfg + state using your preferred format

// Restore
ESN restored(cfg);           // cfg.reservoir.dim restores the dimension
restored.SetReadoutState(state);
// Ready to predict -- no retraining needed.
```

A standalone `Reservoir` is likewise self-describing: `reservoir.GetConfig()` returns the full `ReservoirConfig`, and `Reservoir::Create(reservoir.GetConfig())` rebuilds an identical reservoir (the weights are deterministic in the seed). The returned `spectral_radius` is the configured target; `GetRealizedSpectralRadius()` exposes the post-rescale value separately.

---

## Dependencies

HypercubeESN depends on a single external project:

**HypercubeCNN** -- library providing the hypercube convolutional network used by `Readout`.

- Location: **vendored** as a read-only snapshot at `third_party/HypercubeCNN` (see its `VENDORED.md` for the pinned upstream commit). No sibling checkout or network fetch — the build is offline and version-pinned.
- Built **transitively** as part of the HypercubeESN build via `add_subdirectory(third_party/HypercubeCNN)`; a small in-tree shim CMakeLists builds only its `HypercubeCNNCore` static lib — no separate pre-build step.
- Public headers are re-exported through `HypercubeESNCore`'s include interface, so consumers of HypercubeESN do not need to add HypercubeCNN to their own link line -- `target_link_libraries(my_app PRIVATE HypercubeESNCore)` pulls it in transitively.
- The HCNN headers used by HypercubeESN consumers are the ones re-exported by `Readout.h` (forward-declared `hcnn::HCNN` via PIMPL); the full HCNN API is not part of the public HypercubeESN surface.

No other external dependencies beyond the C++ standard library.
