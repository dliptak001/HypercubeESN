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
    esn.ReservoirWarmup(signal.data(), warmup);
    esn.ReservoirRun(signal.data() + warmup, collect);

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
    float    input_scaling   = 0.5f;   // weights × input_scaling/√dim (fan-in)
    size_t   num_inputs      = 1;
    size_t   history_depth   = 16;
    float    history_floor   = 1.0f;   // depth taper K in [0.1, 1.0]; 1.0 = none
    bool     verbose         = true;

    // External feedback (caller-owned) — 0 = off, no alloc
    size_t   num_external_feedback_channels = 0;
    float    external_feedback_scaling      = 0.5f;

    // Full-state linear feedback (internal φ = V·x) — false = off, no alloc
    bool     full_state_feedback = false;
    uint64_t fsf_seed            = 1;      // V then B_fsf; not derived from seed
    float    fsf_scaling         = 0.05f;
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
| `seed` | `uint64_t` | `73895` | Master RNG seed (recurrent / input / external-feedback / bias / SR-probe substreams). Different seeds produce measurably different performance; screen per DIM/task and set explicitly. |
| `spectral_radius` | `float` | `0.99` | Target spectral radius of the recurrent operator (the MN×MN augmented companion operator when `history_depth` > 1). Controls the echo-state property — how quickly past inputs fade. Tune per DIM/task. |
| `leak_rate` | `float` | `1.0` | Leaky-integrator coefficient. `state = (1 - leak_rate) * old_output + leak_rate * tanh(drive)`. At 1.0, each step fully replaces state; values < 1.0 add explicit temporal carryover. |
| `input_scaling` | `float` | `0.5` | Input drive coefficient. Input weights are drawn U(−1,1) then scaled by `input_scaling / √DIM` so each vertex’s dim-neighbor input sum has fan-in-normalized variance. That is local weight construction, not a promise that one value is optimal at every DIM or task (not the legacy fixed `0.02`, which was a readout-standardization artifact). Task-dependent, typically O(0.5–3); retune when you change size or task. |
| `num_inputs` | `size_t` | `1` | Number of input channels; must divide N evenly. In multi-input mode (K channels), channel k drives the contiguous vertex block `[k*N/K, (k+1)*N/K)`. |
| `history_depth` | `size_t` | `16` | Per-vertex output-history depth M (the recurrent delay line): each `Step` sums over the M most-recent output slices, each with its own weights. Must be in [1, 64]; M = 1 is the legacy single-slice reservoir. See [Reservoir.md](Reservoir.md). |
| `history_floor` | `float` | `1.0` | Depth-taper floor K. Recurrent weights are linearly scaled by slice from just below 1.0 at the most-recent history slice down to K at the deepest, so older states influence the next state less. Applied before the spectral-radius rescale (which then normalizes overall magnitude, preserving the relative per-slice profile). Must be in [0.1, 1.0]; `1.0` = no taper (identity), and the taper has no effect when `history_depth == 1`. |
| `verbose` | `bool` | `true` | Print the per-construction reservoir banner with the seed/leak/input-scaling, depth-taper floor, and spectral-radius rescale (`[Reservoir DIM=… M=… seed=… leak=… in_scale=… hist_floor=… SR target=… post=… (secant iters=…)]`). |
| `num_external_feedback_channels` | `size_t` | `0` | External-feedback channels D. **0** = path off (no buffer/weights). Else D in **[1, N]** (need not divide N). Caller stages values each step. See [reservoir_feedback_mechanism.md](reservoir_feedback_mechanism.md). |
| `external_feedback_scaling` | `float` | `0.5` | Like input: weights × `scaling / √DIM` (only if D > 0). Outside spectral-radius rescale. |
| `full_state_feedback` | `bool` | `false` | Construction-only enable for internal full-state feedback. **false** ⇒ zero FSF allocation. See [full_state_linear_feedback.md](full_state_linear_feedback.md). |
| `fsf_seed` | `uint64_t` | `1` | Draws V as U(−1,1) (first N) then B_fsf (next N·dim). Standalone — not derived from `seed`. |
| `fsf_scaling` | `float` | `0.05` | B_fsf: U(−1,1) × `scaling / √DIM` (only FSF strength knob; like `input_scaling`). |

---

### ReadoutConfig

Configuration struct for the HCNN readout's architecture and training.

```cpp
struct ReadoutConfig {
    size_t dim           = 0;        // input feature dim = 2^dim; set internally by ESN — do not set
    int num_outputs      = 1;
    ReadoutTask task     = ReadoutTask::Regression;
    int num_layers       = 1;        // Conv(+Pool) stages; 0 = auto: min(DIM-2, 2)
    bool use_pooling     = true;
    ReadoutPoolType pool_type = ReadoutPoolType::Max;
    int conv_channels    = 16;       // first-layer channels
    int channel_growth   = 2;        // multiply after each stage
    bool use_batchnorm   = false;
    ReadoutOptimizer optimizer = ReadoutOptimizer::Adam;
    int epochs           = 200;
    int batch_size       = 32;
    float lr_max         = 0.0015f;
    float lr_min_frac    = 0.01f;
    int   lr_decay_epochs = 0;       // 0 = use epochs
    float weight_decay   = 0.0f;
    float momentum       = 0.0f;     // SGD only; ignored by Adam
    unsigned seed        = 42;
    ReadoutActivation activation = ReadoutActivation::TANH;
    size_t num_threads   = 0;        // 0=auto, 1=single-threaded HCNN, N=N workers
    bool restore_best_epoch = true;  // restore best-epoch weights after Train
    float best_epoch_holdout_frac = 0.0f; // tail hold-out for selection (0=score train)
};
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dim` | `size_t` | `0` | Input feature dimension (features per sample = 2^dim). **Set internally by the ESN** to `cfg.reservoir.dim` (the readout sees all N vertices) — consumers do not set this; any value here is overwritten at construction. |
| `num_outputs` | `int` | `1` | Number of output neurons. For regression: number of targets. For classification: number of classes. |
| `task` | `ReadoutTask` | `Regression` | Task head. See [ReadoutTask](#readouttask). |
| `num_layers` | `int` | `1` | Number of Conv(+Pool) stages. `1` (default) builds a single stage. `0` auto-computes `min(DIM - 2, 2)`. Each Pool halves the hypercube dimension, capped by `DIM - 2`. |
| `use_pooling` | `bool` | `true` | Antipodal pool after each conv. |
| `pool_type` | `ReadoutPoolType` | `Max` | Max or Avg pool when pooling is on. |
| `conv_channels` | `int` | `16` | Base channel count for the first Conv layer. |
| `channel_growth` | `int` | `2` | Multiplier applied to channels after each stage (historical default: double). |
| `use_batchnorm` | `bool` | `false` | Per-conv BN. Off keeps the weight-blob layout stable; enabling grows the blob. |
| `optimizer` | `ReadoutOptimizer` | `Adam` | Forwarded to HypercubeCNN. |
| `epochs` | `int` | `200` | Training epochs (batch mode). Structured signals saturate at ~25 epochs; chaotic signals need ~2000. Ignored in online mode. |
| `batch_size` | `int` | `32` | Mini-batch size (batch mode). Use 128+ on multi-core CPUs to saturate threading. |
| `lr_max` | `float` | `0.0015` | Peak learning rate for cosine annealing. **Keep <= 0.005 to avoid weight divergence into denormal/NaN territory.** |
| `lr_min_frac` | `float` | `0.01` | Cosine schedule floor as fraction of `lr_max`. Effective `lr_min = lr_max * lr_min_frac`. |
| `lr_decay_epochs` | `int` | `0` | Cosine decay horizon. 0 = use `epochs`. Set > epochs to trace only a prefix of the cosine curve (keeps lr high when shortening a run). |
| `weight_decay` | `float` | `0.0` | L2 weight decay. |
| `momentum` | `float` | `0.0` | Heavy-ball SGD momentum. `0` = plain SGD; ignored under Adam. |
| `seed` | `unsigned` | `42` | Seed for weight initialization. |
| `activation` | `ReadoutActivation` | `TANH` | Activation applied after each Conv layer. See [ReadoutActivation](#readoutactivation). |
| `num_threads` | `size_t` | `0` | HCNN worker pool: `0` auto, `1` single-threaded (**required** when the host parallelizes across many ESNs — avoid nested pools), `N` workers. |
| `restore_best_epoch` | `bool` | `true` | After each epoch, score MSE (regression) or accuracy (classification) and restore the best weights at the end of `Train`. Set `false` for last-epoch weights. |
| `best_epoch_holdout_frac` | `float` | `0.0` | With `restore_best_epoch`: fraction of samples (tail, input order) held out for scoring only; train on the prefix. `0` scores the full train set. Clamped to [0, 0.5]. |

**Architecture:** built via HypercubeCNN `LayerSpec` / `HCNNConfig`. The default `num_layers = 1` builds a single Conv+Pool stage (16 channels). Set `num_layers = 0` for auto-sizing — `min(DIM - 2, 2)` stages with channels growing by `channel_growth`. See [Readout.md](Readout.md) and the vendor pin in [../third_party/HypercubeCNN/VENDORED.md](../third_party/HypercubeCNN/VENDORED.md).

**Multi-ESN threading:** set `readout.num_threads = 1` whenever outer code runs many ESNs in parallel (e.g. Lorenz survey). Leave `0` for a single-ESN process so HCNN can use an auto pool.

---

### ESN

The complete pipeline wrapper: Reservoir -> Readout. Constructed from a single `ESNConfig` that bundles the reservoir and readout configs (the hypercube dimension comes from `cfg.reservoir.dim`); no further config arguments are passed to `Train`.

```cpp
struct ESNConfig {
    ReservoirConfig reservoir;
    ReadoutConfig   readout;
};

// Construction (dimension comes from cfg.reservoir.dim)
ESN esn(cfg);

// Reservoir driving
esn.ReservoirWarmup(inputs, num_steps);
esn.ReservoirRun(inputs, num_steps);                    // accumulate into the batch
esn.ReservoirRun(inputs, num_steps, /*clear_recorded=*/true);  // start a fresh batch
esn.ReservoirClear();

// Batch training (readout hyperparameters come from cfg.readout)
esn.Train(targets, train_size);

// Streaming training (task fixed at construction)
esn.ReservoirWarmup(warmup_inputs, warmup_count);   // settle the reservoir before streaming
esn.TrainStep(target, lr, weight_decay);              // one step on the live state
esn.TrainStepBatch(states, targets, count, lr, weight_decay);  // step over accumulated states
esn.CopyReservoirState(out);

// Prediction & evaluation (recorded states)
esn.PredictFromRecorded(timestep);     // -> std::vector<float> (NumOutputs())
esn.R2(targets, start, count);
esn.NRMSE(targets, start, count);
esn.Accuracy(labels, start, count);

// Prediction (reservoir state)
esn.Predict();                         // -> std::vector<float> (NumOutputs())
esn.PredictFromState(state);           // run readout on a caller-supplied state

// State access & persistence
esn.CollectedStates();
esn.NumCollectedStates();
esn.NumOutputs();
esn.NumInputs();
esn.ReservoirHypercubeDimension();  // hypercube dimension (cfg.reservoir.dim)
esn.ReservoirNeuronCount();         // reservoir neuron count N = 2^ReservoirHypercubeDimension()
esn.GetConfig();             // returns ESNConfig
esn.GetReadoutState();
esn.SetReadoutState(state);
```

---

#### Construction

```cpp
explicit ESN(const ESNConfig& cfg);
```

Creates the reservoir from `cfg.reservoir` and builds the readout from `cfg.readout`. Reservoir weights are generated and spectral-radius-rescaled at construction time; the HCNN readout is also built eagerly here, so it is ready before the first `Train()` / `TrainStep` call.

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

##### `ReservoirStep`

```cpp
void ReservoirStep(const float* inputs, const float* external_feedback = nullptr);
```

One timestep: stage task `inputs` (always), optionally stage **external** feedback,
then `Step`. If `full_state_feedback` was enabled at construction, **FSF applies
automatically** inside the reservoir (φ = V·x from the current gain V) — do not
pass φ here.

**Parameters:**
- `inputs` -- `NumInputs()` floats for this step.
- `external_feedback` -- `nullptr` to skip; otherwise `NumExternalFeedbackChannels()`
  floats. Throws if non-null when D == 0.

---

##### `ReservoirWarmup`

```cpp
void ReservoirWarmup(const float* inputs, size_t num_steps);
```

Drives the reservoir for `num_steps` timesteps without recording state. Use this to wash out the reservoir's initial transient (zero state) before collecting data for training. Calls `ReservoirStep` **without** external feedback. **If FSF is enabled, FSF still applies each step.**

**Parameters:**
- `inputs` -- Pointer to `num_steps * num_inputs` floats, row-major. Each timestep has `num_inputs` consecutive values (one per channel). When `num_inputs == 1` (default), this is simply `num_steps` scalars. Values are **not** clamped — pass already-bounded signals (the `1/√DIM` input normalization sets the `tanh` operating point via `input_scaling`).
- `num_steps` -- Number of timesteps to drive. Typical: 100-500 depending on task.

---

##### `ReservoirRun`

```cpp
void ReservoirRun(const float* inputs, size_t num_steps, bool clear_recorded = false);
```

Drives the reservoir for `num_steps` timesteps, recording the readout input at each step. States are appended to the internal buffer -- multiple `ReservoirRun()` calls accumulate. External feedback is not injected; **FSF still applies if enabled**.

**Parameters:**
- `inputs` -- Pointer to `num_steps * num_inputs` floats, row-major. Same layout as `ReservoirWarmup()`.
- `num_steps` -- Number of timesteps to drive and record.
- `clear_recorded` -- If `true`, discard everything recorded by previous `ReservoirRun()` calls (and reset `NumCollectedStates()`) before recording this batch, so this call starts fresh. The reservoir's live state and the trained readout are untouched. Default `false` (accumulate). Use this between independent recording batches — clear + record in one call — instead of rebuilding the ESN.

---

##### `ReservoirClear`

```cpp
void ReservoirClear();
```

Clears the reservoir's live state — `vtx_state_` plus every output-history slice — so a new input sequence starts from rest. Recurrent weights, input weights, FSF gain V (if any), and all hyperparameters are untouched. Recorded states are **not** cleared. The trained readout is preserved.

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

#### Streaming Training

For applications where data arrives continuously. The reservoir advances one step at a time; the readout is updated via per-sample or mini-batch gradient steps. The readout CNN is built eagerly at construction, so there is no separate init call — just `ReservoirWarmup()` the reservoir to wash out the initial transient, then start the `TrainStep` / `TrainStepBatch` loop. The task (regression vs classification) is fixed at construction; both methods dispatch on it, so a single `const float*` target serves both. (`cfg.readout.epochs` is ignored here — the loop length is the caller's.)

##### `TrainStep`

```cpp
void TrainStep(const float* target, float lr, float weight_decay = 0.0f);
```

One streaming gradient step on the reservoir's current state. For regression, `target` is `NumOutputs()` floats; for classification, a single float holding the class index (cast to int internally).

---

##### `TrainStepBatch`

```cpp
void TrainStepBatch(const float* states, const float* targets, size_t count,
                    float lr, float weight_decay = 0.0f);
```

One streaming gradient step over a mini-batch of pre-accumulated states (parallelized across threads). `states` is `count` rows of `ReservoirNeuronCount()` floats (from `CopyReservoirState`). For regression, `targets` is `count * NumOutputs()` floats (row-major); for classification, `count` floats (class indices).

---

##### `CopyReservoirState`

```cpp
void CopyReservoirState(float* out) const;
```

Copies the current reservoir state into `out` (`ReservoirNeuronCount()` floats — all N vertices). Use to accumulate states for `TrainStepBatch`.

---

#### Prediction and Evaluation (Recorded States)

##### `PredictFromRecorded`

```cpp
[[nodiscard]] std::vector<float> PredictFromRecorded(size_t timestep) const;
```

Returns `NumOutputs()` floats for a recorded timestep. For regression: de-centered predictions. For classification: raw logits (apply argmax for the predicted class).

**Parameters:**
- `timestep` -- Index into recorded states, in [0, NumCollectedStates()).

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

#### Prediction (Reservoir State)

For streaming inference without recording states.

##### `Predict`

```cpp
[[nodiscard]] std::vector<float> Predict() const;
```

Returns `NumOutputs()` floats from the reservoir's current state. For autoregressive / streaming inference loops.

---

##### `PredictFromState`

```cpp
[[nodiscard]] std::vector<float> PredictFromState(const float* state) const;
```

Runs the readout on a state buffer you pass in (`ReservoirNeuronCount()` floats, e.g. one saved earlier with `CopyReservoirState`), returning `NumOutputs()` floats. Unlike `Predict`, it never reads the reservoir — so you can predict from a stored state, or adjust the state before the readout sees it (for example, overwriting the first few entries with an external signal). This is the prequential predict-then-train primitive used by the streaming examples.

---

#### State Access and Persistence

##### `CollectedStates`

```cpp
[[nodiscard]] std::vector<float> CollectedStates() const;
```

Returns all collected states: `NumCollectedStates() * ReservoirNeuronCount()` floats, row-major.

---

##### Accessors

| Method | Returns | Description |
|--------|---------|-------------|
| `NumCollectedStates()` | `size_t` | Recorded reservoir-state snapshots (one per timestep) from `ReservoirRun()`. |
| `NumOutputs()` | `size_t` | From `cfg.readout.num_outputs` (set at construction). |
| `NumInputs()` | `size_t` | Number of input channels from config. |
| `NumExternalFeedbackChannels()` | `size_t` | D from `cfg.reservoir.num_external_feedback_channels` (0 = no external-feedback port). |
| `ReservoirHypercubeDimension()` | `size_t` | Hypercube dimension of the underlying reservoir (`cfg.reservoir.dim`). |
| `ReservoirNeuronCount()` | `size_t` | Reservoir neuron count N = 2^`ReservoirHypercubeDimension()`. |
| `FullStateFeedbackEnabled()` | `bool` | True if built with `full_state_feedback`. |
| `GetConfig()` | `ESNConfig` | Full config (reservoir + readout), including FSF knobs. V is reconstructed from `fsf_seed` / `fsf_scaling`. |

See [full_state_linear_feedback.md](full_state_linear_feedback.md).

---

##### Readout State Serialization

The ESN exposes its trained readout state for save/restore. Reservoir topology, weights, and FSF gain V are deterministic from config + seeds (`seed`, and when FSF is on `fsf_seed` / `fsf_scaling`). Persist config (`GetConfig()`) and readout (`GetReadoutState()`). On restore, construct a fresh `ESN` from the saved `ESNConfig` and call `SetReadoutState`.

**`ReadoutState` struct** (nested in `ESN`):

| Field | Type | Description |
|-------|------|-------------|
| `weights` | `std::vector<double>` | Opaque flattened blob of all conv kernels, biases, and dense-head weights. Round-trip only -- do not interpret. |
| `is_trained` | `bool` | True if the readout has been trained. |

| Method | Description |
|--------|-------------|
| `GetReadoutState()` | Extract trained readout for serialization (unversioned double blob). |
| `SetReadoutState(state, mode=Eval)` | Restore a previously saved readout state into the live CNN. `ReadoutLoadMode::Eval` (default) loads parameters only; `ResumeTrain` also resets optimizer moments for continued online training. |
| `ReadoutBestEpoch()` | 1-based epoch restored when `restore_best_epoch` was used, else 0. |
| `SaveReadoutHcnnModel(stem)` | Write portable `stem.hcnw` + `stem.arch.json` (HypercubeCNN-native). |
| `LoadReadoutHcnnModel(stem, mode=Eval)` | Load HCNW after validating arch sidecar against the live net. |
| `ReadoutArchSummary()` | Human-readable layer stack and parameter counts. |

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

A standalone `Reservoir` is likewise self-describing: `reservoir.GetConfig()` returns the full `ReservoirConfig` (including FSF knobs). `Reservoir::Create(reservoir.GetConfig())` rebuilds matching weight blocks and FSF gain V from `seed` / `fsf_seed` and `fsf_scaling`. The returned `spectral_radius` is the configured target; `GetRealizedSpectralRadius()` exposes the post-rescale value separately.

---

## Dependencies

HypercubeESN depends on a single external project:

**HypercubeCNN** -- library providing the hypercube convolutional network used by `Readout`.

- Location: **vendored** as a read-only snapshot at `third_party/HypercubeCNN` (see its `VENDORED.md` for the pinned upstream commit). No sibling checkout or network fetch — the build is offline and version-pinned.
- Built **transitively** as part of the HypercubeESN build via `add_subdirectory(third_party/HypercubeCNN)`; a small in-tree shim CMakeLists builds only its `HypercubeCNNCore` static lib — no separate pre-build step.
- Public headers are re-exported through `HypercubeESNCore`'s include interface, so consumers of HypercubeESN do not need to add HypercubeCNN to their own link line -- `target_link_libraries(my_app PRIVATE HypercubeESNCore)` pulls it in transitively.
- The HCNN headers used by HypercubeESN consumers are the ones re-exported by `Readout.h` (forward-declared `hcnn::HCNN` via PIMPL); the full HCNN API is not part of the public HypercubeESN surface.

No other external dependencies beyond the C++ standard library.
