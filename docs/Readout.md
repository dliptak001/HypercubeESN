# Readout — HypercubeCNN as ESN Readout Layer

The reservoir does the hard, fixed work of lifting an input stream into a
high-dimensional state; the readout is the one part that learns. In HypercubeESN
that readout is a HypercubeCNN — a convolutional network that reads the reservoir
state *on the cube*, never flattening it into an anonymous feature vector. Two
things follow from that single choice: the readout's inductive bias matches the
reservoir's topology exactly, and it discovers its own nonlinear features instead
of settling for a linear fit.

## A readout that speaks the reservoir's language

The pairing is topology-native. A HypercubeESN reservoir is a Boolean hypercube —
neurons sit on hypercube vertices, wired by XOR-addressed edges — so the reservoir
state *is* a signal on a hypercube graph. HypercubeCNN's convolutions are built to
operate on exactly that structure: Hamming-distance kernels that respect the same
vertex addressing and neighbor relationships the reservoir uses to evolve.

So the state reaches the readout with zero topological distortion. Nothing is
reshaped into a flat vector for a linear fit, nothing is packed into a fabricated
2D grid for an image CNN — the data never leaves the hypercube it was born on, and
the convolution kernels exploit the very adjacency that generated the dynamics.
Locality on the reservoir graph becomes locality in the kernel: neurons that
influenced each other as the reservoir ran are neighbors again when the features
are learned.

That is what separates it from the alternatives. Ridge regression on the flattened
state forgets the graph; a spatial CNN invents one that has nothing to do with the
reservoir's wiring; HypercubeCNN inherits it — the only readout whose inductive
bias is the reservoir's own.

## What HypercubeCNN brings

[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN) is a standalone
convolutional library that swaps the 2D pixel grid for a Boolean hypercube, and
each of its primitives is the reservoir's own geometry seen from the other side:

- **Exact weight sharing.** Every vertex has exactly DIM neighbors, reached by
  flipping one bit, and the hypercube is vertex-transitive — every vertex sees an
  identical neighborhood — so one kernel is shared across the entire graph, exact
  under the hypercube's Z₂ⁿ symmetry, with no boundary where that sharing breaks
  down. Neighbor lookup is a single XOR instruction; there are no adjacency lists,
  no padding, and no border effects.
- **Pooling that stays on a cube.** Each Conv+Pool stage pairs every vertex with
  its bitwise complement — the maximally distant vertex — and drops DIM by one,
  leaving a perfect sub-hypercube. Stacking stages builds a feature hierarchy the
  way a spatial CNN does: DIM shrinks, channel count grows.
- **One pipeline, either task.** Classification (softmax + cross-entropy) and
  regression (MSE) run through the same network, trained end-to-end with
  backpropagation and Adam.

## The only thing that learns

```
Reservoir (N states) ──────────────────────────────> Readout
    fixed random                                       TRAINED
```

Everything upstream is frozen: the reservoir's weights are random and fixed at
initialization, and every parameter that learns lives in the readout. That is the
core bargain of reservoir computing — let a fixed nonlinear system do the
projection, and train only the layer that reads it out.

**Data path:** raw state (N = 2^DIM) → HCNN (Conv→Pool stack → Flatten → Linear)
→ output.

## Architecture

The conv stack is sized from DIM and built through HypercubeCNN’s architecture
product (`LayerSpec` / `HCNNConfig::Build` in `Readout.cpp`) — not hand-rolled
private-layer calls. Each Conv+Pool stage halves the hypercube when pooling is
on (antipodal pool drops DIM by one). HypercubeCNN requires DIM >= 3 for conv;
ESN reservoirs use DIM 5–16, so a reservoir of dimension DIM admits at most
`DIM - 2` pooled stages. Channels grow by `channel_growth` (default 2) after
each conv, starting from `conv_channels` (16 -> 32 -> …).

`num_layers` chooses the depth:

- **`1` (default)** — a single Conv(+Pool) stage at the base `conv_channels`.
- **`0` (auto)** — `min(DIM - 2, 2)`; across all supported DIMs (5-16) that cap of
  2 is always hit, giving a 2-layer stack (16 -> 32 with default growth).
- **explicit `n`** — exactly `n` stages, asserted `<= DIM - 2` when pooling.

Additional knobs (defaults preserve historical stacks): `use_pooling`,
`pool_type` (Max/Avg), `use_batchnorm` (off — keeps weight blobs stable),
`optimizer` (Adam), `channel_growth`.

| Component        | Supported DIM | Source                                |
|------------------|---------------|---------------------------------------|
| HypercubeCNN     | 3–30          | public `HCNN` ctor contract           |
| HypercubeESN ESN | 5–16          | `Reservoir::Create` (validates `dim`) |
| Readout          | **5–16**      | Intersection; matches the ESN range   |

Host integration plan: [revendor_HypercubeCNN.md](revendor_HypercubeCNN.md).

## Training (batch)

A stack of hypercube Conv(+Pool) layers feeds a flatten and a dense head, trained
under HypercubeCNN’s unified train API with a cosine-annealed learning rate.

1. Build the stack via `HCNNConfig` / `LayerSpec` from DIM (see [Architecture](#architecture));
   default optimizer is Adam.
2. `Readout::Train` drives `hcnn::HCNNTrainer`: full-capacity `HCNNInputView`,
   per-epoch shuffle seed, cosine LR via `hcnn::cosine_lr` from `lr_max` down to
   `lr_max * lr_min_frac` over `lr_decay_epochs` (0 = `epochs`; last scheduled
   epoch hits the floor when the horizon is > 1).
3. Two task heads (overload by target type — no `*Regression` names):
   - **Regression** — MSE loss; raw network output at inference (the readout does
     **not** center targets — see [Task Types](#task-types)).
   - **Classification** — integer class labels, softmax + cross-entropy; logits via
     `PredictRaw` or argmax via `PredictClass`.
4. After training, the weights are flattened via `HCNN::GetWeights()` for
   serialization and restored with `SetWeights()` on reload (`ReadoutLoadMode::Eval`
   default; `ResumeTrain` resets optimizer moments).

### Best-epoch restore (optional)

By default (`ReadoutConfig::restore_best_epoch = true`), `Train` scores after
every epoch and restores the best snapshot at the end. Set `false` for
historical last-epoch weights. Metrics:

| Task | Metric | Helper |
|------|--------|--------|
| Regression | min MSE | `HCNNBestMetricCheckpoint` |
| Classification | max accuracy | `HCNNDualCheckpoint` best-acc |

Optional `best_epoch_holdout_frac` (0…0.5): take that fraction of samples from
the **tail** of the batch as a hold-out score set; train only on the prefix.
`0` scores the full training set. Cost is one full forward over the score set
per epoch. Query the selected epoch with `Readout::BestEpoch()` /
`ESN::ReadoutBestEpoch()` (1-based, or 0 if unused).

### Multi-ESN threading

HypercubeCNN may open its own worker pool (`num_threads`: 0 = auto, 1 = none,
N = N workers). When the **host** already parallelizes across many `ESN`
instances (seed surveys, grid search), set `readout.num_threads = 1` so each
net stays single-threaded and you do not nest pools. Lorenz’s survey does this;
single-ESN demos can leave the default `0` (auto).

The conv stack sees raw reservoir state, with no per-vertex standardization — and
that is deliberate. Reservoir outputs are already tanh-bounded in `(-1, +1)`, the
distribution the kernel is tuned for; centering or scaling each vertex on its own
would shift them independently and break the very spatial correlations across the
hypercube that the kernel exists to read.

## Task Types

| Task             | targets layout                         | Output             | Metric   |
|------------------|----------------------------------------|--------------------|----------|
| Regression       | num_samples x num_outputs (row-major)  | raw network output | R2, NRMSE |
| Classification   | num_samples floats (class indices)     | logits (argmax)    | Accuracy |

For regression with non-zero-mean targets, center your targets before
training and add the mean back to predictions — the readout no longer
does this for you.

Configured via `ReadoutConfig::task` (`ReadoutTask::Regression` / `Classification`)
and `ReadoutConfig::num_outputs`.

## ReadoutConfig

```cpp
struct ReadoutConfig {
    size_t dim           = 0;        // input feature dim: 2^dim features per sample (set by ESN)
    int num_outputs      = 1;        // classes or regression targets
    ReadoutTask task     = ReadoutTask::Regression;
    int num_layers       = 1;        // Conv(+Pool) stages; 0 = auto: min(DIM-2, 2)
    bool use_pooling     = true;     // antipodal pool after each conv
    ReadoutPoolType pool_type = ReadoutPoolType::Max;
    int conv_channels    = 16;       // base channels (first conv)
    int channel_growth   = 2;        // multiply channels after each stage
    bool use_batchnorm   = false;    // off keeps weight-blob layout stable
    ReadoutOptimizer optimizer = ReadoutOptimizer::Adam;
    int epochs           = 200;
    int batch_size       = 32;
    float lr_max         = 0.0015f;  // cosine annealing peak (keep <= 0.005 to avoid NaN)
    float lr_min_frac    = 0.01f;    // floor = lr_max * lr_min_frac
    int   lr_decay_epochs = 0;       // cosine decay horizon; 0 = use epochs
    float weight_decay   = 0.0f;
    float momentum       = 0.0f;     // SGD momentum; ignored by Adam
    unsigned seed        = 42;       // CNN weight init seed
    ReadoutActivation activation = ReadoutActivation::TANH;
    size_t num_threads   = 0;        // HCNN pool: 0=auto, 1=ST, N=N workers
    bool restore_best_epoch = true;  // restore best-epoch weights after Train (default)
    float best_epoch_holdout_frac = 0.0f; // tail hold-out for best metric (0=score train)
};
```

**Cost:** O(epochs * samples * layer_flops). For a typical DIM=8
configuration (~256 states per sample, 1-2 Conv+Pool pairs, ~20k samples,
a few hundred epochs) this runs in seconds to minutes depending on core
count. CPU cores saturate at `batch_size >= 128`.

**Stability note:** `lr_max` above ~0.005 can drive weights into
denormal/NaN territory, where CPU falls off fast math paths and
throughput collapses.

## When to Use

- Tasks where a linear-readout ceiling is hit and nonlinear feature
  discovery is worth the training cost.
- Classification problems. HCNN natively supports multi-class with
  softmax+cross-entropy. See `examples/SignalClassification.cpp`.
- DIM 7+ where the auto-sized architecture gets enough Conv+Pool depth
  to be expressive.

**When not to use:** Small-DIM tasks (5-6) where the architecture has
minimal depth and training cost isn't justified by accuracy gains.

## Streaming Training API

Readout supports per-sample and mini-batch streaming gradient steps
for applications where data arrives continuously.

### Setup

The architecture and Adam optimizer are built eagerly in the `Readout`
constructor — no separate init call. Warm up the reservoir (via
`ESN::ReservoirWarmup`) before the first gradient step.

### Gradient steps

Both methods dispatch on the construction-time task (`config_.task`), so a
single `const float*` target serves both — for regression it is `num_outputs`
floats; for classification, a single class-index float (cast to int).

| Method | Granularity | target |
|--------|-------------|--------|
| `TrainStep(state, target, lr, wd)` | Single sample | `num_outputs` floats, or a (1,) class index |
| `TrainStepBatch(states, targets, count, lr, wd)` | Mini-batch | `count * num_outputs`, or `count` class indices |

Mini-batch is parallelized via unified `HCNN::TrainBatch` (int* labels or
float* targets, overload by type).

See `examples/StreamingText/` for a working streaming implementation
and `examples/StreamingAnomaly.cpp` for an anomaly-detection use case.

## Serialization

### ESN-native blob (`Weights` / `SetState`)

Weights are flattened/restored via `HCNN::GetWeights()` / `SetWeights()`.
Layout: per conv (kernel, bias?, BN stats if enabled) then readout weights + bias.
Optimizer moments are **not** in the blob. The blob is **unversioned** (same layout
as the live HCNN); architecture is implied by `ReadoutConfig` used to build the net.

The network is built eagerly from `config_` in the Readout ctor; `SetState`
injects the blob into the live net. `ReadoutLoadMode::Eval` (default) restores
parameters only; `ResumeTrain` also resets optimizer moments for continued online
training.

### HypercubeCNN-native model (`SaveHcnnModel` / `LoadHcnnModel`)

Portable pair written from a path **stem** (no extension):

| File | Contents |
|------|----------|
| `stem.hcnw` | Versioned HCNW binary (`hcnn::save_weights`) — dims, task, layer counts, float32 weights |
| `stem.arch.json` | ESN arch sidecar (`format: hypercube_esn_readout_arch`, `version: 1`) — knobs + expanded `layers` + `weight_count` |

Load validates the sidecar against the **live** readout architecture when the JSON
is present, then calls `hcnn::load_weights`. Missing sidecar still loads HCNW
(HCNN checks dim/task/layer counts). ESN wrappers: `SaveReadoutHcnnModel` /
`LoadReadoutHcnnModel`. Logging: `ArchSummary()` / `ReadoutArchSummary()`.

```cpp
esn.SaveReadoutHcnnModel("models/lorenz_readout");
// ... rebuild same ESNConfig / architecture ...
esn.LoadReadoutHcnnModel("models/lorenz_readout"); // Eval by default
```

## Readout Public Interface

`Readout` is the readout class used by `ESN`. ESN holds it as a
direct `Readout readout_` member and delegates training, prediction,
and evaluation to it. The methods below are on Readout; see
`docs/CPP_SDK.md` for the ESN-level wrappers.

| Method | Returns |
|--------|---------|
| `Train(states, targets, num_samples)` | void |
| `PredictRaw(state, output)` | void (multi-output) |
| `PredictClass(state)` | int (argmax over logits) |
| `R2(states, targets, num_samples)` | double |
| `Accuracy(states, labels, num_samples)` | double |
| `Weights()` | `vector<double>` (unversioned flattened blob) |
| `SetState(weights, mode=Eval)` | void (load blob into live net) |
| `SaveHcnnModel(stem)` | void — write `stem.hcnw` + `stem.arch.json` |
| `LoadHcnnModel(stem, mode=Eval)` | void — validate arch sidecar, load HCNW |
| `ArchSummary()` | `string` — layers + param counts |
| `BestEpoch()` | int — 1-based best epoch after `restore_best_epoch` Train, else 0 |
| `NumFeatures()` | size_t — features per sample = 2^dim; equals the reservoir's N (the readout sees all N vertices) |
| `NumOutputs()` | size_t |

### ESN Integration Points

The readout's `ReadoutConfig` travels inside `ESNConfig` and is passed once at ESN construction — `Train` takes no config argument, and the readout CNN is built eagerly in the ESN ctor.

- `ESN::Train(targets, train_size)` → `Readout::Train` using `cfg.readout`
- `ESN::ReservoirWarmup(inputs, num_steps)` → settle the reservoir before `TrainStep` / `TrainStepBatch`
- `ESN::Predict()` / `ESN::PredictFromRecorded(timestep)` / `ESN::PredictFromState(state)` → return `std::vector<float>` (NumOutputs())
- `ESN::NumOutputs()` → delegates to `Readout::NumOutputs()`
- `ESN::R2/NRMSE/Accuracy` → handle multi-output target layout

## Implementation Notes

- Lives at the project root with separate .h/.cpp files.
- Holds a `std::unique_ptr<hcnn::HCNN>` via PIMPL so that
  `#include "HCNN.h"` stays in the .cpp only.
- Not templated -- accepts arbitrary feature counts at runtime.
- Does not store training data -- only learned weights and the config
  used to rebuild the network on reload.
