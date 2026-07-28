# Readout — HypercubeCNN as the trained half

The reservoir does the hard, fixed work of lifting an input stream into a
high-dimensional state; the readout is the only part that learns. In HypercubeESN
that readout is a HypercubeCNN — a convolutional network that reads the reservoir
*on the cube*. Two things follow: the readout's inductive bias matches the
reservoir's topology, and it discovers its own nonlinear features instead of
settling for a linear fit on a geometry-blind flat vector.

Implementation: `Readout` / `ReadoutConfig` in `Readout.h` and `Readout.cpp`
(PIMPL over `hcnn::HCNN`). ESN owns a `Readout` and assembles its input from the
reservoir delay line — see [Reservoir.md](Reservoir.md) and [CPP_SDK.md](CPP_SDK.md).

## Role in the pipeline

```
  inputs [+ optional ext-fb]
       │
       ▼
  Reservoir (fixed)
       │
       ▼
  SliceAt(0 .. B-1)  ── pack B blocks of N ──▶  HCNN readout (trained) ──▶ y
```

Only the readout emits **y**. The reservoir never writes the task output; it
publishes state the readout reads. External feedback is an *input* into the
reservoir (caller-owned), not a second path to **y**.

Vocabulary (shared with ESN):

| Term | Meaning |
|------|---------|
| **N** | Reservoir neurons = 2<sup>reservoir.dim</sup>; length of one delay-line slice |
| **B** | `ESNConfig::readout_slices` — power of two, 1 ≤ B ≤ M |
| **Reservoir state** | Newest slice only (N floats) |
| **Readout input** | What the HCNN sees: B blocks of N = 2<sup>readout.dim</sup> floats. Equals reservoir state only when B = 1 |

## A readout that speaks the reservoir's language

The pairing is topology-native. A HypercubeESN reservoir is a Boolean hypercube —
neurons sit on hypercube vertices, wired by XOR-addressed edges — so each
published slice *is* a signal on a hypercube graph. HypercubeCNN's convolutions
are built for exactly that structure: Hamming-distance kernels that respect the
same vertex addressing and neighbor relationships the reservoir uses to evolve.

So the data reaches the stack with zero topological distortion. Nothing is packed
into a fabricated 2D grid for an image CNN, and nothing is thrown at ridge
regression as an anonymous length-N vector. The data stays on the hypercube through
the conv (and optional pool) stages; only the **final linear head** flattens
`(channel, vertex)` features. Locality on the reservoir graph becomes locality in
the kernel: neurons that influenced each other as the reservoir ran are neighbors
again when the features are learned.

That is what separates it from the alternatives. Ridge on the flattened state
forgets the graph; a spatial CNN invents one that has nothing to do with the
reservoir's wiring; HypercubeCNN inherits the reservoir's topology through the
body of the network.

## What HypercubeCNN brings

[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN) is a standalone
convolutional library that swaps the 2D pixel grid for a Boolean hypercube. Each
primitive is the reservoir's geometry from the other side (v1.0.0 facade used by
this host):

- **Exact weight sharing with a self tap.** Every vertex has **dim** Hamming
  neighbors (one bit flip each) **plus a self/center contribution** — kernel width
  **K = dim + 1**, shared across all vertices. The hypercube is vertex-transitive,
  so sharing is exact under the Z₂ⁿ symmetry; neighbor lookup is XOR, with no
  adjacency list and no image border to pad. (ESN always feeds **full capacity**
  `input_channels` × 2<sup>dim</sup>; short raw vectors would be zero-padded by HCNN, which
  this host does not rely on.)
- **Pooling that stays on a cube.** Optional antipodal pool pairs each vertex with
  its bitwise complement, drops dim by one, and leaves a perfect sub-hypercube.
  Stacking stages builds a feature hierarchy: dim shrinks, channel count grows.
- **One pipeline, either task.** Classification (softmax + cross-entropy) and
  regression (MSE) share the forward graph; loss is fixed by `TaskType`. Default
  optimizer is Adam (SGD remains available).

Spatial embed/aug helpers exist in HypercubeCNN for images; **reservoir readout
does not use them** — state is already length N (or multi-block length N·B) on the
cube.

Vendor pin: [third_party/HypercubeCNN/VENDORED.md](../third_party/HypercubeCNN/VENDORED.md).

## The only thing that learns

Everything upstream is frozen: reservoir weights are random and fixed at init.
Every trainable parameter lives in the readout — the core reservoir-computing
bargain.

**Data path** (HCNN start dimension = `readout.dim`):

```
  readout_input[2^dim] ──▶ Embed ──▶ [Conv (+ optional Pool)] × L
                                      ──▶ Flatten ──▶ Linear ──▶ num_outputs
```

Channels grow by `channel_growth` after each conv (default 16 → 32 → …). Softmax
is only inside the classification **loss**, never in `PredictRaw` / `Predict` /
`Forward`.

## Architecture

The stack is built through HypercubeCNN's architecture product (`LayerSpec` /
`HCNNConfig::Build` in `Readout.cpp`) — not hand-rolled private-layer includes.
Each pooled stage drops dim by one when `use_pooling` is on. HypercubeCNN allows
start dim in [3, 30]; ESN reservoirs are dim 5–16, and with pooling the stack is
asserted to leave dim ≥ 2 (`num_layers ≤ dim − 2`). Channels grow by
`channel_growth` (default 2) after each conv, starting from `conv_channels`.

`num_layers` chooses depth:

- **`1` (default)** — one Conv(+Pool) stage at `conv_channels`.
- **`0` (auto)** — `min(dim − 2, 2)`; for dim 5–16 that is always a 2-stage stack
  (16 → 32 with default growth) when pooling allows.
- **explicit `n`** — `n` stages; with pooling, asserted `n ≤ dim − 2`.

Other shape knobs: `use_pooling`, `pool_type` (Max/Avg), `use_batchnorm` (default
off — keeps weight-blob layout stable), `optimizer` (Adam), `activation`,
`channel_growth`. `input_channels` is always **1** at the HCNN ctor; multi-block
ESN inputs expand **dim**, not channel count (below).

| Component    | Supported dim | Source |
|--------------|---------------|--------|
| HypercubeCNN | 3–30          | public `HCNN` ctor |
| HypercubeESN | 5–16          | `Reservoir::Create` |
| Readout      | **5–16**      | Intersection with ESN (asserted at stack build) |

### Input size: single block vs multi-block

`ReadoutConfig::dim` is the hypercube dimension of the **readout input**, not
always the reservoir dim. ESN sets it at construction via `MakeReadoutConfig`
(do not set `readout.dim` yourself):

- **Default** (`readout_slices = 1`): `readout.dim = reservoir.dim`,
  `NumFeatures()` = N = 2<sup>reservoir.dim</sup> — one float per reservoir vertex.
- **Multi-block** (`readout_slices = B > 1`): B must be a **power of two** and
  ≤ `reservoir.history_depth`. Then
  `readout.dim = reservoir.dim + log2(B)` and `NumFeatures() = N × B`.

#### How ESN packs B slices

Each step ESN assembles the readout input from the delay line:

1. Logical age `k` comes from `Reservoir::SliceAt(k)` (age 0 = newest).
2. Age `k` is copied into **physical block** `ReadoutBlockOf(k)`.

For B ≤ 2 the map is the identity. For B > 2, consecutive ages land on block
indices that differ in **two** bits (pair map in `ESN::MakeBlockMap`). Reason:
HCNN conv has **no self term** and only 1-bit neighbors, so a single filter can
see “velocity” {age0, age1} only if those blocks are two bit-flips apart.

Antipodal pooling mixes **every** bit, including block-index bits when B > 1;
set `use_pooling = false` to keep block structure intact into the flatten head
(more features, more parameters). Seam field lives on `ESNConfig::readout_slices`
— see [CPP_SDK.md](CPP_SDK.md).

## Training (batch)

1. Build the stack via `HCNNConfig` / `LayerSpec` (see [Architecture](#architecture));
   default optimizer Adam (`SetOptimizer` / config).
2. `Readout::Train` drives `hcnn::HCNNTrainer`: full-capacity `HCNNInputView`,
   per-epoch shuffle seed, cosine LR via `hcnn::cosine_lr` from `lr_max` down to
   `lr_max * lr_min_frac` over horizon `lr_decay_epochs` if > 0, else `epochs`.
   When the horizon is > 1, the cosine schedule reaches the floor at the last
   **schedule** index (epoch `horizon − 1`); if `epochs` differs from the horizon,
   training may stop before or hold at the floor afterward.
3. Task overload (no `*Regression` names):
   - **Regression** — MSE; raw network output at inference (no automatic target
     centering — see [Task types](#task-types)).
   - **Classification** — integer class labels; softmax CE in the loss; logits via
     `PredictRaw`, argmax via `PredictClass`.
4. A second `Train()` **continues** from current weights; construct a new `Readout`
   (or ESN) for a fresh random init.
5. Weights for checkpoints: `Weights` / `SetState` (see [Serialization](#serialization)).

### Best-epoch restore (default on)

By default (`restore_best_epoch = true`), after each epoch `Train` scores a metric
and at the end restores the best snapshot. Set `false` for last-epoch weights.

| Task | Metric | Helper |
|------|--------|--------|
| Regression | min MSE | `HCNNBestMetricCheckpoint` |
| Classification | max accuracy | `HCNNDualCheckpoint` best-acc |

`best_epoch_holdout_frac` in [0, 0.5]: fraction of samples (input order, **tail**)
held out for scoring only; train on the prefix. `0` scores the full training set
(not a pure validation early-stop). Cost: one full forward over the score set
**every epoch**. Query with `Readout::BestEpoch()` / `ESN::ReadoutBestEpoch()`
(1-based; 0 if restore was off or no snapshot).

### Multi-ESN threading

HCNN worker pool: `num_threads` 0 = auto, 1 = no background workers, N = N workers.
When the host parallelizes across many ESNs (seed surveys), set
`readout.num_threads = 1` to avoid nested pools. Lorenz's survey does this;
single-ESN demos can leave `0`.

### Input scaling note

The stack sees raw readout input with no per-vertex standardization — deliberate.
Reservoir units are typically tanh-bounded in (−1, +1); centering or scaling each
vertex independently would break the spatial correlations the kernel is meant to
read.

## Task types

| Task | `targets` layout | Output | Readout metric |
|------|------------------|--------|----------------|
| Regression | `num_samples × num_outputs` (row-major) | raw predictions | `R2` (below) |
| Classification | `num_samples` class indices as float | logits / argmax | `Accuracy` (below) |

**R²:** average of per-output coefficients of determination over the sample set
(multi-output). Perfect fit → 1.0; degenerate target variance on an output → 0 for
that output.

**Accuracy:** multi-class = fraction of argmax matches; single-output classif
thresholds the logit at 0. **NRMSE** is an **ESN** helper (RMSE / target std),
not a `Readout` method.

For regression with non-zero-mean targets, center targets before training and
add the mean back on predictions if you need absolute scale — the readout does
not center for you.

Configured via `ReadoutTask` and `num_outputs`.

## ReadoutConfig

```cpp
struct ReadoutConfig {
    size_t dim           = 0;        // 2^dim features per sample (set by ESN)
    int num_outputs      = 1;
    ReadoutTask task     = ReadoutTask::Regression;
    int num_layers       = 1;        // 0 = auto min(dim-2, 2)
    bool use_pooling     = true;
    ReadoutPoolType pool_type = ReadoutPoolType::Max;
    int conv_channels    = 16;
    int channel_growth   = 2;
    bool use_batchnorm   = false;
    ReadoutOptimizer optimizer = ReadoutOptimizer::Adam;
    int epochs           = 200;
    int batch_size       = 32;
    float lr_max         = 0.0015f;  // keep <= ~0.005 to avoid NaN
    float lr_min_frac    = 0.01f;
    int   lr_decay_epochs = 0;       // 0 = use epochs as cosine horizon
    float weight_decay   = 0.0f;
    float momentum       = 0.0f;     // SGD only; ignored by Adam
    unsigned seed        = 42;
    ReadoutActivation activation = ReadoutActivation::TANH;
    size_t num_threads   = 0;        // 0=auto, 1=ST, N=N workers
    bool restore_best_epoch = true;
    float best_epoch_holdout_frac = 0.0f;
};
```

**Cost:** roughly O(epochs × samples × layer flops), plus an extra score forward
per epoch when `restore_best_epoch` is on (default). Typical dim=8, hundreds of
epochs: seconds to minutes. `batch_size >= 128` often saturates multi-core HCNN
pools when `num_threads` is not 1.

**Stability:** `lr_max` above ~0.005 can push weights into denormal/NaN territory
and tank throughput.

## When to use

- Tasks where a linear readout ceiling is hit and nonlinear features are worth
  the train cost.
- Multi-class problems (native CE). See `examples/SignalClassification.cpp`.
- dim 7+ when auto depth and pooling give a more expressive stack.

**When not to use:** tiny dim (5–6) if train cost is not worth the accuracy gain;
or pure linear diagnostics (e.g. MemoryCapacity uses ridge, not this readout).

## Streaming training API

Per-sample and mini-batch gradient steps for continuous data.

### Setup

Architecture and optimizer are built eagerly in the `Readout` ctor — no separate
init. Warm the reservoir (`ESN::ReservoirWarmup`) before the first online step.

### Gradient steps

Dispatch on construction-time task. Caller supplies **lr** (and optional
weight decay) each call; **momentum** comes from config (SGD).

| Method | Granularity | target |
|--------|-------------|--------|
| `TrainStep(state, target, lr, wd)` | one sample | `num_outputs` floats, or one class-index float |
| `TrainStepBatch(states, targets, count, lr, wd)` | mini-batch | `count × num_outputs`, or `count` class indices |

`state` / each row of `states` is a **readout input** (`NumFeatures()` floats) —
B blocks of N when used under ESN with B > 1, not necessarily the newest
reservoir slice alone.

Batch path uses unified `HCNN::TrainBatch`. For online schedules, hosts often use
`CosineLR` / `ExponentialDecayLR` from `Readout.h` (batch `Train` uses HCNN's
`cosine_lr` instead — same shape, different ownership of the epoch index).

See `examples/StreamingAnomaly.cpp` for online/streaming training.

## Serialization

### ESN-native blob (`Weights` / `SetState`)

`HCNN::GetWeights` layout as `vector<double>`: per conv (kernel, bias?, BN stats
if enabled) then linear head. **No** optimizer moments. Unversioned; architecture
must already match the live net (built in the ctor). `SetState` **injects** into
that net — it does not rebuild layers. `ReadoutLoadMode::Eval` (default) loads
parameters only; `ResumeTrain` also resets optimizer moments for continued online
training.

### HypercubeCNN-native model (`SaveHcnnModel` / `LoadHcnnModel`)

Path **stem** (no extension):

| File | Contents |
|------|----------|
| `stem.hcnw` | Versioned HCNW (`hcnn::save_weights`) |
| `stem.arch.json` | `format: hypercube_esn_readout_arch`, `version: 1` — knobs, expanded layers, `weight_count` |

Load validates the sidecar against the live architecture when present, then
`load_weights`. Missing sidecar: HCNW's own dim/task/layer checks still apply.
ESN: `SaveReadoutHcnnModel` / `LoadReadoutHcnnModel`. Logs: `ArchSummary()` /
`ReadoutArchSummary()`.

```cpp
esn.SaveReadoutHcnnModel("models/lorenz_readout");
// rebuild matching ESNConfig / architecture
esn.LoadReadoutHcnnModel("models/lorenz_readout"); // Eval by default
```

## Readout public interface

ESN holds `Readout readout_` and delegates. Methods on `Readout` (see also
[CPP_SDK.md](CPP_SDK.md)):

| Method | Returns |
|--------|---------|
| `Train(states, targets, num_samples)` | void (continues weights if called again) |
| `TrainStep` / `TrainStepBatch` | void |
| `PredictRaw(state, output)` | void |
| `PredictClass(state)` | int |
| `R2` / `Accuracy` | double |
| `Weights()` | `vector<double>` |
| `SetState(weights, mode=Eval)` | void |
| `SaveHcnnModel` / `LoadHcnnModel` | void |
| `ArchSummary()` | string |
| `BestEpoch()` | int |
| `NumFeatures()` / `NumOutputs()` | size_t |
| `GetConfig()` / `IsTrained()` | config / always true once constructed |

`NumFeatures()` = 2<sup>dim</sup> for this readout's input cube (reservoir N only when
`dim` equals reservoir dim — see multi-block above).

`IsTrained()` means the network exists and has weights worth persisting (true
from construction on) — **not** “has seen training data.”

### ESN integration

`ReadoutConfig` lives in `ESNConfig`; the CNN is built in the ESN ctor
(`MakeReadoutConfig` rewrites `dim` from `readout_slices`).

| ESN | Readout |
|-----|---------|
| `Train(targets, train_size)` | `Train` on collected readout inputs |
| `TrainStep` / `TrainStepBatch` | streaming |
| `Predict` / `PredictFromRecorded` / `PredictFromState` | `PredictRaw` |
| `R2` / `NRMSE` / `Accuracy` | eval helpers (NRMSE is ESN-only) |
| `GetReadoutState` / `SetReadoutState(..., mode)` | `Weights` / `SetState` |
| `SaveReadoutHcnnModel` / `LoadReadoutHcnnModel` | HCNW + arch |
| `ReadoutArchSummary` / `ReadoutBestEpoch` | `ArchSummary` / `BestEpoch` |
| `NumOutputs` / `ReadoutInputWidth` | `NumOutputs` / `NumFeatures` |
| `CopyReadoutInput` / `ReadoutBlockOf` | assemble / inspect packing |

## Implementation notes

- Project root: `Readout.h` / `Readout.cpp`.
- `std::unique_ptr<hcnn::HCNN>` PIMPL — `#include "HCNN.h"` only in the .cpp.
- Not templated; capacity is a power of two (2<sup>dim</sup>), with `dim ≥ 5` asserted at
  stack build.
- Does not store training data — only live weights, config, and best-epoch
  metadata after `Train`.
- Examples print `ReadoutArchSummary()` after construction (or once before multi-seed
  surveys). MemoryCapacity uses ridge, not this readout.
