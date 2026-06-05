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

The conv stack is sized from DIM. Each Conv+Pool pair halves the hypercube — pooling
drops DIM by one — and HCNNConv needs DIM >= 3, so a reservoir of dimension DIM
admits at most `DIM - 2` pairs. Channels double at every layer, starting from
`conv_channels` (16 -> 32 -> ...).

`num_layers` chooses the depth:

- **`1` (default)** — a single Conv+Pool layer at the base `conv_channels`.
- **`0` (auto)** — `min(DIM - 2, 2)`; across all supported DIMs (5-16) that cap of
  2 is always hit, giving a 2-layer stack (16 -> 32).
- **explicit `n`** — exactly `n` pairs, asserted `<= DIM - 2`.

Whatever the source, the count is clamped to at least 1 (`std::max(layers, 1)`),
and the final hypercube dimension is the start DIM minus the layer count.

| Component        | Supported DIM | Source                                |
|------------------|---------------|---------------------------------------|
| HypercubeCNN     | 3-32          | `HCNNNetwork.cpp:24`                  |
| HypercubeESN ESN  | 5-16          | `Reservoir::Create` (validates `dim`) |
| Readout          | **5-16**      | Intersection; matches the ESN range   |

## Training (batch)

A stack of hypercube Conv+MaxPool layers feeds a flatten and a dense head, trained
with Adam under a cosine-annealed learning rate.

1. Build the conv stack from DIM (see [Architecture](#architecture)) and attach the
   Adam optimizer.
2. Each epoch, shuffle the samples into mini-batches — the shuffle seed varies per
   epoch — and run Adam forward/backward over the full training set. The learning
   rate follows a cosine schedule from `lr_max` down to `lr_max * lr_min_frac` over
   `lr_decay_epochs` (0 = `epochs`).
3. Two task heads:
   - **Regression** — MSE loss; raw network output at inference (the readout does
     **not** center targets — see [Task Types](#task-types)).
   - **Classification** — integer class labels, softmax + cross-entropy; logits via
     `PredictRaw` or argmax via `PredictClass`.
4. After training, the weights are flattened via `HCNN::GetWeights()` for
   serialization and restored with `SetWeights()` on reload.

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
    int num_layers       = 1;        // Conv+Pool pairs; 0 = auto: min(DIM-2, 2)
    int conv_channels    = 16;       // base channels (doubles per layer)
    int epochs           = 200;
    int batch_size       = 32;
    float lr_max         = 0.0015f;  // cosine annealing peak (keep <= 0.005 to avoid NaN)
    float lr_min_frac    = 0.01f;    // floor = lr_max * lr_min_frac
    int   lr_decay_epochs = 0;       // cosine decay horizon; 0 = use epochs
    float weight_decay   = 0.0f;
    float momentum       = 0.0f;     // SGD momentum; 0.9 typical for CNN
    unsigned seed        = 42;       // CNN weight init seed
    bool verbose         = false;    // print per-epoch lr
    bool verbose_train_acc = false;  // also print train accuracy/MSE each epoch
    ReadoutActivation activation = ReadoutActivation::TANH;  // per-Conv-layer (TANH/RELU/LEAKY_RELU/NONE)
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

## Online Training API

Readout supports per-sample and mini-batch online gradient steps
for streaming applications where data arrives continuously.

### Setup

`InitOnline()` builds the architecture and sets the Adam optimizer.

### Gradient steps

| Method | Task | Granularity |
|--------|------|-------------|
| `TrainOnlineStep(state, class, lr, wd)` | Classification | Single sample |
| `TrainOnlineBatch(states, classes, count, lr, wd)` | Classification | Mini-batch |
| `TrainOnlineStepRegression(state, target, lr, wd)` | Regression | Single sample |
| `TrainOnlineBatchRegression(states, targets, count, lr, wd)` | Regression | Mini-batch |

Mini-batch variants are parallelized via `HCNN::TrainBatch` (classification)
and `HCNN::TrainBatchRegression` (regression).

See `examples/StreamingText/` for a working streaming implementation
and `examples/StreamingAnomaly.cpp` for an anomaly-detection use case.

## Serialization

Weights are flattened/restored via `HCNN::GetWeights()` / `SetWeights()`.
Layout: conv kernels + biases (per layer, in order) then readout weights + bias.

`rebuild_from_blob()` reconstructs the full HCNN network from `config_`
(which carries `dim` and the architecture knobs) via `build_architecture()`,
then injects the weight blob -- no retraining needed.

## Readout Public Interface

`Readout` is the readout class used by `ESN`. ESN holds it as a
direct `Readout readout_` member and delegates training, prediction,
and evaluation to it. The methods below are on Readout; see
`docs/CPP_SDK.md` for the ESN-level wrappers.

| Method | Returns |
|--------|---------|
| `Train(states, targets, num_samples)` | void |
| `PredictRaw(state)` | float (scalar, single-output) |
| `PredictRaw(state, output)` | void (multi-output) |
| `PredictClass(state)` | int (argmax over logits) |
| `R2(states, targets, num_samples)` | double |
| `Accuracy(states, labels, num_samples)` | double |
| `Weights()` | `vector<double>` (flattened blob) |
| `SetState(weights)` | void (rebuild network from blob) |
| `NumFeatures()` | size_t — features per sample = 2^dim; equals the reservoir's N unless the ESN's `output_fraction` stride-reduces it |
| `NumOutputs()` | size_t |

### ESN Integration Points

The readout's `ReadoutConfig` travels inside `ESNConfig` and is passed once at ESN construction — `Train` and `InitOnline` take no config argument.

- `ESN::Train(targets, train_size)` → `Readout::Train` using `cfg.readout`
- `ESN::InitOnline(warmup_inputs, warmup_count)` → build architecture
- `ESN::PredictRaw(timestep)` → scalar (asserts `num_outputs == 1`)
- `ESN::PredictRaw(timestep, float* output)` → multi-output
- `ESN::NumOutputs()` → delegates to `Readout::NumOutputs()`
- `ESN::R2/NRMSE/Accuracy` → handle multi-output target layout and state subsampling

## Implementation Notes

- Lives at the project root with separate .h/.cpp files.
- Holds a `std::unique_ptr<hcnn::HCNN>` via PIMPL so that
  `#include "HCNN.h"` stays in the .cpp only.
- Not templated -- accepts arbitrary feature counts at runtime.
- Does not store training data -- only learned weights and the config
  used to rebuild the network on reload.
