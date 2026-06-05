# HypercubeESN

[![Build wheels](https://github.com/dliptak001/HypercubeESN/actions/workflows/wheels.yml/badge.svg)](https://github.com/dliptak001/HypercubeESN/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/hypercube-esn)](https://pypi.org/project/hypercube-esn/)
[![Python](https://img.shields.io/pypi/pyversions/hypercube-esn)](https://pypi.org/project/hypercube-esn/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)

HypercubeESN reimagines the echo-state reservoir as a signal living on a Boolean
hypercube. Its neurons live on the vertices of the hypercube and connect only to
their Hamming-distance-1 neighbors, with every adjacency resolved by a single XOR
on the vertex's binary index — a deterministic O(1) lookup that stores nothing at
all. There is no adjacency list to build, store, or serialize; the entire
connectivity is implicit in the indices themselves.

In contrast to the arbitrary sparse graph of a conventional ESN, the structure is
symmetric, deterministic, and reproducible across any two implementations at the
same dimension — while the neurons themselves stay familiar continuous tanh
units. Only the addressing is binary; the dynamics are fully real-valued.

That same implicit addressing extends into time. Each vertex update reaches not
only across its neighbors' current states but back through an addressable delay
line of each neighbor's last M states — one XOR-addressed gather spanning space
and time together. Temporal memory is intrinsic to the topology: memory by
construction rather than by luck. The result is an ESN that is at once
mathematically clean, strikingly memory-frugal, and strong where reservoirs are
meant to be: long memory and nonlinear computation.

## What is Reservoir Computing?

Reservoir computing is a machine learning paradigm for temporal data. Training a
recurrent network end-to-end is expensive and unstable — backpropagation through
time wrestles with vanishing gradients and rarely converges cheaply. Reservoir
computing sidesteps the problem entirely by splitting it in two:

1. **A fixed, random recurrent network — the reservoir.** It receives the input
   and lifts it into a high-dimensional, time-varying state. The recurrent weights
   are set once at initialization and never trained.

2. **A trained readout.** It learns to map that state to the desired output.
   Classically this is a single linear regression; HypercubeESN replaces the linear
   fit with a learned convolutional readout
   ([HypercubeCNN](https://github.com/dliptak001/HypercubeCNN)) that discovers
   nonlinear features directly on the hypercube topology.

The insight is that a rich enough dynamical system, once driven by input, builds
its own high-dimensional embedding of the input's history for free. The recurrence
supplies the computational power; the readout supplies the learning. Training only
that readout is what makes reservoir computing converge orders of magnitude faster
than a backprop-trained RNN — while staying competitive on tasks that demand
memory and nonlinear computation.

## What is HypercubeESN?

HypercubeESN is a reservoir computing architecture in which the reservoir's wiring
*is* the Boolean hypercube. Each neuron sits on a vertex and receives only from
its Hamming-distance-1 neighbors — the vertices one bit-flip away — every address
computed by XOR, no adjacency list stored. The reservoir state is therefore not
an abstract vector; it is a *signal on a hypercube graph*, a field of activations
laid out on vertices and shaped by XOR-addressed dynamics.

The question, then, is what reads that signal. A conventional reservoir flattens
its state into a vector and fits a line through it, discarding the geometry that
produced it. A spatial CNN would force the activations onto a 2D grid they never
lived on. HypercubeESN does neither. Its readout is
[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN) — a convolutional
network whose kernels are defined on the very same hypercube. Convolutions gather
over Hamming-distance neighborhoods with weights shared under the hypercube's
symmetry group; pooling pairs each vertex with its bitwise complement and folds
DIM down by one, yielding a perfect sub-hypercube. No padding, no borders, no
reshaping — neighbor lookup is the same single XOR the reservoir already speaks.

The pairing is topology-native: the readout consumes the reservoir's output with
zero distortion, and the learned kernels exploit the very locality that generated
the dynamics. The data never leaves the hypercube it was born on.

HypercubeESN targets DIM 5–16 (32 to 65,536 neurons), the practical range for
reservoir computing.

## Why a Hypercube?

A random reservoir graph is an arbitrary object: it must be generated, stored,
and trusted. The hypercube is none of those — its structure is a mathematical
given, and that gives the architecture properties a random graph cannot:

**Zero storage overhead.** No adjacency list, ever. A random reservoir keeps three
arrays — states, weights, and the graph wiring them together; the hypercube keeps
only the first two, because the third is implied by the indices. Connectivity is
*computed*, never stored — so the cache never fills with adjacency indices, and
each neighbor is reached by arithmetic (`v XOR (1 << i)`) rather than a pointer
chased through memory.

**Perfect homogeneity.** The hypercube is vertex-transitive: every neuron has
exactly DIM neighbors and sees an identical local world. No hubs, no dead ends, no
degree lottery — none of the structural variance a random sparse graph drags in.
That same uniformity is what lets HypercubeCNN share one set of kernel weights
across the entire graph.

**Logarithmic reach.** Any two of the neurons are at most DIM = log₂N bit-flips
apart. A signal's influence can span the whole reservoir in logarithmically few
hops, even though each neuron wires to only DIM others — sparse local connectivity
with global reach, exactly the property that makes a reservoir mix.

**Implicit, reproducible structure.** XOR addressing is deterministic: two
implementations at the same DIM agree on every connection automatically — no
graph to serialize, exchange, or version. And the reproducibility runs deeper than
the wiring. Because the weights are drawn from a seeded generator and rescaled to
a target spectral radius, the *entire* reservoir reconstructs from a handful of
scalars — DIM, a seed, and a few drive parameters (spectral radius, leak, input
scaling, history depth). A reservoir is *specified*, not stored.

## Architecture Summary

| Property | Detail |
|---|---|
| Neurons | N = 2^DIM neurons on hypercube vertices — DIM 5–16, so 32 to 65,536 |
| Connectivity | DIM neighbors per neuron: the single-bit-flip (Hamming-distance-1) vertices, addressed `v XOR (1 << i)` |
| Addressing | XOR on vertex indices — O(1), branchless, zero storage (no adjacency list) |
| Neuron model | Leaky-integrator tanh: `state = (1 − leak)·prev + leak·tanh(drive)` |
| History depth | M = `history_depth` (default 16, range 1–64) — each update taps the last M states via an addressable delay line; M = 1 is a single-step ESN, M > 1 deepens temporal memory |
| Step cost | O(N · DIM · M) per timestep — sparse, never O(N²) |
| Configuration | `ReservoirConfig` (`Reservoir.h`): `seed`, `spectral_radius`, `leak_rate`, `input_scaling`, `history_depth` |
| Readout | HypercubeCNN; `output_fraction` selects a stride-reduced vertex subset — decouples reservoir size from readout cost |

## Pipeline

A fixed hypercube reservoir feeds a trained HypercubeCNN readout. Each reservoir
vertex updates from an input term plus a recurrent term gathered over the M
delay-line slices of its DIM neighbors, then publishes through a leaky-integrator
tanh:

```
# drive s: an input term, plus a recurrent term over the M delay-line slices
s = input_term(v)
for j in 0..M-1:            # M = history_depth — the delay line
    for i in 0..DIM-1:      # DIM spatial neighbors per slice
        s += slice_j[v XOR (1<<i)] * W_rec[v][j][i]
state[v] = (1 - leak_rate) * slice_0[v] + leak_rate * tanh(s)
```

`slice_0` is the previous step's output (the leaky carryover); deeper slices
expose older states as separately-weighted taps, so each vertex is a fixed
spatiotemporal filter over the last M steps. The recurrent weights are random and
frozen, rescaled once at construction to the target spectral radius — estimated by
power iteration over the M-slice companion operator.

The readout, [HypercubeCNN](https://github.com/dliptak001/HypercubeCNN), is the
only trained component. It convolves directly on the reservoir's hypercube
topology (see [What is HypercubeESN?](#what-is-hypercubeesn)) and supports
regression (single/multi-output), multi-class classification, and online
streaming training.

See [docs/Reservoir.md](docs/Reservoir.md) and
[docs/Readout.md](docs/Readout.md) for full architectural detail.

## Related Work

The hypercube has met reservoir computing before. Katori (2019),
"[Reservoir Computing Based on Dynamics of Pseudo-Billiard System in
Hypercube](https://ieeexplore.ieee.org/document/8852329/)" (IJCNN 2019, Best
Paper Award), drives a reservoir through pseudo-billiard chaotic dynamics in a
hypercube *state space*, where binary units interact via a Chaotic Boltzmann
Machine. HypercubeESN uses the hypercube differently — not as the space the state
moves through, but as the *wiring* between neurons: XOR-addressed connectivity for
an echo-state network of continuous tanh units. Same structural primitive,
opposite role — Katori computes *in* the hypercube; HypercubeESN computes *across*
it.

## Python SDK

Pre-built wheels are available on [PyPI](https://pypi.org/project/hypercube-esn/)
for Python 3.10–3.13 on Windows (x64), Linux (x86_64, aarch64), and macOS
(x86_64, arm64). No compiler required.

```bash
pip install hypercube-esn
```

```python
import numpy as np
import hypercube_esn as he

signal = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)
esn = he.ESN(dim=7, seed=73895)  # surveyed default seed
esn.fit(signal, warmup=200)
print(f"R2 = {esn.r2():.6f}")
```

See [docs/Python_SDK.md](docs/Python_SDK.md) for the full API reference.

## Building and Running (C++)

**Requirements:** C++23 compiler (GCC 13+, Clang 17+, MSVC 2022+), CMake 4.1+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/HypercubeESN
```

The build produces seven executables:

| Target | Purpose |
|---|---|
| `HypercubeESN` | Stub entry point |
| `BasicPrediction` | Minimal example: sine wave prediction |
| `SignalClassification` | Multi-class waveform recognition with confusion matrix |
| `StreamingAnomaly` | Streaming anomaly detection with recovery dynamics |
| `MemoryCapacity` | Jaeger memory-capacity diagnostic (white-noise MC sweep) |
| `NARMA` | NARMA-N nonlinear benchmark (M×seed sweep) |
| `StreamingText` | Streaming character-level memorization of a text corpus |

Start with `BasicPrediction` to see the pipeline end-to-end. Each example has a
companion `.md` file with a detailed walkthrough.

## Project Structure

```
HypercubeESN/
  CMakeLists.txt         Top-level build (core lib + examples; pulls in HCNN subdir)
  Reservoir.h/cpp        Hypercube reservoir (N = 2^DIM vertices); ReservoirConfig
  Readout.h/cpp          Learned convolutional readout (PIMPL)
  ESN.h/cpp              Unified pipeline: warmup, run, train, predict
  main.cpp               Stub entry point

  examples/
    BasicPrediction.cpp/md       Minimal sine wave prediction
    SignalClassification.cpp/md  Multi-class waveform recognition
    StreamingAnomaly.cpp/md      Streaming anomaly detection
    MemoryCapacity/              Jaeger memory-capacity diagnostic
    NARMA/                       NARMA-N nonlinear benchmark (M×seed sweep)
    StreamingText/               Streaming character-level text memorization

  python/                Python bindings (pybind11 module + pyproject)
  cmake/                 Package config template (find_package support)

  docs/
    Reservoir.md          Reservoir architecture, connectivity, parameters
    Readout.md            HCNN readout: architecture, training, streaming mode
    CPP_SDK.md            C++ static-library consumer guide
    Python_SDK.md         Python SDK API reference
```

## Documentation

| Document | Covers |
|---|---|
| [docs/Reservoir.md](docs/Reservoir.md) | Hypercube graph, connectivity, deep-vertex history depth, leaky integrator, spectral-radius tuning, DIM-invariant input drive |
| [docs/Readout.md](docs/Readout.md) | HCNN readout architecture, training algorithm, streaming mode, ESN interface |
| [docs/Python_SDK.md](docs/Python_SDK.md) | Python SDK: pip install, fit/predict API, streaming, persistence |
| [docs/CPP_SDK.md](docs/CPP_SDK.md) | C++ static library: build, install, find_package usage, API reference |

Each example in `examples/` has a companion `.md` walkthrough with sample
results and interpretation guidance.
