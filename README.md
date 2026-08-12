# HypercubeESN

[![Build wheels](https://github.com/dliptak001/HypercubeESN/actions/workflows/wheels.yml/badge.svg)](https://github.com/dliptak001/HypercubeESN/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/hypercube_esn)](https://pypi.org/project/hypercube-esn/)
[![Python](https://img.shields.io/pypi/pyversions/hypercube_esn)](https://pypi.org/project/hypercube-esn/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)

**HypercubeESN** — reservoir computing on a Boolean hypercube. Neurons sit on the
vertices, each carrying a short delay line of its own past, wired to
single-bit-flip neighbors by XOR.

Three properties follow:

- **A topology you don't store.** Connectivity is implicit in the vertex indices —
  no adjacency list
- **Hidden multi-scale structure.** Full neighbor connectivity with random weights
  turns the cube into nested clusters — local, regional, and global at once —
  that nobody designed in.
- **Memory you can address.** Each vertex carries a delay line of its own recent
  past, so the reservoir remembers *specific* lags by construction, not echoes.

The reservoir state is a *signal on that graph*, not an anonymous vector. What
reads it is [HypercubeCNN](https://github.com/dliptak001/HypercubeCNN) —
convolutions on the same vertices and XOR neighborhoods, not a ridge fit on a
flat state and not an image CNN on a fabricated 2D grid. The pairing is
topology-native: the readout consumes the reservoir with zero distortion, and
the learned kernels exploit the locality that generated the dynamics. **The data
never leaves the hypercube it was born on.**

**2.0 readout upgrade.** Each HCNN conv site now has
an explicit **self/center** weight alongside its dim Hamming-1 neighbors. The change significantly improves readout quality
**across the board** (for all tasks and dims).

---

<p align="center">
  <strong>HypercubeAI ecosystem</strong><br/>
  <sub>One geometry. Topology-native intelligence.</sub>
</p>

<p align="center">
  <a href="https://github.com/dliptak001/HypercubeESN"><strong>HypercubeESN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeCNN"><strong>HypercubeCNN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeHopfield"><strong>HypercubeHopfield</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeWTF"><strong>HypercubeWTF</strong></a>
</p>

HypercubeESN is an experiment in the **HypercubeAI** project — our quest to map
AI and ML strategies onto the hypercube as a computational substrate.

Why the hypercube? A few properties keep showing up — and they explain why a
frozen reservoir and a HypercubeCNN readout fit together so cleanly:

- **A topology you don’t store** — the graph is specified: connectivity is
  implicit in the vertex indices; with a seed and a few config scalars the whole
  reservoir reconstructs mathematically.
- **Perfect homogeneity** — every vertex has the same degree and the same local
  world, so local dynamics mean the same thing everywhere — no structural
  favorites baked in by a random graph.
- **Cheap navigation** — each neighbor is a few bit operations on the vertex
  index, not a pointer chase through a stored edge list, so walks stay
  arithmetic and cache-friendly.
- **Topology-native pairing** — the readout consumes the reservoir’s output with
  zero geometric distortion, and the learned kernels exploit the same locality
  that generated the dynamics. The data never leaves the hypercube it was born
  on.

Each product in the family is a different architecture on that same foundation:

| Product | Natural data | Role of the hypercube |
|---------|--------------|------------------------|
| **[HypercubeESN](https://github.com/dliptak001/HypercubeESN)** | Low-dim **streams** over time | Frozen **reservoir** stepped each sample; multi-slice state → HypercubeCNN readout |
| **[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN)** | Static patterns on the cube | Trainable **spatial** conv/pool on the cube (no recurrent reservoir) |
| **[HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield)** | Patterns / attractors | Associative memory dynamics on the cube |
| **[HypercubeWTF](https://github.com/dliptak001/HypercubeWTF)** | Static high-dim fields (**no** intrinsic time) | Same **frozen hypercube reservoir** discipline as ESN, driven for a short orbit per sample, then HypercubeCNN on the **end state** |

---

## Headline results

Primary validators — open-loop (NARMA, MC) and closed-loop free-run (Lorenz).
Details: [NARMA](examples/NARMA/NARMA.md) · [MemoryCapacity](examples/MemoryCapacity/MemoryCapacity.md) · [Lorenz](examples/Lorenz/README.md).

### NARMA

tanh-wrapped orders 30 / 50 / 70; **Same operating point (same dim, sr, memory depth, reservoir seed, ...) for all three orders.**

| Order | Best-5 mean test NRMSE |
|------:|-----------------------:|
| 30 |             **0.0441** |
| 50 |             **0.0751** |
| 70 |             **0.1251** |

### Memory capacity (Jaeger MC)

Linear short-term memory (ridge on reservoir state — not HCNN). **Tunable** via
dim, memory depth, and spectral radius.

| dim | N | Peak TotalMC |
|----:|--:|-------------:|
| 5 | 32 | ~30 |
| 8 | 256 | ~250 |
| 10 | 1024 | ~820 |
| 12 | 4096 | ~1380 |

### Lorenz (free-run)

Closed-loop free-run on Lorenz-63: **input-bank self-feedback** (predicted
`[x, y, z, x*z]` re-injected as the next drive). dim 10,
M = 2; VPT threshold θ = 0.25. Best orbit VPT in Lyapunov times for three
trained seeds:

| ESN seed | Best VPT (LT) |
|---------:|--------------:|
| 3079493423467196890 | **14.13** |
| 696634088797950509 | **13.00** |
| 7934791766227647176 | **10.67** |

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

The reservoir’s wiring *is* the Boolean hypercube (see the three properties
above). The state is a *signal on that graph* — a field of activations on vertices
shaped by XOR-addressed dynamics — not an abstract length-N vector.

The question is what reads that signal. A conventional reservoir flattens its
state and fits a line through it, discarding the geometry that produced it. A
spatial CNN would force the activations onto a 2D grid they never lived on.
HypercubeESN does neither. Its readout is
[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN): convolutions over
Hamming neighborhoods with weights shared under the cube’s symmetry; optional
antipodal pooling folds dim by one into a perfect sub-hypercube. No padding, no
borders — neighbor lookup is the same single XOR the reservoir already speaks.

The pairing is topology-native: zero distortion into the readout; learned kernels
exploit the locality that generated the dynamics. **The data never leaves the
hypercube it was born on.** Practical range: dim 5–16 (32 to 65,536 neurons).

## Why a Hypercube?

A random reservoir graph is an arbitrary object: it must be generated, stored,
and trusted. The hypercube is none of those — its structure is a mathematical
given, and that gives the architecture properties a random graph cannot realize:

**Zero storage overhead.** No adjacency list, ever. A random reservoir keeps three
arrays — states, weights, and the graph wiring them together; the hypercube keeps
only the first two, because the third is implied by the indices. Connectivity is
*computed*, never stored — so the cache never fills with adjacency indices, and
each neighbor is reached by arithmetic (`v XOR (1 << i)`) rather than a pointer
chased through memory.

**Perfect homogeneity.** The hypercube is vertex-transitive: every neuron has
exactly dim neighbors and sees an identical local world. No hubs, no dead ends, no
degree lottery — none of the structural variance a random sparse graph drags in.
That same uniformity is what lets HypercubeCNN share one set of kernel weights
across the entire graph.

**Logarithmic reach.** Any two of the neurons are at most dim = log₂N bit-flips
apart. A signal's influence can span the whole reservoir in logarithmically few
hops, even though each neuron wires to only dim others — sparse local connectivity
with global reach, exactly the property that makes a reservoir mix.

**Implicit, reproducible structure.** XOR addressing is deterministic: two
implementations at the same dim agree on every connection automatically — no
graph to serialize, exchange, or version. And the reproducibility runs deeper than
the wiring. Because the weights are drawn from a seeded generator and rescaled to
a target spectral radius, the *entire* reservoir reconstructs from a handful of
scalars — dim, a seed, and a few drive parameters (spectral radius, leak, input
scaling, history depth). A reservoir is *specified*, not stored.

## Architecture Summary

| Property | Detail |
|---|---|
| Neurons | N = 2<sup>dim</sup> on hypercube vertices; **dim** = hypercube dimension (5–16 → 32 to 65,536 neurons) |
| Connectivity | dim neighbors per neuron: the single-bit-flip (Hamming-distance-1) vertices, addressed `v XOR (1 << i)` |
| Addressing | XOR on vertex indices — O(1), branchless, zero storage (no adjacency list) |
| Neuron model | Leaky-integrator tanh: `state = (1 − leak)·prev + leak·tanh(drive)` |
| History depth | M = `history_depth` (default 16, range 1–64) — each update taps the last M states via an addressable delay line; M = 1 is a single-step ESN, M > 1 deepens temporal memory |
| Step cost | O(N · dim · M) per timestep — sparse, never O(N²) |
| Configuration | `ReservoirConfig` (`Reservoir.h`): `seed`, `spectral_radius`, `leak_rate`, `input_scaling`, `history_depth` |
| Readout | HypercubeCNN; full N features; conv self tap **K = dim + 1** |

## Pipeline

A fixed hypercube reservoir feeds a trained HypercubeCNN readout. Each reservoir
vertex updates from an input term plus a recurrent term gathered over the M
delay-line slices of its dim neighbors, then publishes through a leaky-integrator
tanh:

```
# drive s: an input term, plus a recurrent term over the M delay-line slices
s = input_term(v)
for j in 0..M-1:            # M = history_depth — the delay line
    for i in 0..dim-1:      # dim spatial neighbors per slice
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
topology with a **self tap** at each vertex (**K = dim + 1** — neighbors plus
center; see [What is HypercubeESN?](#what-is-hypercubeesn)) and supports
regression (single/multi-output), multi-class classification, and online
streaming training.

See [docs/Reservoir.md](docs/Reservoir.md) and
[docs/Readout.md](docs/Readout.md) for full architectural detail.

## Related Work

The hypercube has met reservoir computing before. Katori (2019),
"[Reservoir Computing Based on Dynamics of Pseudo-Billiard System in
Hypercube](https://ieeexplore.ieee.org/document/8852329/)" (IJCNN 2019, Best
Paper Award), builds a reservoir from a Chaotic Boltzmann Machine: continuous
internal states move as a pseudo-billiard inside the unit hypercube
[0,1]<sup>N</sup>, with units interacting through binary, time-domain signals.
HypercubeESN uses the hypercube differently — not as the continuous space the
state trajectory lives in, but as the *wiring graph* among continuous tanh
neurons: XOR-addressed Hamming-1 connectivity (N = 2<sup>dim</sup>). Same word,
different object — Katori’s state moves *in* a cube; HypercubeESN’s activations
propagate *on* a cube.

## Install

### Python (recommended)

```bash
pip install hypercube-esn
```

Wheels for **Python 3.10–3.14** on Windows (x64), Linux (x86_64, aarch64), and
macOS (x86_64, arm64). No compiler required.

```python
import numpy as np
import hypercube_esn as he

signal = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)
esn = he.ESN(dim=7, seed=73895)  # surveyed default seed
esn.fit(signal, warmup=200)
print(f"R² = {esn.r2():.6f}")
```

Full API: [docs/Python_SDK.md](docs/Python_SDK.md) · package README:
[python/README.md](python/README.md) · runnable hosts (git tree, not in the
wheel): [python/examples/](python/examples/README.md).

### C++

**Requirements:** C++23 (GCC 13+, Clang 17+, MSVC 2022+), **CMake 4.1+**.

The HCNN readout is **vendored** in-tree (`third_party/HypercubeCNN/`, pin **v1.0.4**)
and builds as `HypercubeCNNCore` — no separate install or network fetch.

**From this repo (library + examples):**

```bash
git clone https://github.com/dliptak001/HypercubeESN.git
cd HypercubeESN
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target BasicPrediction
./build/BasicPrediction          # Windows: build\BasicPrediction.exe
```

**As a dependency (FetchContent):**

```cmake
include(FetchContent)
FetchContent_Declare(
    HypercubeESN
    GIT_REPOSITORY https://github.com/dliptak001/HypercubeESN.git
    GIT_TAG        v2.0.0          # pin a release tag
)
FetchContent_MakeAvailable(HypercubeESN)
target_link_libraries(my_app PRIVATE HypercubeESNCore)  # #include "ESN.h"
```

Installed SDK (`cmake --install` + `find_package(HypercubeESN)`): see
[docs/CPP_SDK.md](docs/CPP_SDK.md).

### Example targets

A full `cmake --build build` also produces:

| Target | Purpose |
|---|---|
| `BasicPrediction` | Minimal example: sine wave prediction |
| `SignalClassification` | Multi-class waveform recognition with confusion matrix |
| `StreamingAnomaly` | Streaming anomaly detection with recovery dynamics |
| `MemoryCapacity` | Jaeger memory-capacity diagnostic (white-noise MC sweep) |
| `NARMA` | NARMA open-loop validator (orders 30/50/70, best-5 NRMSE 0.0441 / 0.0751 / 0.1251) — [NARMA.md](examples/NARMA/NARMA.md) |
| `Lorenz` | Lorenz attractor tracking / free-run |

Start with `BasicPrediction` to see the pipeline end-to-end. Each example has a
companion `.md` file with a detailed walkthrough.

### C++ tests

| Target | Purpose |
|---|---|
| `reservoir_snapshot` | Snapshot/restore + `Create(GetConfig)` bit-identical fidelity (`tests/`) |

```bash
cmake --build build --target reservoir_snapshot
ctest --test-dir build -R reservoir_snapshot --output-on-failure
# or:  ./build/reservoir_snapshot
```

## Project Structure

```
HypercubeESN/
  CMakeLists.txt         Top-level build (core lib + examples + tests; pulls in HCNN)
  Reservoir.h/cpp        Hypercube reservoir (N = 1<<dim vertices); ReservoirConfig
  Readout.h/cpp          Learned convolutional readout (PIMPL)
  ESN.h/cpp              Unified pipeline: warmup, run, train, predict

  tests/
    reservoir_snapshot.cpp   CTest: snapshot/restore + Create(GetConfig) fidelity

  examples/
    BasicPrediction.cpp/md       Minimal sine wave prediction
    SignalClassification.cpp/md  Multi-class waveform recognition
    StreamingAnomaly.cpp/md      Streaming anomaly detection
    MemoryCapacity/              Jaeger memory-capacity diagnostic
    NARMA/                       NARMA validator (one config · N30/50/70)
    Lorenz/                      Lorenz attractor free-run (closed-loop VPT storefront)

  python/                Python bindings (pybind11 module + pyproject)
  cmake/                 Package config template (find_package support)

  docs/
    Reservoir.md          Reservoir architecture, connectivity, parameters
    Readout.md            HCNN readout: architecture, training, streaming mode
    CPP_SDK.md            C++ static-library consumer guide
    Python_SDK.md         Python SDK API reference
  third_party/
    HypercubeCNN/         Vendored HypercubeCNN v1.0.4 (read-only; see VENDORED.md)
```

## Documentation

| Document | Covers |
|---|---|
| [CHANGELOG.md](CHANGELOG.md) | **v2.0** release notes, breaking changes, migration |
| [docs/Reservoir.md](docs/Reservoir.md) | Hypercube graph, connectivity, deep-vertex history depth, leaky integrator, spectral-radius tuning, input fan-in scaling |
| [docs/Readout.md](docs/Readout.md) | HCNN readout architecture, training algorithm, streaming mode, ESN interface |
| [docs/Python_SDK.md](docs/Python_SDK.md) | Python SDK: pip install, fit/predict API, streaming, persistence |
| [docs/CPP_SDK.md](docs/CPP_SDK.md) | C++ static library: build, install, find_package usage, API reference |
| [third_party/HypercubeCNN/VENDORED.md](third_party/HypercubeCNN/VENDORED.md) | HypercubeCNN vendor pin and re-vendor rule |

Each example in `examples/` has a companion `.md` walkthrough with sample
results and interpretation guidance.
