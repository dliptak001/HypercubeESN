# HypercubeESN

[![Build wheels](https://github.com/dliptak001/HypercubeESN/actions/workflows/wheels.yml/badge.svg)](https://github.com/dliptak001/HypercubeESN/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/hypercube_esn)](https://pypi.org/project/hypercube-esn/)
[![Python](https://img.shields.io/pypi/pyversions/hypercube_esn)](https://pypi.org/project/hypercube-esn/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/dliptak001/HypercubeESN/blob/main/LICENSE)

Python bindings for **HypercubeESN** — reservoir computing on a Boolean
hypercube. Neurons sit on the vertices, each carrying a short delay line of its
own past, wired to single-bit-flip neighbors by XOR.

Three properties follow:

- **A topology you don't store.** Connectivity is implicit in the vertex indices —
  no adjacency list.
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

### NARMA

tanh-wrapped orders 30 / 50 / 70; **Same operating point (same dim, sr, memory depth, reservoir seed, ...) for all three orders.**

| Order | Best-5 mean test NRMSE |
|------:|-----------------------:|
| 30 |             **0.0441** |
| 50 |             **0.0751** |
| 70 |             **0.1251** |

[Campaign write-up](https://github.com/dliptak001/HypercubeESN/blob/main/examples/NARMA/NARMA.md)

### Memory capacity (Jaeger MC)

Linear short-term memory (ridge on reservoir state — not HCNN). **Tunable** via
dim, memory depth, and spectral radius.

| dim | N | Peak TotalMC |
|----:|--:|-------------:|
| 5 | 32 | ~30 |
| 8 | 256 | ~250 |
| 10 | 1024 | ~820 |
| 12 | 4096 | ~1380 |

[MemoryCapacity](https://github.com/dliptak001/HypercubeESN/blob/main/examples/MemoryCapacity/MemoryCapacity.md)

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

## Installation

```bash
pip install hypercube-esn
```

Pre-built wheels for Python **3.10–3.14** on Windows (x64), Linux (x86_64,
aarch64), and macOS (x86_64, arm64). No compiler required for wheels.

## Quick start

```python
import numpy as np
import hypercube_esn as he

signal = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)
esn = he.ESN(dim=7, seed=73895)
esn.fit(signal, warmup=200)
print(f"R² = {esn.r2():.6f}")
print(f"NRMSE = {esn.nrmse():.6f}")
```

## Examples

Runnable Python hosts (public API only — no CMake / no C++ example binaries).
Scripts live in the **git tree** under
[`python/examples/`](https://github.com/dliptak001/HypercubeESN/tree/main/python/examples);
they are **not** installed by the PyPI wheel.

| Script | |
|--------|--|
| [basic_prediction.py](https://github.com/dliptak001/HypercubeESN/blob/main/python/examples/basic_prediction.py) | Next-step sine prediction (`fit` → R² / NRMSE) |
| [classification.py](https://github.com/dliptak001/HypercubeESN/blob/main/python/examples/classification.py) | Binary sign classification (`accuracy`) |

```bash
# clone HypercubeESN, then from the repo root:
pip install hypercube-esn
python python/examples/basic_prediction.py
python python/examples/classification.py
```

Onboarding demos only — easy synthetic signals, not storefront metrics. Frozen
NARMA / MemoryCapacity / Lorenz numbers live in the C++
[`examples/`](https://github.com/dliptak001/HypercubeESN/tree/main/examples)
campaigns. Index:
[python/examples/README.md](https://github.com/dliptak001/HypercubeESN/blob/main/python/examples/README.md).

## Features

- **Simple API** — `fit()` runs warmup, collect, and batch train in one call
- **Hypercube dim 5–16** — N = 2<sup>dim</sup> neurons (32…65,536); delay-line depth M
- **HCNN readout (self tap)** — conv on the hypercube with **K = dim + 1**
  (neighbors + center); 2.0 upgrade vs neighbor-only kernels; not ridge alone
- **Multi-slice readout** — optional B ages packed into the readout (`readout_slices`)
- **Multi-input** — channels map to contiguous vertex blocks
- **Closed-loop drive** — external feedback channels + `reservoir_step`
- **Regression & classification** — same `ESN` surface, task-selected head
- **Streaming** — online `train_step` / `train_step_batch`
- **Persistence** — pickle, `save` / `load`, optional HCNW export
- **Wheels** — Python 3.10–3.14 on Windows, Linux, macOS (no local C++ toolchain)

## Documentation

| Doc | |
|-----|--|
| [Python SDK](https://github.com/dliptak001/HypercubeESN/blob/main/docs/Python_SDK.md) | API reference (`fit`, streaming, config, pickle) |
| [Python examples](https://github.com/dliptak001/HypercubeESN/blob/main/python/examples/README.md) | Runnable hosts (git tree; not in the wheel) |
| [NARMA campaign](https://github.com/dliptak001/HypercubeESN/blob/main/examples/NARMA/NARMA.md) | Open-loop validator (N30 / N50 / N70) |
| [Project README](https://github.com/dliptak001/HypercubeESN#readme) | Architecture and C++ side |
| [C++ SDK](https://github.com/dliptak001/HypercubeESN/blob/main/docs/CPP_SDK.md) | Native library |

Repository: [github.com/dliptak001/HypercubeESN](https://github.com/dliptak001/HypercubeESN)

## License

Apache-2.0
