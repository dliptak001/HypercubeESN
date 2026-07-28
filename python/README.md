# HypercubeESN

[![Build wheels](https://github.com/dliptak001/HypercubeESN/actions/workflows/wheels.yml/badge.svg)](https://github.com/dliptak001/HypercubeESN/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/hypercube_esn)](https://pypi.org/project/hypercube-esn/)
[![Python](https://img.shields.io/pypi/pyversions/hypercube_esn)](https://pypi.org/project/hypercube-esn/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/dliptak001/HypercubeESN/blob/main/LICENSE)

Python bindings for **HypercubeESN** — reservoir computing on a Boolean
hypercube. Neurons sit on the vertices, wired to their single-bit-flip neighbors
by XOR. That topology is **never stored, only computed.** **DIM** is the
hypercube dimension; N = 2<sup>DIM</sup> continuous `tanh` units (DIM 5–16 → 32
to 65,536 neurons).

Three properties follow:

- **A topology you don't store.** Connectivity is implicit in the vertex indices —
  no adjacency list, at any size.
- **Hidden multi-scale structure.** Full neighbor connectivity with random weights
  turns the cube into nested clusters — local, regional, and global at once —
  that nobody designed in.
- **Memory you can address.** Each vertex carries a delay line of its own recent
  past, so the reservoir remembers *specific* lags by construction, not by lucky
  echoes.

The reservoir state is a *signal on that graph*, not an anonymous vector. What
reads it is [HypercubeCNN](https://github.com/dliptak001/HypercubeCNN) —
convolutions on the same vertices and XOR neighborhoods, not a ridge fit on a
flat state and not an image CNN on a fabricated 2D grid. The pairing is
topology-native: the readout consumes the reservoir with zero distortion, and
the learned kernels exploit the locality that generated the dynamics. **The data
never leaves the hypercube it was born on.**

## Headline results

Primary validators — open-loop (NARMA, MC) and half-anchored free-run (Lorenz).

### NARMA (open-loop system ID)

One fixed config, tanh-wrapped orders 30 / 50 / 70; **best 5 of 20** seeds
(test NRMSE). Same op-point for all three orders.

| Order | Best-5 mean |
|------:|------------:|
| 30 | **0.0441** |
| 50 | **0.0751** |
| 70 | **0.1251** |

[Campaign write-up](https://github.com/dliptak001/HypercubeESN/blob/main/examples/NARMA/NARMA.md)

### Memory capacity (Jaeger MC)

Linear short-term memory (ridge on reservoir state — not HCNN). **Tunable** via
DIM, delay-line depth M, and spectral radius: peak TotalMC from about **30**
(DIM 5) to **1400+** (DIM 12) in the reference grids; small cubes sit near the
theoretical ceiling (MC/F ≈ 1).

| DIM | N | Peak TotalMC |
|----:|--:|-------------:|
| 5 | 32 | ~30 |
| 8 | 256 | ~250 |
| 10 | 1024 | ~820 |
| 12 | 4096 | ~1380 |

[MemoryCapacity](https://github.com/dliptak001/HypercubeESN/blob/main/examples/MemoryCapacity/MemoryCapacity.md)

### Lorenz (half-anchored free-run)

Janus dual-cursor train; free-run with real past + self-feedback on the future
port. Report VPT / free-run RMSE with the half-anchored protocol stated.

| Metric | Result |
|--------|--------|
| VPT (Lyapunov times) / protocol | **TBD** |

[Lorenz](https://github.com/dliptak001/HypercubeESN/blob/main/examples/Lorenz/README.md)

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
esn = he.ESN(reservoir_hypercube_dimension=7)
esn.fit(signal, warmup=200)
print(f"R² = {esn.r2():.6f}")
print(f"NRMSE = {esn.nrmse():.6f}")
```

## Features

- **Simple API** — `fit()` runs warmup, collect, and batch train in one call
- **Hypercube dimension DIM 5–16** — N = 2<sup>DIM</sup> neurons (32 to 65,536); delay-line depth M
- **HCNN readout** — convolutional readout on the hypercube (not ridge alone)
- **Multi-input** — channels map to contiguous vertex blocks
- **Streaming** — online `train_step` / `train_step_batch` for continuous data
- **Persistence** — pickle, `save` / `load`, optional HCNW export

## Documentation

| Doc | |
|-----|--|
| [Python SDK](https://github.com/dliptak001/HypercubeESN/blob/main/docs/Python_SDK.md) | API reference (`fit`, streaming, config, pickle) |
| [NARMA campaign](https://github.com/dliptak001/HypercubeESN/blob/main/examples/NARMA/NARMA.md) | Open-loop validator (N30 / N50 / N70) |
| [Project README](https://github.com/dliptak001/HypercubeESN#readme) | Architecture and C++ side |
| [C++ SDK](https://github.com/dliptak001/HypercubeESN/blob/main/docs/CPP_SDK.md) | Native library |

Repository: [github.com/dliptak001/HypercubeESN](https://github.com/dliptak001/HypercubeESN)

## License

Apache-2.0
