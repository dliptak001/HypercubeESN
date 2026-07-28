# HypercubeESN

[![Build wheels](https://github.com/dliptak001/HypercubeESN/actions/workflows/wheels.yml/badge.svg)](https://github.com/dliptak001/HypercubeESN/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/hypercube_esn)](https://pypi.org/project/hypercube-esn/)
[![Python](https://img.shields.io/pypi/pyversions/hypercube_esn)](https://pypi.org/project/hypercube-esn/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/dliptak001/HypercubeESN/blob/main/LICENSE)

Python bindings for **HypercubeESN** — reservoir computing on a Boolean
hypercube. Neurons sit on vertices and connect only to Hamming-distance-1
neighbors via XOR addressing: no adjacency list. **DIM** is the hypercube
dimension; there are N = 2<sup>DIM</sup> continuous `tanh` units (DIM 5–16 → 32
to 65,536 neurons). An addressable delay line of depth M gives temporal memory
by construction. The reservoir state is a *signal on that graph*, not an
anonymous vector.

What reads it is [HypercubeCNN](https://github.com/dliptak001/HypercubeCNN) —
convolutions on the same vertices and XOR neighborhoods, not a ridge fit on a
flat state and not an image CNN on a fabricated 2D grid. The pairing is
topology-native: the readout consumes the reservoir with zero distortion, and
the learned kernels exploit the locality that generated the dynamics. **The data
never leaves the hypercube it was born on.** Freeze the reservoir; train the
head.

**Headline result** — one fixed config, tanh-wrapped NARMA, **best 5 of 20**
seeds (test NRMSE):

| Order | Best-5 mean |
|------:|------------:|
| 30 | **0.0441** |
| 50 | **0.0751** |
| 70 | **0.1251** |

Same op-point for all three orders; only the NARMA order changes.
[Campaign write-up](https://github.com/dliptak001/HypercubeESN/blob/main/examples/NARMA/NARMA.md)
· [project README](https://github.com/dliptak001/HypercubeESN#headline-result--one-fixed-config-tanh-wrapped-narma-best-5-of-20-seeds-test-nrmse)

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
