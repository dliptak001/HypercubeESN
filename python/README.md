# HypercubeESN

[![Build wheels](https://github.com/dliptak001/HypercubeESN/actions/workflows/wheels.yml/badge.svg)](https://github.com/dliptak001/HypercubeESN/actions/workflows/wheels.yml)

Python bindings for **HypercubeESN** — an Echo State Network whose reservoir is a
Boolean hypercube. Each neuron sits on a vertex and connects only to its
Hamming-distance-1 neighbors, with every adjacency resolved by a single XOR on
the vertex's binary index: a deterministic O(1) lookup that stores nothing at
all. There is no adjacency list to build, store, or serialize — the connectivity
is implicit in the indices. The neurons themselves stay familiar continuous
`tanh` units; only the *addressing* is binary, the dynamics fully real-valued.

That same implicit addressing extends into time: each update also reaches back
through an addressable delay line of its neighbors' last M states, so temporal
memory is intrinsic to the topology — memory by construction rather than by luck.
N = 2^DIM neurons (DIM 5-16, i.e. 32 to 65,536).

## Installation

```bash
pip install hypercube-esn
```

Pre-built wheels for Python 3.10-3.13 on Windows (x64), Linux (x86_64,
aarch64), and macOS (x86_64, arm64). No compiler required.

## Quick Start

```python
import numpy as np
import hypercube_esn as he

# One-step-ahead sine prediction
signal = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)
esn = he.ESN(dim=7)
esn.fit(signal, warmup=200)
print(f"R2 = {esn.r2():.6f}")
print(f"NRMSE = {esn.nrmse():.6f}")
```

## Features

- **Simple API** -- `fit()` handles warmup, run, and train in one call
- **DIM 5-16** -- 32 to 65,536 neurons
- **HCNN readout** -- learned convolutional readout on raw reservoir state
- **Multi-input** -- multiple input channels via contiguous-block driving
- **Streaming mode** -- online training for real-time applications
- **Model persistence** -- pickle, save/load to disk

## Documentation

Full API reference: [docs/Python_SDK.md](https://github.com/dliptak001/HypercubeESN/blob/main/docs/Python_SDK.md)

Project repository: [github.com/dliptak001/HypercubeESN](https://github.com/dliptak001/HypercubeESN)
