# HypercubeESN

[![Build wheels](https://github.com/dliptak001/HypercubeESN/actions/workflows/wheels.yml/badge.svg)](https://github.com/dliptak001/HypercubeESN/actions/workflows/wheels.yml)

Python bindings for a reservoir computer whose neurons live on a Boolean hypercube — a
DIM-dimensional graph where each vertex is addressed by a DIM-bit binary
index, with all connectivity defined by XOR operations on those indices.
**Neuron states are continuous real values** (driven through `tanh`
nonlinearity); only the *addressing scheme* is binary. No adjacency list
is stored. N = 2^DIM neurons (DIM 5-16, i.e. 32 to 65,536 neurons).
DIM-invariant hyperparameters: the same spectral radius and input_scaling
work at every DIM.

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
- **DIM 5-16** -- 32 to 65,536 neurons, DIM-invariant defaults
- **HCNN readout** -- learned convolutional readout on raw reservoir state
- **Multi-input** -- multiple input channels via contiguous-block driving
- **Streaming mode** -- online training for real-time applications
- **Model persistence** -- pickle, save/load to disk

## Documentation

Full API reference: [docs/Python_SDK.md](https://github.com/dliptak001/HypercubeESN/blob/main/docs/Python_SDK.md)

Project repository: [github.com/dliptak001/HypercubeESN](https://github.com/dliptak001/HypercubeESN)
