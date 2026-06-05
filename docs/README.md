# Documentation Guide

This directory contains detailed documentation for each component of
HypercubeESN. If you're new to the project, start with the
[project README](../README.md) for an overview, then follow the reading
order below.

## Suggested reading order

### 1. Understand the architecture

| Document | What you'll learn |
|----------|-------------------|
| [Reservoir.md](Reservoir.md) | How the hypercube reservoir works — topology, connectivity, timestep mechanics, deep-vertex history depth, spectral-radius tuning, DIM-invariant input drive |
| [Readout.md](Readout.md) | HCNN readout architecture, training algorithm, and streaming mode |

These documents cover the full pipeline:

```
Input ──> Reservoir (N states) ──────────────────────> Readout ──> Prediction
           [Reservoir.md]                               [Readout.md]
```

### 2. See it in action

The `examples/` directory contains worked examples, each with a companion
`.md` walkthrough:

| Example | What it demonstrates |
|---------|---------------------|
| [BasicPrediction](../examples/BasicPrediction.md) | Simplest end-to-end demo — predict a sine wave. Start here. |
| [SignalClassification](../examples/SignalClassification.md) | Reservoir as feature extractor for multi-class waveform recognition |
| [StreamingAnomaly](../examples/StreamingAnomaly.md) | Anomaly detection in a simulated industrial process |
| [StreamingText](../examples/StreamingText/StreamingText.md) | Streaming prequential character memorization of a text corpus |
| [MemoryCapacity](../examples/MemoryCapacity/MemoryCapacity.md) | Jaeger memory-capacity benchmark — the reservoir's linear short-term memory |
| [NARMA](../examples/NARMA/NARMA.md) | Nonlinear system-identification benchmark — memory depth × nonlinear mixing |

### 3. Build with the SDK

API reference for embedding HypercubeESN in your own project:

| Document | What you'll learn |
|----------|-------------------|
| [CPP_SDK.md](CPP_SDK.md) | C++ static library: build, install, `find_package` / FetchContent, full `ESN` / `ReservoirConfig` / `ReadoutConfig` API reference |
| [Python_SDK.md](Python_SDK.md) | Python bindings: install, fit/predict, streaming, persistence |

## Key source files

For readers who prefer to learn from code, the class-level doc comments
in the header files are written for an educational audience:

| Header | Class/Functions |
|--------|----------------|
| `ESN.h` | `ESN` — the pipeline wrapper (warmup, run, collect states) |
| `Reservoir.h` | `Reservoir` — the hypercube reservoir core |
| `Readout.h` | `Readout` — learned convolutional readout |
