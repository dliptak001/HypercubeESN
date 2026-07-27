# Documentation Guide

This directory contains detailed documentation for each component of
HypercubeESN. If you're new to the project, start with the
[project README](../README.md) for an overview, then follow the reading
order below.

## Suggested reading order

### 1. Understand the architecture

| Document | What you'll learn |
|----------|-------------------|
| [Reservoir.md](Reservoir.md) | Hypercube reservoir — topology, timestep, history depth, SR, drive ports (input / external feedback) |
| [Readout.md](Readout.md) | HCNN readout architecture, training algorithm, and streaming mode |
| [reservoir_feedback_mechanism.md](reservoir_feedback_mechanism.md) | External-feedback port (caller-owned closed loop) |
| [ActivationFunctionA.md](ActivationFunctionA.md) | **Archive** — central-slope tanh envelope experiments (implementation removed) |
| [Rotating-input-map-temporalization.md](Rotating-input-map-temporalization.md) | **Design proposal** — RIMT mechanism for **HypercubeMLP** (own product; static/MLP-class tasks atop HypercubeESN) |
| [HypercubeLSM.md](HypercubeLSM.md) | **Concept** — HypercubeLSM: spiking liquid state machine on the hypercube (ESN’s event-native sibling; own future project) |

These documents cover the full pipeline:

```
Input (+ optional external feedback)
        ──> Reservoir (N states) ──> Readout ──> Prediction
             [Reservoir.md]          [Readout.md]
```

### 2. See it in action

The `examples/` directory contains worked examples, each with a companion
`.md` walkthrough:

| Example | What it demonstrates |
|---------|---------------------|
| [BasicPrediction](../examples/BasicPrediction.md) | Simplest end-to-end demo — predict a sine wave. Start here. |
| [SignalClassification](../examples/SignalClassification.md) | Process-mode ID (Cruise/Chatter/Ramp/Spin-up); conf + TTL stream |
| [StreamingAnomaly](../examples/StreamingAnomaly.md) | Anomaly detection in a simulated industrial process |
| [MemoryCapacity](../examples/MemoryCapacity/MemoryCapacity.md) | Jaeger memory-capacity benchmark — the reservoir's linear short-term memory |
| [NARMA](../examples/NARMA/NARMA.md) | Nonlinear system-identification benchmark — memory depth × nonlinear mixing |

### 3. Build with the SDK

API reference for embedding HypercubeESN in your own project:

| Document | What you'll learn |
|----------|-------------------|
| [CPP_SDK.md](CPP_SDK.md) | C++ static library: build, install, `find_package` / FetchContent, full `ESN` / `ReservoirConfig` / `ReadoutConfig` API reference |
| [Python_SDK.md](Python_SDK.md) | Python bindings: install, fit/predict, streaming, persistence |

### 4. Maintainers (dependency)

HypercubeCNN is vendored under `third_party/HypercubeCNN/` (pin and re-vendor rule:
[VENDORED.md](../third_party/HypercubeCNN/VENDORED.md)). Do not hand-edit the snapshot.

## Key source files

For readers who prefer to learn from code, the class-level doc comments
in the header files are written for an educational audience:

| Header | Class/Functions |
|--------|----------------|
| `ESN.h` | `ESN` — the pipeline wrapper (warmup, run, collect states) |
| `Reservoir.h` | `Reservoir` — the hypercube reservoir core |
| `Readout.h` | `Readout` — learned convolutional readout |
