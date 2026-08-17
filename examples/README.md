# Examples

## BasicPrediction

The minimal hello-world for HypercubeESN. Demonstrates the complete pipeline on
a sine wave: drive the reservoir, collect states, train the HCNN readout, and
evaluate.

This is the place to start if you want to understand how the pieces fit
together.

**What it shows:**
- ESN construction from an `ESNConfig`
- Warmup (wash out initial transients) and Run (collect states)
- HCNN readout training with cosine LR schedule
- R² and NRMSE evaluation on held-out test set

---

## SignalClassification

Name which of four industrial process modes is active (Cruise / Chatter / Ramp /
Spin-up — sine, square, triangle, chirp under the hood) from reservoir state
alone. dim=8 gives strong overall ID; residual error clusters on Cruise ↔
Spin-up, with a short lock-on delay after each mode switch. A live block-stream
table reports accuracy, softmax confidence, and time-to-lock.

**What it shows:**
- Reservoir as a feature extractor for multi-class mode ID
- HCNN native 4-class classification (softmax + cross-entropy)
- Live stream monitor: conf, TTL, LOCKED / SWITCHING / SETTLING
- Confusion matrix + early-window lock-on + mean time-to-lock

---

## StreamingAnomaly

Simulates industrial process monitoring. The reservoir learns normal process
behavior during a priming phase, then monitors a live stream in 200-step
windows. Three anomaly types are injected — noise spike, DC drift, and
frequency shift — each for 3 windows, separated by normal operation to show
both detection and recovery.

**What it shows:**
- Batch training on historical "normal" data (priming)
- Anomaly detection via prediction error exceeding a threshold (10x baseline RMSE)
- Three distinct anomaly signatures with different RMSE ratios
- Automatic recovery without retraining as anomalies end

---

## MemoryCapacity

Measures the standard Jaeger (2001) linear memory capacity. No HCNN is involved
— only the raw reservoir state is read.

**What it shows:**
- How much short-term linear memory the hypercube reservoir holds, and how it
  scales with spectral radius, `history_depth`, leak rate, and dim

---

## NARMA

Primary **open-loop** validator: white input `u(t)` → reconstruct NARMA-N `y(t)`.
**One fixed** config across tanh-wrapped orders 30 / 50 / 70 (**best 5 of 20**,
test NRMSE). Full write-up: [`NARMA/NARMA.md`](NARMA/NARMA.md).

---

## Lorenz

Primary **closed-loop** validator: Lorenz-63 free-run via **input-bank
self-feedback** (predicted `[x, y, z, x*z]` re-injected as the next drive).
VPT at θ = 0.25. Full write-up: [`Lorenz/README.md`](Lorenz/README.md).

---

## Building

Example targets build automatically alongside the main harness; build the
Release tree with the bundled toolchain, then run any target from `cmake-build-release`:

```
cmake-build-release\BasicPrediction.exe
cmake-build-release\SignalClassification.exe
cmake-build-release\StreamingAnomaly.exe
cmake-build-release\MemoryCapacity.exe
cmake-build-release\NARMA.exe
cmake-build-release\Lorenz.exe
```

In CLion, select the target from the run configuration dropdown (top toolbar).
