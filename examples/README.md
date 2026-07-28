# Examples

## BasicPrediction

The minimal hello-world for HypercubeESN. Demonstrates the complete pipeline on a
sine wave: drive the reservoir, collect states, train the HCNN readout, and evaluate.

This is the place to start if you want to understand how the pieces fit together.

**What it shows:**
- ESN construction from an `ESNConfig` (hypercube dimension set via `cfg.reservoir.dim`)
- Warmup (wash out initial transients) and Run (collect states)
- HCNN readout training with cosine LR schedule
- R² and NRMSE evaluation on held-out test set

**Expected output (abbreviated):**
```
=== HypercubeESN: Sine Wave Prediction ===

  Config: N=256  raw state (all vertices)
  R2:    1.000000   (effectively perfect)
  NRMSE: ~0.00003   (sub-0.1% error)
```

**Make it yours:** Replace the sine wave generation with your own time series data.
Keep values roughly in [-1, 1] — input weights are `1/√DIM`-normalized and the
`tanh` nonlinearity bounds the state, so larger inputs just push neurons toward
saturation. Adjust `DIM` to control reservoir size, and `warmup`/`collect` to match
your data volume.

## SignalClassification

Name which of four industrial process modes is active (Cruise / Chatter / Ramp /
Spin-up — sine, square, triangle, chirp under the hood) from reservoir state alone.
DIM=8 gives strong overall ID; residual error clusters on Cruise ↔ Spin-up, with
a short lock-on delay after each mode switch. A live block-stream table reports
accuracy, softmax confidence, and time-to-lock.

**What it shows:**
- Reservoir as a feature extractor for multi-class mode ID
- HCNN native 4-class classification (softmax + cross-entropy)
- Live stream monitor: conf, TTL, LOCKED / SWITCHING / SETTLING
- Confusion matrix + early-window lock-on + mean time-to-lock

**Expected output (abbreviated):**
```
=== HypercubeESN: Signal Classification ===

Config: DIM=8  N=256  History Depth=16  ...  Classes=4

  Blk | True      Pred      | Acc%  Conf  TTL | Status
    6 | Cruise    Cruise   |   82  0.81    6 | SWITCHING
   11 | Cruise    Cruise   |  100  0.99    0 | LOCKED

Overall step accuracy: ~94.5%
  Cruise ~88%  Chatter ~99%  Ramp ~100%  Spin-up ~90%
  Mean TTL (switch blocks): ~2.1 steps
```

**Make it yours:** Add your own waveform types to `GenerateWaveform()` and increase
`NUM_CLASSES`. Adjust `block_size` to match your expected signal duration.

## StreamingAnomaly

Simulates industrial process monitoring. The reservoir learns normal process behavior
during a priming phase, then monitors a live stream in 200-step windows. Three anomaly
types are injected — noise spike, DC drift, and frequency shift — each for 3 windows,
separated by normal operation to show both detection and recovery.

**What it shows:**
- Batch training on historical "normal" data (priming)
- Anomaly detection via prediction error exceeding a threshold (10x baseline RMSE)
- Three distinct anomaly signatures with different RMSE ratios
- Automatic recovery without retraining as anomalies end
- Effect of leak rate on detection sensitivity vs recovery speed

**Expected output (abbreviated):**
```
=== HypercubeESN: Streaming Anomaly Detection ===

Config: DIM=7  N=128  History Depth=24  Leak=1  Input Scaling=0.1  Threshold=10x baseline

Baseline (prime test, RMSE): ~0.0060   threshold ~0.060

  Window | Condition   |    RMSE     Ratio | Status
      1  | Normal      |  ~0.006     ~1.0  |
      6  | Noise spike |  ~0.073    ~12.1  | ** ANOMALY **
     14  | DC drift    |  ~0.32     ~52.4  | ** ANOMALY **
     22  | Freq shift  |  ~0.38     ~62.3  | ** ANOMALY **
     25  | Normal      |  ~0.18     ~30.5  | ** ANOMALY **  (washout)
```

**Make it yours:** Replace `GenerateProcess()` with your real sensor data feed.
Adjust `normal_noise` to match your signal characteristics. Tune the
`anomaly_threshold` multiplier (10x is conservative; lower it toward 5x to
catch subtler changes at the cost of more washout-window flags).

## MemoryCapacity

A reservoir-quality **diagnostic**, not a task demo. It measures the standard
Jaeger (2001) linear memory capacity: drive the reservoir with i.i.d. white
noise, fit a ridge readout to reconstruct the input delayed by `k` steps for
each lag, and report the per-lag squared correlation `r²(k)` and the total
`MC = Σ r²(k)`. No HCNN is involved — only the raw reservoir state is read.

**What it shows:**
- How much short-term linear memory the hypercube reservoir holds, and how it
  scales with spectral radius, `history_depth`, leak rate, and DIM
- A held-out ridge readout (the canonical MC protocol), driven through four run
  modes: a single detailed per-lag curve, an `sr × leak × history_depth` grid
  sweep, a reservoir-seed survey, and a side-by-side depth probe

`main()` selects a run mode (a parallel `RunGridSweep` by default); edit the
`MCConfig` / `ReservoirConfig` and grid axes there. The walkthrough, four modes,
and multi-DIM lookup tables (archived reference campaigns) are in
[`MemoryCapacity/MemoryCapacity.md`](MemoryCapacity/MemoryCapacity.md).

## NARMA

Primary **open-loop** validator: white input `u(t)` → reconstruct NARMA-N `y(t)`.
**One fixed** config across tanh-wrapped orders 30 / 50 / 70 (**best 5 of 20**,
test NRMSE):

| Order | Best-5 mean |
|------:|------------:|
| 30 | **0.0441** |
| 50 | **0.0751** |
| 70 | **0.1251** |

CLI: `NARMA.exe [order]`. Full write-up: [`NARMA/NARMA.md`](NARMA/NARMA.md).

## Building

Example targets build automatically alongside the main harness; build the
Release tree with the bundled toolchain (see
[Building and Running](../README.md#building-and-running-c) in the project
README), then run any target from `cmake-build-release`:

```
cmake-build-release\BasicPrediction.exe
cmake-build-release\SignalClassification.exe
cmake-build-release\StreamingAnomaly.exe
cmake-build-release\MemoryCapacity.exe
cmake-build-release\NARMA.exe
cmake-build-release\Lorenz.exe
```

In CLion, select the target from the run configuration dropdown (top toolbar).
