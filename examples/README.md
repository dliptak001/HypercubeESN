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

Classify four waveform types — sine, square, triangle, chirp — from reservoir state
alone. The HCNN readout performs native 4-class classification (softmax + cross-entropy),
producing a confusion matrix and transition dynamics analysis showing how quickly the
reservoir locks on after a waveform switch.

**What it shows:**
- Reservoir as a feature extractor for pattern recognition
- HCNN native multi-class classification
- Confusion matrix and per-class accuracy breakdown
- Transition dynamics: accuracy vs steps after waveform switch

**Expected output (abbreviated):**
```
=== HypercubeESN: Signal Classification ===

Overall accuracy: 100.0%

  Steps after switch  | Accuracy
  0 - 3               |  100.0%
  Entire block        |  100.0%
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

Config: DIM=8  N=256  History Depth=16  Leak=1  Input Scaling=1.9  Threshold=10x baseline

Baseline (prime test, RMSE): ~0.0061   threshold ~0.061

  Window | Condition   |    RMSE     Ratio | Status
      1  | Normal      |  ~0.006     ~1.0  |
      6  | Noise spike |  ~0.072    ~12.0  | ** ANOMALY **
     14  | DC drift    |  ~0.27     ~44.0  | ** ANOMALY **
     22  | Freq shift  |  ~0.23     ~38.0  | ** ANOMALY **
```

**Make it yours:** Replace `GenerateProcess()` with your real sensor data feed.
Adjust `normal_noise` to match your signal characteristics. Tune the
`anomaly_threshold` multiplier (10x is conservative; lower it toward 5x to
catch subtler changes at the cost of more washout-window flags).

## StreamingText

A streaming character-level memorization task.
The corpus (Tiny Shakespeare by default) is treated as a ring buffer and
streamed through the reservoir continuously; at each character the model
predicts the next one **before** it is folded into the online training batch
(a prequential metric), and progress is shown via teacher-forced
predicted-vs-actual samples. There is no held-out split and no serialization —
the question is pure capacity, not generalization.

**What it shows:**
- Online (streaming) readout training with a prequential rolling-BPC metric
- A self-contained `Corpus` + random `CharEmbedding` (no shared library)
- How reservoir size (DIM), depth, and readout width trade off on a
  discontinuous-input task

This is a deeper example with its own walkthrough and configuration. See
[`StreamingText/StreamingText.md`](StreamingText/StreamingText.md). All knobs
live in `StreamingText/Config.h`; the binary takes no arguments.

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
`MCConfig` / `ReservoirConfig` there to probe a different operating point. The
full walkthrough — protocol, the two-config split, and a worked DIM-11 sweep —
is in [`MemoryCapacity/MemoryCapacity.md`](MemoryCapacity/MemoryCapacity.md).

## NARMA

A classic reservoir **benchmark** for nonlinear system identification. The
reservoir is driven by white noise `u(t)` and the readout must reconstruct the
NARMA-N output `y(t)`, whose recurrence couples a long nonlinear history of its
own past with a delayed copy of the input — so it stresses memory depth and
nonlinear mixing at once. Unlike MemoryCapacity (which isolates *linear*
memory), this is the full ESN pipeline with a trained HCNN readout.

**What it shows:**
- The canonical NARMA task with correct input/target alignment (system
  identification, not one-step forecasting)
- A `history_depth (M) × reservoir-seed` sweep proving the task is
  memory-bound — reconstruction quality tracks delay-line depth, with the
  optimum just above the NARMA order (too few taps starve the lag history, too
  many dilute the readout, so the error curve is U-shaped)
- A `tanh`-wrapped vs. legacy coefficient-schedule A/B switch
  (`NARMA_TANH_WRAP`) so difficulty scales honestly with order

**Expected output (abbreviated):**
```
=== HypercubeESN: NARMA-30 history_depth (M) x seed sweep ===

  Variant: tanh-wrapped (fixed coeffs -- honest order-scaling)
  Config: DIM=10 N=1024  sr=0.92 leak=1 input_scaling=0.5

    M     mean
    ----  -------
       1   0.7824
      16   0.1257
      32   0.0885
```

Edit the `constexpr` parameters in `main()` (order, DIM, sweep points) to probe
a different regime. Full walkthrough — the recurrence, coefficient schedule,
literature reference bands, and the target-alignment fix — is in
[`NARMA/NARMA.md`](NARMA/NARMA.md).

## Building

All six example targets build automatically alongside the main benchmark; build
the Release tree with the bundled toolchain (see
[Building and Running](../README.md#building-and-running-c) in the project
README), then run any target from `cmake-build-release`:

```
cmake-build-release\BasicPrediction.exe
cmake-build-release\SignalClassification.exe
cmake-build-release\StreamingAnomaly.exe
cmake-build-release\MemoryCapacity.exe
cmake-build-release\NARMA.exe
cmake-build-release\StreamingText.exe
```

In CLion, select the target from the run configuration dropdown (top toolbar).
