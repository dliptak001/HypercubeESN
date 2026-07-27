# Examples

## Full-state feedback A/B (all demos)

Every example can enable the optional **full-state linear feedback** port for
A/B testing against the open-loop baseline. Knobs are **local to each example**
(no shared header): set `full_state_feedback`, `fsf_seed`, and `fsf_scaling` on
the `ReservoirConfig` / `ESNConfig` (or `config::` in Lorenz) next to the other
reservoir fields.

**How to A/B**

1. **Off:** `full_state_feedback = false` (default in the demos).
2. **On:** `full_state_feedback = true` — V from `fsf_seed`; strength =
   `fsf_scaling` on B_fsf.

Logs print a NARMA-style line: `FSF: OFF` or
`FSF: ON   fsf_seed=…  fsf_scaling=…` (NARMA also lists a seed sweep when on).
Theory/API: [docs/full_state_linear_feedback.md](../docs/full_state_linear_feedback.md).

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
  FSF: OFF

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
  FSF: OFF

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
