# Streaming Anomaly Detection — Industrial Process Monitoring

## What this example demonstrates

A reservoir learns what "normal" looks like for a simulated industrial
process, then monitors a live stream for deviations. Three distinct
anomaly types are injected — a noise spike, a DC drift, and a frequency
shift — each separated by normal operation. The example shows clean
detection of all three and automatic recovery without retraining.

The HCNN readout is trained once in Phase 1 and frozen for Phase 2.
Prediction error is the anomaly signal.

This is the most practical example in the collection: it demonstrates
how a frozen reservoir model can serve as a real-time process monitor.

## Conceptual background

The core idea: if a model can predict normal behavior accurately, then
a sudden increase in prediction error signals that something has changed.
The reservoir is trained to predict the next value of a multi-harmonic
process signal. During monitoring, each window's RMSE is compared to the
baseline error from the training phase. If the ratio exceeds a threshold
(default 10x), an anomaly is flagged.

This approach has several appealing properties:

- **No anomaly labels needed.** The model only learns "normal" — it doesn't
  need examples of every possible failure mode.
- **Automatic recovery.** Once the anomaly ends, the reservoir state washes
  out and prediction error returns to baseline. No retraining required.
- **Interpretable signal.** The RMSE ratio directly indicates how different
  the current behavior is from normal. A ratio of 12x is a much stronger
  anomaly than 2x.

## The simulated process

The "normal" process is a multi-harmonic signal:

```
signal(t) = 0.6 * sin(0.1t) + 0.2 * sin(0.3t) + noise
```

where noise is uniform in [-0.01, +0.01]. Three anomaly types are injected:

| Anomaly | What changes | Magnitude | Physical analogy |
|---------|-------------|-----------|------------------|
| Noise spike | Noise level 0.01 -> 0.12 | 12x noise | Sensor degradation |
| DC drift | +0.30 offset added | Systematic bias | Sensor fouling |
| Freq shift | Frequency multiplied by 1.3 | Dynamics change | Motor speed change |

## The pipeline

```
Phase 1: Learn normal                    Phase 2: Monitor

Normal signal ──> ESN ──> HCNN train     Live signal ──> ESN ──> HCNN RMSE ──> > 10x baseline?
  4000 steps              (frozen after)   200-step               (frozen)          ALERT
                                           windows
```

**Phase 1 — Learn normal behavior:**

1. Generate 4000 steps of the normal process signal.
2. Warmup the reservoir (500 steps), then collect states (4000 steps).
3. Train the HCNN readout to predict the next value from reservoir state.
4. Measure baseline RMSE on a held-out portion of the training data.
5. Set anomaly threshold at 10x baseline.

**Phase 2 — Stream monitoring:**

1. Process the signal in 200-step windows (30 windows total).
2. For each window: run the reservoir, predict with the frozen readout,
   compute RMSE.
3. If the window RMSE exceeds the threshold, flag as anomaly.

Between windows, `ReservoirRun(..., clear_recorded=true)` clears the recorded output
buffer (so the readout can index the new window's timesteps from zero). The
reservoir neurons' live activations are not reset — they carry over,
which is what allows the model to detect when dynamics have changed
and to recover gradually when they return to normal.

## What to expect

dim=7, 128 neurons, `history_depth = 24` and `spectral_radius = 0.99`
(realized ~0.99), readout on all 128 vertices, `input_scaling = 0.1`,
`conv_channels = 16`, `leak_rate = 1.0` (full replacement — the struct
default, no override). Trained once in Phase 1 (2,800 samples,
1000 epochs, batch 64, `lr_max = 0.0015` cosine, ~28 s on a typical
Release build); Phase 2 is frozen prediction. The numbers below come from
a representative run; exact values track the random seed and HCNN init,
but the qualitative pattern is reproducible.

Baseline RMSE: ~0.0060, threshold ~0.060 (10x).

| Window | Condition   | RMSE          | Ratio   | Status                  |
|--------|-------------|---------------|---------|-------------------------|
| 1-5    | Normal      | ~0.006        | ~1.0    |                         |
| 6-8    | Noise spike | ~0.070-0.073  | ~12     | **ANOMALY**             |
| 9      | Normal      | ~0.008        | ~1.4    | mild residual           |
| 10-13  | Normal      | ~0.006        | ~1.0    | back to baseline        |
| 14-16  | DC drift    | ~0.317-0.320  | ~52-53  | **ANOMALY**             |
| 17     | Normal      | ~0.042        | ~7.0    | residual (under 10x)    |
| 18-21  | Normal      | ~0.006        | ~1.0    |                         |
| 22-24  | Freq shift  | ~0.339-0.377  | ~56-62  | **ANOMALY**             |
| 25     | Normal      | ~0.184        | ~30     | **ANOMALY** (washout)   |
| 26-30  | Normal      | ~0.006        | ~1.0    |                         |

Flagged windows: 10 (9 anomaly + 1 washout).

**What to notice:**

- **Clean detection of all three anomaly types.** Every anomaly window
  is flagged well above the 10x threshold — noise ~12x, DC drift ~52-53x,
  freq shift ~56-62x (strongest structured signal in this config).
- **Near-instant recovery from noise spike.** Random noise leaves little
  persistent state — first normal window ~1.4x, then baseline. No washout
  flag.
- **DC drift is a strong structured alarm (~52-53x).** Once the offset
  ends, one residual normal window sits at ~7x (under threshold, not
  flagged), then baseline. No washout flag.
- **One washout window after the frequency shift.** Changed dynamics leave
  a trace in the reservoir's recurrent/delay-line state that outlasts the
  anomaly by one window (~30x still flagged as "Normal"), then recovery.
  That washout is a feature, not a false positive — it confirms the
  reservoir hasn't fully stabilized yet, exactly what you want an alarm
  to signal.

### Effect of leak rate on anomaly detection

The example uses `leak_rate = 1.0` (full replacement, struct default).
A leaky integrator (`leak_rate < 1.0`) causes neurons to retain a
fraction of their previous state at each step, which would have two
qualitative effects:

**Higher sensitivity to sustained anomalies.** With full replacement, a DC
offset enters only through the current step's activation — there is no extra
leaky-integrator term holding onto it, though the recurrent and delay-line
state still carry it forward (which is what produces the one washout window
above). A leak adds that integrator term: a fraction of the old
(offset-contaminated) state persists directly, so the error compounds harder
across steps — sustained drift and dynamics changes register more strongly.

**Slower recovery after anomalies.** The sticky state takes longer to
wash out, producing more washout windows after the anomaly ends.

**Noise spike is largely unaffected.** Random noise averages out
regardless of leak rate.

For anomaly detection, slower recovery is generally a feature, not a
bug — you want the alarm to persist until the system fully stabilizes.
Try `cfg.reservoir.leak_rate = 0.4` to see the contrast.

## Things to try

- **Leak rate.** Set `cfg.reservoir.leak_rate` in the source. The
  example uses the struct default 1.0 (no override). Try 0.4 (moderate
  leak) for higher sensitivity to drift/freq shifts with slower
  washout, or 0.1 for an even longer memory.

- **HCNN epochs.** The example uses 1000. On this smooth multi-harmonic
  process signal HCNN saturates early — try lowering to 100 or 25 to
  verify. Keep `lr_max <= 0.005` to avoid denormal/NaN.

- **Lower the threshold.** Change `anomaly_threshold` from 10.0 to 5.0.
  You'll catch anomalies sooner (and pick up the drift washout window)
  but may see more false positives during washout windows.

- **Window size.** Smaller windows (e.g., 50 steps) make detection faster
  but noisier. Larger windows (e.g., 500) smooth the RMSE estimate but
  delay detection.

## A note on online training

`Readout` also supports online training for streaming applications.
For workloads that need to track drift, HCNN online training can adapt
the model incrementally — see `docs/Readout.md` for details. This
example uses frozen-readout mode to demonstrate pure anomaly detection.

## Build and run

cmake/g++ ship with CLion and are not on `PATH`. Build the Release tree with
the bundled toolchain, then run the `StreamingAnomaly` target:

```
cmake-build-release\StreamingAnomaly.exe
```
