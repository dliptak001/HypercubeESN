# Signal Classification — Waveform Recognition

## What this example demonstrates

The reservoir acts as a feature extractor for pattern recognition.
Four waveform types — sine, square, triangle, chirp — are fed to the
reservoir in alternating blocks. The HCNN readout classifies which
waveform is active at each timestep, using only the reservoir's
internal state.

This example focuses on multi-class classification, with a confusion
matrix and transition dynamics analysis.

## Conceptual background

Reservoir computing is often presented as a time series prediction tool,
but the reservoir state is also a powerful feature vector for classification.
At any given timestep, the N-dimensional state encodes the recent input
history — and different waveforms produce different trajectories through
state space.

The key insight: you don't need to design features by hand. The reservoir's
nonlinear dynamics and fading memory automatically transform the raw input
into a high-dimensional representation where different signal classes become
separable.

**HCNN: native multi-class.** The CNN readout supports multi-class
natively via `num_outputs=4` and `ReadoutTask::Classification`, using
softmax + cross-entropy loss. A single readout handles all four classes.

## The four waveforms

| Waveform | Frequency | Character |
|----------|-----------|-----------|
| Sine | 0.11 | Smooth, continuous |
| Square | 0.13 | Discontinuous jumps |
| Triangle | 0.12 | Piecewise linear |
| Chirp | 0.10 | Accelerating frequency |

Frequencies are deliberately close together and 15% uniform noise is
added to the signal. This forces the readout to classify by waveform
shape rather than frequency alone.

## The pipeline

```
Waveform blocks ──> Reservoir ──> HCNN: 4-class softmax ──> Class
  40 steps each      256 neurons    (all 256 vertices)       0,1,2,3
                     (fixed)
```

**Step by step:**

1. **Generate signal** — 75 full cycles through all 4 waveforms (300 blocks
   total, 40 steps each = 12,000 timesteps). Each block starts at phase 0.
   Uniform noise in [-0.15, +0.15] is added to every sample.

2. **Warmup** — 300 steps of sine to wash out initial conditions.

3. **Collect** — 12,000 steps with per-step class labels (train=8,400,
   test=3,600 at 70/30 split).

4. **Train** — Single 4-class HCNN readout on 70% of the data, trained
   with Adam and cosine LR schedule.

5. **Evaluate** — Confusion matrix, per-class accuracy, and transition
   dynamics (how quickly the reservoir locks onto a new waveform after a
   block switch).

## What to expect

### Default configuration

DIM=8, 256 neurons, `history_depth = 16` and `spectral_radius = 0.99`
(realized ~0.99), `input_scaling = 1.5`, `leak_rate = 1.0` (full replacement —
the struct default, no override), readout on all 256 vertices. Trained for
50 epochs (batch 32, `lr_max = 0.0015` cosine, ~4 s). At this capacity the
task is trivially separable.

**Overall accuracy: 100.0%** (3,600 test samples)

| Class | Accuracy | Notes |
|-------|----------|-------|
| Sine | 100.0% (880/880) | Perfectly separable |
| Square | 100.0% (880/880) | Perfectly separable |
| Triangle | 100.0% (920/920) | Perfectly separable |
| Chirp | 100.0% (920/920) | Perfectly separable |

The confusion matrix is diagonal — zero off-diagonal entries.

### Transition lock-on

| Steps after switch | Accuracy |
|--------------------|----------|
| 0-3 | 100.0% |
| 0-5 | 100.0% |
| 0-10 | 100.0% |
| 0-20 | 100.0% |
| Entire block | 100.0% |

Instant lock-on after each block transition. With `leak_rate = 1.0`,
neurons fully replace their state at each step — there's no carryover
from the previous waveform to slow recognition of the new one.

### Making the task harder

This default configuration leaves no headroom to study. To recover an
interesting tradeoff:

- **Drop DIM.** At DIM=5 (32 neurons) the reservoir's representational
  capacity becomes the limit and per-class errors appear.
- **Shrink blocks.** Smaller `block_size` (e.g., 10-20) means the
  transition zone dominates each block; lock-on speed starts to matter.
- **Add a leaky integrator.** `cfg.reservoir.leak_rate < 1.0` makes
  old-waveform dynamics persist briefly into a new block, slowing
  lock-on and producing measurable transition-zone errors.
- **More noise.** Raising `NOISE_LEVEL` toward 0.25+ blurs the shape
  cues the readout relies on.

## Things to try

- **Leak rate.** Set `cfg.reservoir.leak_rate` in the source. The
  example uses the struct default 1.0 (instant lock-on, no
  carryover). Try 0.65 (some history retention, slower transitions)
  or 0.3 (longer memory, may improve steady-state accuracy on harder
  variants but hurts lock-on).

- **Block size.** The default is 40 steps. Try 150 for an easier task
  (more context per block) or 20 for a harder one (transition zone
  dominates each block).

- **Noise level.** `NOISE_LEVEL` is 0.15. Raise to 0.25+ for a harder
  task, or remove noise entirely to see the floor.

- **DIM.** The default is DIM=8 (256 neurons). Try DIM=5 (32 neurons)
  to see capacity-limited behavior with visible per-class confusion.

- **HCNN epochs.** The example uses 50. Try 25 to see undertrained
  behavior; 200 or 1000 to push the loss further on harder variants.

## Build and run

cmake/g++ ship with CLion and are not on `PATH`. Build the Release tree with
the bundled toolchain (see [Building and Running](../README.md#building-and-running-c)
in the project README), then run the `SignalClassification` target:

```
cmake-build-release\SignalClassification.exe
```
