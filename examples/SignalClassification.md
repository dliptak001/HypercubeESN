# Signal Classification — Process Mode Recognition

## What this example demonstrates

A single vibration channel streams through the reservoir while an industrial
process switches among four operating modes. The HCNN readout names the
active mode from **reservoir state alone** — not from raw samples. The demo
shows multi-class classification with a live block-stream monitor (confidence
+ time-to-lock), a structured confusion matrix, and a residual hard pair
(Cruise ↔ Spin-up) even when overall accuracy is high.

## Conceptual background

Reservoir computing is often presented as a time-series prediction tool, but
the reservoir state is also a strong feature vector for classification. At
each timestep the N-dimensional state encodes recent input history — and
different waveforms leave different trajectories through state space.

You do not design those features by hand. Nonlinear dynamics and fading
memory lift the raw channel into a space where mode classes become
separable. With enough capacity, Chatter and Ramp go near-perfect; the
interesting residual is the smooth-shape pair (Cruise vs Spin-up) and the
few steps of lock-on delay after each mode switch.

**HCNN: native multi-class.** The CNN readout uses `num_outputs=4` and
`ReadoutTask::Classification` (softmax + cross-entropy). One readout handles
all four modes.

## The four process modes

| Mode | Shape | Frequency | Physical analogy |
|------|-------|-----------|------------------|
| Cruise | sine | 0.11 | Smooth harmonic load |
| Chatter | square | 0.13 | Hard-edged clutch / relay chatter |
| Ramp | triangle | 0.12 | Linear load ramp |
| Spin-up | chirp | 0.10 | Accelerating runaway |

Frequencies are deliberately close. Uniform noise in [-0.18, +0.18] is
added so the readout must key on **shape**, not tone. Blocks of 40 steps
are drawn in **random order** with no immediate class repeat — a stream of
mode switches rather than a fixed 0→1→2→3 cycle.

## The pipeline

```
Mode blocks ──> Reservoir ──> HCNN: 4-class softmax ──> Mode label
  40 steps        256 neurons    (dim=8)                 + confidence
  random order    (fixed)        frozen after Phase 1    + time-to-lock
```

**Phase 1 — Learn the modes**

1. Build 600 random-order mode blocks (24,000 steps) + 300-step warmup.
2. Warmup, then collect reservoir states for train+test (70/30 by block).
3. Train one 4-class HCNN readout on 420 blocks (16,800 steps), 100 epochs.

**Phase 2 — Monitor the held-out stream**

1. Score every test step (argmax + softmax confidence).
2. Print a live table for the first 24 blocks: true / majority pred /
   block accuracy / mean conf / time-to-lock / status.
3. Aggregate: overall accuracy, per-mode breakdown, confusion matrix,
   early-window lock-on curve, TTL stats on switch blocks.
4. Print a short **What happened** epilogue derived from those numbers.

**Time-to-lock (TTL):** first step index where the next `K=3` predictions
are all correct. Captures how long residual state from the previous mode
delays a stable ID after a switch.

## What to expect

dim=8, 256 neurons, `history_depth = 16`, `spectral_radius = 0.95`
(realized ~0.95), `input_scaling = 0.1`, `leak_rate = 1.0` (struct default),
Trained 100 epochs (batch 32, `lr_max = 0.0015` cosine, ~23 s on a
typical Release build). Exact figures track seed and HCNN init; the
qualitative pattern is reproducible.

**Overall step accuracy: ~94.5%** (7,200 test steps / 180 blocks)

| Mode | Accuracy | Notes |
|------|----------|-------|
| Cruise | ~88% | Hard pair with Spin-up (~12% of Cruise steps) |
| Chatter | ~99% | Strong |
| Ramp | ~100% | Near-perfect (sharp linear shape) |
| Spin-up | ~90% | Second-hardest; leaks into Cruise (~10%) |

Representative confusion (row = actual, % of row):

| Actual \ Pred | Cruise | Chatter | Ramp | Spin-up |
|---------------|--------|---------|------|---------|
| Cruise | ~88 | ~0 | ~0 | ~12 |
| Chatter | ~1 | ~99 | ~0 | ~0 |
| Ramp | ~0 | ~0 | ~100 | ~0 |
| Spin-up | ~10 | ~0 | ~0 | ~90 |

**Lock-on (accuracy in first M steps of each block):**

| Window | Accuracy |
|--------|----------|
| 0–3 | ~84% |
| 0–5 | ~84% |
| 0–10 | ~85% |
| 0–20 | ~89% |
| Entire block | ~94.5% |

**TTL (switch blocks, K=3 consecutive correct):** mean ~2.1 steps, max ~17;
all switch blocks eventually lock under this config.

**What to notice**

- **Strong overall, structured residual.** Chatter and Ramp are essentially
  solved; almost all remaining error sits in Cruise ↔ Spin-up.
- **Cruise ↔ Spin-up is the hard pair.** Smooth harmonic vs accelerating
  chirp share early-block shape under noise.
- **Early window costs more.** First 3–10 steps after a switch sit ~10
  points below full-block accuracy — residual dynamics from the previous
  mode, not a broken classifier. Mean TTL ~2 steps.
- **Stream statuses.** `LOCKED` = high block accuracy + confidence;
  `SWITCHING` = TTL still counting after a mode change; `SETTLING` /
  `CONFUSED` / `NO LOCK` mark partial or failed locks.

### Making it harder / easier

- **Easier:** more epochs, lower `NOISE_LEVEL` (e.g. 0.10), or longer
  `block_size` (more steady-state samples per mode).
- **Harder:** drop dim to 6 (N=64), `NOISE_LEVEL` 0.25+, `block_size` 15–20,
  or `leak_rate < 1.0` so the previous mode hangs longer into the next block.
- **More memory hangover:** try `cfg.reservoir.leak_rate = 0.5` and watch
  mean TTL and the 0–5 window climb.

## Things to try

- **dim.** Default is 8. Try 6 (capacity-starved residual grows) or 5.
- **Noise.** `NOISE_LEVEL` is 0.18. Raise toward 0.25+ or cut to 0.05.
- **Block size.** Default 40. Short blocks make TTL dominate the score.
- **Epochs.** Default 100. Try 50 for a quicker / slightly weaker readout.
- **Leak rate.** Struct default 1.0. Try 0.65 for slower washout after switches.
- **Lock streak `lock_k`.** Default 3. Raise to 5 for a stricter “locked” definition.

## Build and run

cmake/g++ ship with CLion and are not on `PATH`. Build the Release tree with
the bundled toolchain (see [Building and Running](../README.md#building-and-running-c)
in the project README), then run the `SignalClassification` target:

```
cmake-build-release\SignalClassification.exe
```
