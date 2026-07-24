# StreamingText — Streaming Memorization

## Goal

Drive a hypercube reservoir continuously over one corpus and measure how well
the learned readout reproduces it. There is no held-out set and no notion of
generalization here — the question is *capacity*: how much of a fixed text can
this reservoir + readout commit to, and how does that scale with reservoir
size (DIM), depth (`history_depth`), and readout width (`conv_channels`)?

This example is intentionally small. It exists to give the memorization
objective a clean, self-contained home.

**FSF A/B:** set `full_state_feedback` / `fsf_seed` / `fsf_scaling` on the
default `ESNConfig` in `Config.h`. Log in `main.cpp`: `FSF: ON|OFF …`.

## The corpus (not bundled)

The default corpus is **Tiny Shakespeare** — a single ~1.1 MB plain-text file
(`L = 1,115,394` chars) holding a concatenation of Shakespeare's works,
popularized by Andrej Karpathy's `char-rnn`. It is **not checked into this
repo**; you must supply it yourself and point `corpus_path` (`Config.h`) at it.

- **Get it:** download
  <https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>
  and save it to the `corpus_path` location (default `tinyshakespeare.txt`).
  Paths are resolved relative to the **process working directory** (or use an
  absolute path). If you launch from `cmake-build-release/`, put the file there
  or set `corpus_path` to an absolute path — not relative to
  `examples/StreamingText/`.
- **Substitute freely:** any plain-text file whose bytes all fall in the
  **fixed 96-token alphabet** (newline + printable ASCII `0x20–0x7E`). Corpus
  length is discovered at load time; the alphabet is not — any other byte
  (tab, smart quotes, multi-byte UTF-8, etc.) is a hard load error. The
  reported BPC/top-1 numbers below are specific to Tiny Shakespeare.

## Operating model

### Corpus as a ring buffer
The entire corpus is a ring of length `L = text.size()`. The stream advances
`pos = (pos + 1) % L` and never resets. Each full traversal is a **lap** (the
analogue of an epoch); reservoir state flows continuously across laps. There is
exactly one artificial adjacency per lap, at the wrap seam (`text[L-1] →
text[0]`); over a long run this is negligible and is left unhandled.

`total_steps` is the single stream budget — laps ≈ `total_steps / L`. There is
no per-pass reset and no per-pass LR reset; the cosine schedule runs once across
the whole budget.

### Prequential metric (predict-before-update)
The metric is computed inline on the training stream — no separate pass, no
rewind. At each character:

1. Advance the reservoir one char (feed the true corpus char).
2. Assemble the live **readout input** (all `readout_slices` delay-line ages)
   and **predict** the next char.
3. Score that prediction against the **true** next char `text[(pos+1) % L]` —
   softmax cross-entropy, accumulated into a **rolling-window** BPC and a
   rolling top-1 hit rate over the last `report_window` chars.
4. *Then* fold `(readout input, target)` into the next online training mini-batch.

Because the prediction in steps 2–3 happens **before** the weight update in
step 4 lands (updates are applied per `mini_batch_size`-char batch, and the
prediction uses the pre-update weights), the rolling loss is a fair online
signal. A **rolling-window** average — not a cumulative mean — is reported, so
the number reflects current capability: under memorization the loss is expected
to trend steadily downward across laps. A cumulative mean would be dominated by
early high-loss chars and would mask progress.

### Teacher-forced sampling (no reset, no restore)
At the **end of every lap** (every `L` train steps — one full ring over the
corpus), the loop emits the next `sample_len` chars as a **teacher-forced**
readout starting at the lap boundary (`sample_len = 0` disables samples):

- Normal streaming continues — the reservoir is *not* reset and *not* re-primed.
- At each step the model's predicted-next char (argmax of the logits) is shown
  beside the actual next char.

This shows how well the model has memorized the text at the wrap (and following
chars). There is no autoregressive feedback of the model's own output, so the
sample leaves no synthetic echo in the reservoir. The sample reuses the same
logits already computed for the prequential metric — no extra forward pass.

## Pipeline

1. **Load + validate corpus** — fixed 96-token vocab (newline + printable ASCII
   `0x20–0x7E`); any out-of-vocab byte is a hard error.
2. **Warmup** (`warmup_chars`): reservoir-only drive (`ReservoirWarmup`) — no
   readout forward and no weight updates. Washes out zero-init and fills the
   delay line before training.
3. **Ring loop** (`total_steps`): the per-char body above (advance → predict →
   score → teacher-forced display → accumulate → mini-batch update), wrapping
   `pos = (pos + 1) % L`.

No states buffer — RAM is constant (reservoir state + CNN weights + optimizer).
No model file is written: this is a streaming, exploration-only example.

## Configuration (`Config.h`)

A single `Cfg` struct (no mode enum):

| Knob | Meaning |
|------|---------|
| `corpus_path` | plain-text corpus, not bundled — see [The corpus](#the-corpus-not-bundled) (default `tinyshakespeare.txt` or an absolute path; relative paths use process CWD) |
| `warmup_chars` | reservoir-only washout before the train loop (no readout) |
| `total_steps` | train-stream budget; laps ≈ `total_steps / L` |
| `esn` | `ESNConfig` — reservoir + readout CNN + `readout_slices` (power-of-two with aux; `<= history_depth`). Loop uses `CopyReadoutInput` / `ReadoutInputWidth()` (multi-slice OK). **Do not rely on** `readout.lr_min_frac` or `readout.momentum` here — streaming LR is `Cfg::lr_min_frac` + `CosineLR`; Adam ignores momentum. |
| `mini_batch_size` | grad-accum chunk for `TrainStepBatch` |
| `lr_min_frac` | cosine floor as a fraction of `readout.lr_max` (the **active** floor) |
| `report_window` | rolling-BPC / top-1 window length (chars) |
| `report_every` | live-line print cadence (0 = end only) |
| `sample_len` | chars of teacher-forced text at **end of each lap** (0 = off) |
| `verbose` | gates periodic `roll_bpc` lines only (startup banner always prints) |

Compile-time constants (in `Config.h` / `CharEmbedding.h`): `kDIM` (edit freely;
banner reports live ESN dim/N — stock survey below used **11**), `kCharEmbedDim`
(64-d random per-char embedding; one char per step), and `kCharEmbedSeedXor`.
Readout activation is set via `esn.readout.activation` (`ReadoutActivation::TANH`).

`readout_slices` may be > 1. The stream loop always accumulates the full
assembled readout input, not only the newest slice.

**Lab vs stock:** `Config.h` is often edited for experiments (DIM, FSF, slices,
paths). The [Results](#results) table is a fixed historical survey, not a mirror
of whatever is currently in `Config.h`.

The Tiny Shakespeare file is **gitignored** (`tinyshakespeare.txt`); supply it
locally — see [The corpus](#the-corpus-not-bundled).

## Design choices (memorization, not prediction)

| Aspect | StreamingText |
|---|---|
| Objective | fit one corpus (no generalization claim) |
| Data split | none — the full corpus is a single ring |
| Metric | prequential rolling BPC on the stream (no separate eval pass) |
| Eval intrusion | none — inline, no reset, no snapshot/restore |
| Samples | teacher-forced (predicted-vs-actual, no autoregressive feedback) |
| Persistence | none — streaming, exploration-only |
| Binary | a single train binary (no Train / Eval / Infer modes) |

StreamingText carries its own self-contained `Corpus` and `CharEmbedding`
(`namespace streaming_text`) rather than sharing a library, so it has no
cross-example coupling.

## Why this is hard: the input is the bottleneck *(hypothesis)*

> **Hypothesis — not a proven result.** The account below is our current
> working explanation. The lever effects (what helps, what hurts) are
> measured; the causal story tying them to input discontinuity — "the
> discontinuity is the *signal*" — is interpretation, and we have not yet
> run the experiments that would confirm it.

StreamingText is harder than the smooth-input examples (BasicPrediction,
SignalClassification, StreamingAnomaly) because its random-embedding input is
*discontinuous*: each char maps to a fixed random vector, so consecutive chars
are uncorrelated and the drive teleports every step. The bottleneck is the
input, not the reservoir — but the discontinuity turns out to be the **signal**,
not noise (it carries per-char identity). The levers that help all imprint each
char *harder* rather than smoothing it: high `input_scaling`, `leak = 1.0`,
supercritical `spectral_radius` (peak ~1.3; >1.3 degrades), and larger `DIM`
(which in turn wants *lower* per-channel drive). Input-side smoothing — an EMA
low-pass on the drive — was tried and **rejected** (it never beat raw drive),
and `leak < 1` only helped at weak drive.

## Build & run

Built as the `StreamingText` target (see root `CMakeLists.txt`). Build Release
with the CLion-bundled toolchain (see [Building and Running](../../README.md#building-and-running-c));
run `StreamingText.exe` with no arguments — all configuration is in `Config.h`.

**Corpus path:** `corpus_path` is resolved from the process working directory.
From a shell, either `cd` to the directory that holds `tinyshakespeare.txt` before
launching, or set an absolute path in `Config.h`. CLion run configs often use
`cmake-build-release` as CWD — place the file there or override the path.

Expect: a compact startup banner (seeds, reservoir knobs, `ReadoutArchSummary`,
stream budget / LR), then live rolling-BPC lines and optional teacher-forced
samples once per completed lap. Reservoir ctor verbose is off so you do not
get a second architecture dump from the library.

## Results

### Long run — DIM 11, sr 1.3, input_scaling 2.5 (historical best-of-survey)

> Snapshot of a surveyed peak config (not necessarily current `Config.h`).
> Single-slice readout, no FSF, `history_depth = 1`.

A single continuous stream of the surveyed peak config, taken far past the
lap-0 screen to watch the memorization trend settle.

| Knob | Value |
|---|---|
| DIM / N | 11 / 2048 |
| spectral_radius | 1.3 |
| input_scaling | 2.5 |
| leak_rate | 1.0 |
| history_depth | 1 |
| char embed | 64-d random, one char/step, vocab 96 |
| CNN readout | 1 layer, 16 ch, lr 1e-3→2e-4 cosine, wd 1e-5 |
| reservoir_seed | 7397376 |
| corpus | tinyshakespeare (L = 1,115,394) |
| budget | 45M steps planned (~40 laps); log ends at **39M / lap 34** (run truncated, no final summary line) |

Per-lap rolling-BPC stats (100k-step report points, `report_window` rolling
average). BPC trends steadily downward and top-1 upward (with lap-to-lap
jitter); both flatten out around laps 29–34.

| Lap | BPC mean | BPC min | top-1 mean | top-1 max |
|----:|---------:|--------:|-----------:|----------:|
| 0  | 2.529 | 2.364 | 0.488 | 0.524 |
| 1  | 2.377 | 2.262 | 0.515 | 0.541 |
| 2  | 2.277 | 2.112 | 0.527 | 0.565 |
| 3  | 2.197 | 1.984 | 0.542 | 0.591 |
| 4  | 2.147 | 1.982 | 0.550 | 0.591 |
| 5  | 2.156 | 2.029 | 0.543 | 0.565 |
| 6  | 2.107 | 1.932 | 0.556 | 0.593 |
| 7  | 2.109 | 1.961 | 0.553 | 0.591 |
| 8  | 2.074 | 1.968 | 0.560 | 0.582 |
| 9  | 2.059 | 1.864 | 0.561 | 0.603 |
| 10 | 2.022 | 1.850 | 0.568 | 0.606 |
| 11 | 2.021 | 1.837 | 0.567 | 0.604 |
| 12 | 2.001 | 1.778 | 0.569 | 0.612 |
| 13 | 1.984 | 1.794 | 0.572 | 0.615 |
| 14 | 1.994 | 1.937 | 0.573 | 0.588 |
| 15 | 1.978 | 1.817 | 0.574 | 0.610 |
| 16 | 1.943 | 1.711 | 0.580 | 0.635 |
| 17 | 1.921 | 1.753 | 0.586 | 0.626 |
| 18 | 1.949 | 1.818 | 0.576 | 0.600 |
| 19 | 1.919 | 1.730 | 0.585 | 0.625 |
| 20 | 1.930 | 1.781 | 0.581 | 0.618 |
| 21 | 1.907 | 1.782 | 0.588 | 0.612 |
| 22 | 1.903 | 1.723 | 0.586 | 0.626 |
| 23 | 1.873 | 1.716 | 0.594 | 0.628 |
| 24 | 1.881 | 1.705 | 0.590 | 0.625 |
| 25 | 1.869 | 1.669 | 0.591 | 0.632 |
| 26 | 1.856 | 1.665 | 0.594 | 0.636 |
| 27 | 1.871 | 1.789 | 0.592 | 0.614 |
| 28 | 1.865 | 1.720 | 0.593 | 0.624 |
| 29 | 1.832 | 1.617 | 0.599 | 0.649 |
| 30 | 1.816 | 1.641 | 0.605 | 0.649 |
| 31 | 1.845 | 1.713 | 0.593 | 0.618 |
| 32 | 1.827 | 1.635 | 0.601 | 0.642 |
| 33 | 1.844 | 1.697 | 0.595 | 0.630 |
| 34 | 1.827 | 1.704 | 0.601 | 0.624 |

**Takeaways.**

- **Headroom realized.** Rolling BPC falls from ~2.53 (lap 0) to ~1.82 by lap
  30 — ~0.71 bits gained over the run.
- **Best windows.** The single best 100k-window hit **1.617 BPC** (lap 29), and
  top-1 peaked at **0.649**.
- **Near the ceiling.** The curve is clearly decelerating: laps 29–34 sit in a
  ~1.82–1.85 band with no further gain, so this config is near its memorization
  ceiling well before the 45M-step budget runs out.

**Expected with more capacity.** This ceiling is a DIM-11 (N = 2048) ceiling,
not a fundamental one. Higher-order configurations (larger DIM, hence more
reservoir vertices) are expected to push the BPC mean further down — plausibly
toward ~1.5 — since memorization capacity scales with reservoir size. Those
runs were **not performed here due to hardware limitations**; this DIM-11 long
run is the largest the available hardware could carry to convergence.
