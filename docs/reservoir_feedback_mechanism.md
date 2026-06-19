# Feedback Mechanisms — Exploration Notes

> Status: **exploratory**. This is a design discussion, not an approved build or
> a description of existing behavior. The current codebase is strictly
> **open-loop** (`ESN::Warmup`/`ESN::Run` → `Reservoir::Step` → `Readout`, with
> no path from the readout back into the reservoir).

## The idea: a `vtx_feedback` driver path

Add a second per-vertex driver buffer alongside `vtx_input_`, **mechanistically
identical** to it — its own weight block, summed into each neuron in
`Reservoir::UpdateState` via the dim-neighbor XOR gather, injected each step —
but populated from one of two sources:

- **(a) an internal reservoir-derived metric** (e.g. mean activation, energy), or
- **(b) a metric computed by the readout** (the model's own output).

## What it is, in reservoir-computing terms

- **Readout-fed** feedback is the classic ESN **output-feedback path** — Jaeger's
  `W_fb`. It is well established and powerful: it is what lets an ESN run
  *autonomously / generatively* (free-running a learned attractor, multi-step-ahead
  generation, pattern generators) rather than being purely input-driven. Adding it
  is the feature that takes this project from open-loop to closed-loop. It is also
  notoriously **stability-sensitive** — output-feedback ESNs typically need careful
  `W_fb` scaling and noise injection (plus teacher forcing) during training.
- **Internal-metric-fed** feedback is a different animal: closer to **homeostatic /
  self-gating** feedback. Less standard, more experimental (e.g. drive the reservoir
  from its own mean activation to self-regulate).

Both are mechanistically "another `vtx_input_`," which is why mirroring the input
path is the natural implementation.

## Mechanical implementation (mirrors `vtx_input_`)

- New `vtx_feedback_` buffer (`n_` floats).
- New feedback weight block appended to `vtx_weight_` (another `n_ * dim_`),
  bumping `num_weights_`.
- A third accumulation loop in `UpdateState`:
  `s += vtx_feedback_[v ^ NearestMask(i)] * fw[i];`
- An `InjectFeedback(channel, value)` analogous to `InjectInput`; decide whether
  `Step` clears it each tick (as it does for `vtx_input_`).
- A new `feedback_scaling` config knob with its own fan-in normalization, by
  analogy to the input path's `1/sqrt(dim)`.

## The crux: timing / causality

`vtx_input_` is set **before** `Step` and consumed **during** it. But a
readout-derived metric `y(t)` is computed **from** the post-`Step` state — it does
not exist yet when `Step` needs it. So readout feedback at step *t* must use
**y(t-1)**: an inherent one-step delay loop.

The clean control flow keeps the `Reservoir` ignorant of the `Readout` (preserving
the existing decoupling) and lets `ESN` own the loop, exactly as it already owns
`InjectInput`:

```
Step → Outputs() → readout computes y → InjectFeedback(y) → next Step
```

Internal-metric feedback *could* be computed inside the reservoir, but injecting it
externally preserves symmetry with the input path and avoids coupling the reservoir
to its driver.