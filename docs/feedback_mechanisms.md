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

## Risks the current design does not cover

- **Stability.** The echo-state property is an *input-driven* guarantee; closing a
  feedback loop voids it. Expect to need feedback-scaling control plus noise
  regularization during training.
- **Spectral radius.** `Reservoir::EstimateSpectralRadius` only measures the
  recurrent block. Feedback weights sit **outside** that estimate, so realized
  dynamics can diverge from the configured spectral-radius target.
- **Single-input collapse.** The single-input fan-in note already documented in
  `UpdateState` applies identically to a broadcast feedback signal (all
  dim-neighbor gathers collapse to one scalar when the signal is broadcast to every
  vertex).

## Recommendation for exploration

Prototype as a **parallel `vtx_feedback_` buffer + weight block + `feedback_scaling`
knob**, driven externally from `ESN`. Resist refactoring the input path into a
generalized "array of drivers" up front — prove the dynamics earn it first, then
generalize. With feedback weights zeroed it is a numerical no-op, so it can land
without disturbing the open-loop examples.

## Verification (if it graduates to implementation)

- **Smoke:** Release-build all targets; confirm open-loop examples are bit-for-bit
  unchanged with feedback weights zeroed (numerical no-op).
- **Closed-loop sanity:** a generative task (e.g. autonomous continuation of a sine
  via a `BasicPrediction`-style loop feeding `PredictRaw` back through
  `InjectFeedback`) should free-run without immediate divergence at small
  `feedback_scaling`.

## Related work / prior art

A literature scan was done specifically for the **internal-metric-fed** variant
(drive the reservoir from an aggregate scalar of its own activity, e.g. mean
activation / state norm / energy). The readout-fed variant is uncontroversially
classic ESN output feedback (Jaeger's `W_fb`) and is not surveyed further here.

**Finding.** The *general principle* — self-regulating a reservoir from a measure
of its own activity — is well established under several names. But the *literal*
form proposed here — a **single aggregate scalar**, kept **fixed / non-task-trained**
as a homeostatic measure, **injected back as an input-like drive** (through a
`W_in`-style path) — appears genuinely **under-explored**. Each axis is individually
covered in the literature; the conjunction is not (to the best of a moderate-effort
search; absence is hard to prove).

Closest prior art, ranked by proximity:

1. **Ehlers, Nurdin & Soh (2025), "Improving the performance of echo state networks
   through state feedback," *Neural Networks* 184:107101** —
   <https://arxiv.org/abs/2312.15141>. **Closest mechanism.** Augments the input as
   `u_k → u_k + Vᵀx_k`, feeding a function of the reservoir state back through the
   input pathway. *But* `Vᵀx_k` is a **trained linear projection of the full
   state** (scalar output; `V` optimized against the task by batch gradient
   descent), not a fixed scalar homeostatic metric — right plumbing
   (state→input), different semantics. The single nearest reference.
2. **Schubert & Gros (2021), "Local homeostatic regulation of the spectral radius of
   echo-state networks," *Front. Comput. Neurosci.* 15:587721** ("flow control") —
   <https://pmc.ncbi.nlm.nih.gov/articles/PMC7958921/>. **Closest intent:**
   homeostatic self-regulation toward the edge of chaos from internal activity, and
   it directly regulates the **spectral radius** (relevant to the SR risk noted
   above). *But* it is a **local per-neuron gain** rule, not a global scalar fed as
   input.
3. **Triesch (2005); Schrauwen, Wardermann, Verstraeten, Steil & Stroobandt (2008),
   "Improving reservoirs using intrinsic plasticity," *Neurocomputing*
   71(7–9):1159–1171.** Canonical "self-regulation from internal activity
   statistics," but **per-neuron transfer-function (gain/bias) adaptation**, not an
   aggregate-as-input.
4. **Self-organized-criticality line:** P-CRITICAL
   (<https://arxiv.org/abs/2009.05593>, tunes branching factor toward edge of chaos)
   and astrocyte-modulated plasticity (<https://arxiv.org/abs/2111.01760>, integrates
   population activity into a *global* feedback signal). Closest to an **aggregate**
   signal, but it drives **plasticity**, not an input channel.
5. **Neuromodulation-inspired:** Mei/Logiaco et al. (*Neural Computation* 2026);
   Vecoven et al. (2020), *PLOS ONE* 15(1):e0227922. Broadcast a single global scalar
   to all units — but **externally supplied / learned**, not a self-measured mean fed
   as input.

**Pattern across the field:** internally-derived activity signals almost always
modulate **parameters / plasticity** (gain, bias, leak, E–I balance, branching
ratio) rather than acting as an **input channel**. The one architecture that uses
the state→input plumbing (Ehlers et al.) makes the feedback a *trained task signal*,
not a fixed homeostatic metric.

**Positioning.** The distinguishing conjunction here is *(scalar aggregate) ×
(fixed homeostatic semantics) × (injected via `W_in`)*. To defend novelty, contrast
explicitly against: Jaeger's `W_fb` (feeds *task output*), intrinsic plasticity
(per-neuron *parameter* adaptation), flow control (local gain toward an SR target),
and Ehlers et al. state feedback (trained multi-dim *state*→input). Confidence:
moderate-high on coverage, moderate on the "genuine gap" claim — a dedicated
arXiv/Scholar pass on terms like *"activity-dependent global bias"* or *"homeostatic
input gating"* could still surface a closer hit.
