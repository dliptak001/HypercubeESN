# Reservoir Feedback Substrate — the external-drive port

> Status: **landed**. This describes existing, verified behavior, not a proposal.
> The reservoir carries a second per-vertex driver path — `vtx_feedback_` — that is
> a structural twin of the input path. It is **off by default**
> (`num_feedback_channels = 0`): with zero channels no weights are allocated and the
> open-loop realization is unchanged. With `D > 0` channels the reservoir gains a
> closed-loop hook the caller drives each step.

This is the single-reservoir foundation the [ensemble design](ensemble_esn_feedback.md)
builds its consensus coupling on. This document specifies the **port** (the reservoir
mechanism); the *policy* — what to inject, when, and the loop that produces it — lives
in the caller and is documented there.

## 1. Source-agnostic by design — external drive only

The reservoir exposes a feedback **port**: a buffer the caller stages each step, summed
into the neurons by the reservoir, then cleared. The reservoir does **not** know or care
where the injected values come from. It computes nothing about its own output and routes
nothing back internally.

That is a binding decision, not an omission (ensemble design §7.2): **all feedback is
external; the ESN has no internal feedback policy.** An earlier exploration imagined two
*sources* for the drive —

- **(a)** an internal reservoir-derived metric (mean activation, energy, …), or
- **(b)** the model's own readout output (classic output-feedback, Jaeger's `W_fb`),

— with (a) computed *inside* the reservoir. That internal-metric path was never built and
is explicitly excluded. From the port's view the distinction collapses: **both are just
values the caller injects.** The caller may compute the drive however it likes — a readout
output for a generative single-reservoir loop, an ensemble consensus deviation for coupled
members, a homeostatic metric — and the port treats them identically. There is no
internal-feedback mode anywhere in this codebase.

Keeping the source out of the reservoir preserves the existing `Reservoir`↔`Readout`
decoupling: the reservoir stays ignorant of the readout, and the caller owns the loop —
exactly as it already owns `InjectInput`.

## 2. Mechanism (as built) — a twin of the input port

The feedback path mirrors `vtx_input_` member-for-member. Everything below is realized in
`Reservoir::Initialize`, `Reservoir::UpdateState`, and `Reservoir::Step`.

```
 weight layout in vtx_weight_:   [ input block | feedback block | recurrent block ]
                                   n·dim          n·dim            n·dim·history_depth
                                   └ in_scaling   └ feedback_      └ depth-tapered, then
                                     /√dim          scaling/√dim     rescaled to target SR
                                                                     (input + feedback
                                                                      EXCLUDED from rescale)
```

- **Buffer.** `vtx_feedback_` holds `n_` floats, allocated **only** when
  `num_feedback_channels > 0`. Zero channels ⇒ no buffer, no weights, no work.
- **Weight block.** A dedicated `n_·dim_` block sits between the input and recurrent
  blocks in `vtx_weight_`, drawn once at construction from the **Feedback-labelled
  substream** of the reservoir's single `seed` (the SplitMix64 fan-out — recurrent /
  input / feedback / bias / SR-probe streams are statistically independent off one seed).
- **Fan-in normalization.** Its own `feedback_scaling` knob, carrying the same
  `1/√dim` fan-in factor as `input_scaling`, so a given `feedback_scaling` delivers
  **DIM-invariant** drive.
- **Accumulation.** Summed into each neuron in `UpdateState` via the same
  Hamming-distance-1 (single-bit-flip) XOR gather as the input term:

  ```
  s += Σ_{i<dim}  vtx_feedback_[v ^ NearestMask(i)] · fw[i]
  ```

  i.e. vertex `v` gathers its `dim` neighbors' feedback values, each by its own weight —
  identical in form to the input fan-in, with `fw` pointing at the feedback block.
- **Outside the spectral-radius rescale.** The SR secant solve rescales **only** the
  recurrent block; the feedback block (like input and bias) is excluded. So feedback
  does **not** enter the open-loop stability budget and does **not** bound closed-loop
  stability — that is the caller's responsibility through the drive it injects.
- **Per-step lifecycle.** Staged **before** `Step`, consumed **during** it (in
  `UpdateState`), and **cleared** (memset to 0) at the end of every `Step` — exactly like
  input. So a drive must be re-injected each step it is wanted.
- **Not dynamical state.** `Reset` zeros it; `TakeSnapshot` does **not** capture it (a
  staged drive, not persistent state), and `RestoreSnapshot` clears it — so a
  restore-and-replay reproduces the trajectory bit-for-bit from the snapshot plus the
  subsequent injections alone.

## 3. Injection API

```cpp
// per-channel: broadcast one value to a contiguous vertex block (twin of InjectInput)
void Reservoir::InjectFeedback(size_t channel, float value);

// vector form: stage all D channels at once — the D-channel external-drive entry point
void Reservoir::InjectFeedback(const float* feedback, size_t count);   // count must == D
```

Channel `c` drives the contiguous vertex block `[c·floor(N/D), (c+1)·floor(N/D))`,
broadcasting its value to every vertex in the block — the block-partition layout the
input port uses. The vector form throws unless `count == num_feedback_channels` and loops
the per-channel call.

**D ≤ N — no divisibility requirement.** Any `D` in `[1, N]` is admissible; `D` need
**not** divide `N = 2^dim`. When `D ∤ N`, the `N mod D` (`≤ D−1`) tail vertices are never
written by any channel — they hold reset-zero and act as **benign zero feedback
*sources*** while still *receiving* drive through the neighbor gather (no index leaves
`[0, N)`, and `block = floor(N/D) ≥ 1` for any `D ≤ N`). No power-of-two restriction. The
only guard is `D ≤ N`. (Contrast the input port, which still requires `num_inputs | N`,
since it has no benign-tail story.)

## 4. Timing / causality — port contract vs. caller policy

The **port contract** is just: inject → `Step` → cleared. *What* to inject and *when* is
**caller policy**, and the two natural loops differ only in where the drive comes from:

- **Generative single-reservoir loop.** A drive derived from the readout reads the
  *post-`Step`* state, which does not exist when `Step` needs it — so such a loop injects
  the **previous** step's output: an inherent one-step delay.

  ```
  Step → Outputs() → caller computes drive → InjectFeedback(drive) → next Step
  ```

- **Ensemble consensus loop.** The ensemble reads each member's output at its **current**
  state `x(t)`, forms the consensus, and injects the deviation *before* stepping to
  `x(t+1)` — single-step causality, no separate delay buffer (those outputs already
  exist). See [ensemble design §3](ensemble_esn_feedback.md).

Either way the reservoir stays ignorant of the readout; the caller owns the loop.

## 5. Where this sits in the stack

```
  Reservoir              vtx_feedback_ port  ── this document (the substrate)
     ▲  InjectFeedback / Step / clear
     │
  ESN          StepLiveExternalFeedback(inputs, φ)  ── the ONLY feedback entry point;
     ▲                                                 StepLive(inputs) is input-only
     │
  EnsembleESN  φ_i = κ·Δ_i across M members          ── the policy: consensus coupling
                                                        (ensemble_esn_feedback.md)
```

`ESN::StepLiveExternalFeedback(inputs, φ)` stages `φ` (D floats) on the feedback channels
and the task inputs on the input channels, then `Step`s — the only way feedback enters at
the ESN layer; `ESN::StepLive(inputs)` is the input-only path. `EnsembleESN` drives its M
members through that seam with `φ_i = κ·Δ_i` (each member's scaled deviation from the
consensus). The substrate specified here is the foundation both sit on.
