# Ensemble ESN — Design Gap Analysis

> Tracking doc for gaps found in [ensemble_esn_feedback.md](ensemble_esn_feedback.md).
> The mechanism itself is sound and internally consistent; nearly every gap is about
> the **boundary of the `EnsembleESN` class** and the **lifecycle around the
> steady-state `Step()`**, not about the coupling mechanism.
>
> Worked one at a time. Each entry: what's missing → recommendation → status.

---

## Blocking — must resolve before coding

### G1. Class boundary: does `EnsembleESN` own the *policy* or just the *mechanism*?
**Status:** open

The single biggest ambiguity, and the doc contradicts itself:
- §4.2 describes the competence gate + ramp as part of *the design* (reads like the
  class owns the schedule).
- §7.2's sketch says `// caller advances kappa_ per the competence-gated ramp` — i.e.
  the **caller** owns the schedule and `EnsembleESN` is a dumb lockstep stepper.

These imply very different APIs: either `Step()` advances κ internally (class owns gate
+ ramp + the error signal feeding the gate), or `Step()` takes κ as an argument and the
consumer writes the ramp loop. Everything in §4 and §11.2–3 hinges on which.

**Recommendation:** class owns the ramp. The gate needs per-member training error,
which lives inside the ensemble anyway (see G3). `kappa_target`, gate signal/threshold,
and ramp shape become constructor config. Resolve G1/G2/G3 as one coherent edit.

### G2. Online lifecycle / state machine is missing — only the steady-state step is specified
**Status:** RESOLVED (§7.1). Reshaped during the work: with readouts built up front
(`InitOnline(input, 0)`), warm-up is *not* a separate mode — the lifecycle collapses to
**one run loop + a two-knob schedule** (train-enable at step W, κ-ramp at competence).
Documented in §7.1 with the unified-seam rationale, the `ESN.cpp:162` finding (readout
build is warm-up-independent), reset/sequence-boundary semantics, and the timeline
diagram. Surfaced G11 + G12 (below).

`ESN.h` confirms the online path is `InitOnline(warmup_inputs, warmup_count)` → builds
the readout CNN → *then* `TrainLive*`. The doc's `Step()` assumes readouts already
exist. Missing phases:

```
construct ──► per-member InitOnline(warmup) ──► coupled online train ──► coupled inference
              └─ NECESSARILY UNCOUPLED ─┘        └─ gate → ramp → hold κ* ─┘
              (no readouts yet, so no             (Step() as written)
               consensus to inject)
```

The **uncoupled-warmup necessity** is a real subtlety: coupling cannot begin until
after `InitOnline` because there are no member outputs (no consensus) during warmup.
Also unaddressed: **sequence-boundary reset** — across multiple sequences, do all
members `ResetReservoirOnly()` together?

**Recommendation:** add a §7.0 "lifecycle" with the four phases, the
warmup-is-uncoupled note, and reset semantics.

### G3. Competence-signal plumbing is unspecified (beyond the open *choice*)
**Status:** open

§4.2/§11.2 correctly mark *which* signal and *what* threshold as open. But the
**measurement** is unspecified: online training error must be accumulated somewhere
(running mean? window?) and exposed to the gate. `ProbeLoss` is single-ESN. If the
class owns the ramp (G1), it must own this accumulator.

**Recommendation:** specify that the orchestrator maintains a per-member running
training-error estimate that the gate reads, even if the exact statistic stays open.

---

## Should resolve — underspecified, would cause guesswork

### G4. Online learning-rate / weight-decay ownership
**Status:** open

`TrainLiveStepRegression(target, lr, weight_decay)` needs `lr`/`wd` every step; §7.2
elides them as `/*lr,wd*/…`. Fixed, scheduled, shared across members? Untouched.

**Recommendation:** fixed `lr`/`wd` from config, shared across members; note as such.

### G5. Seed-derivation scheme
**Status:** open

§5 says "M distinct seed pairs (e.g. derived from one ensemble seed)" with no concrete
rule. Reproducibility needs a deterministic mapping
(e.g. `seed_i = hash(ensemble_seed, i)`, `bias_seed_i = hash(ensemble_seed, i, BIAS)`).

**Recommendation:** one sentence fixing the derivation rule.

### G6. Diagnostic accessors
**Status:** open

The §8 sweeps need to read individual member outputs, current κ, gate state, and
per-member error — none are in the §7.2 surface (only `c_out`). The experiments can't
run without them.

**Recommendation:** list the read-only accessors the design commits to.

---

## Polish — structural / minor

### G7. Section §10 is missing entirely
**Status:** open

Headers run …§9, then §11. A §10 "Related work" existed in earlier notes; it's gone and
§11 wasn't renumbered.

**Recommendation:** restore §10 or renumber Open Questions to §10.

### G8. `|κ|` notation inconsistent with the κ > 0 convention
**Status:** open

§8 and §11.4 say "monotone in `|κ|`" / "useful range of κ", but §4.1 fixes κ > 0 by
convention — so `|κ| = κ`.

**Recommendation:** drop the absolute-value bars.

### G9. Parallelism note (optional)
**Status:** open

M independent `Step`s per tick is embarrassingly parallel and the project already links
OpenMP. Not a design driver.

**Recommendation:** one line ("members step independently; parallelizable").

### G10. Confirm D = `NumOutputs()` = `num_feedback_channels`
**Status:** open

The design ties all three together implicitly. Stating the identity once removes a
footgun.

**Recommendation:** state the identity explicitly where D is introduced.

---

## Found during the work (lifecycle / integration audit)

### G11. `num_feedback_channels = D` divisibility — NON-ISSUE
**Status:** RESOLVED (non-issue, verified against the sources). The `Reservoir.cpp:50`
divisibility check is conservative, not load-bearing. `InjectFeedback` (`:269-277`) writes
channel c to `[c·block, (c+1)·block)`, `block = floor(N/D)`; non-dividing D leaves the
`N mod D` (`≤ D−1`) tail vertices at reset-zero. Those act only as zero feedback
**sources** — they still *receive* drive via the neighbor gather at `:244` (no cut-off),
and no index goes out of range (`block ≥ 1` for D ≤ N). Removing the check is safe for any
**D ≤ N**; dropped fraction `(N mod D)/N` is negligible for the realistic small-D /
large-N regime and only material as D → N. No power-of-two restriction. §7.2 updated.

### G12. A feedback mode that drops the internal-F policy — required for D > 1
**Status:** RESOLVED in the doc (§7.2 rewritten, re-verified); implementation pending

The current ESN couples two independent things under `num_feedback_channels > 0`:
(i) the reservoir feedback **substrate** (`vtx_feedback_` + `n_·dim_` weight block), built
automatically by `Reservoir::Create` at `ESN.cpp:66`, already D-channel-ready; and
(ii) the ESN internal **learned-F apparatus** (`if` block at `ESN.cpp:80-97`: F readout +
telemetry buffers) gated by a `num_feedback_channels == 1` throw (message: *"feedback
training (v1) supports exactly 1 feedback channel"*). The ensemble wants (i) at D channels
and **none** of (ii).

A `FeedbackConfig::external_drive` flag makes the ctor **skip the entire internal-F `if`
block** (no F, no telemetry, no guard). It **allocates nothing** — the reservoir already
built the substrate; it only *decouples* "has feedback channels" from "has an internal F
policy." **Required for D > 1** (ctor throws today); a **cleanup for D = 1** (ctor succeeds
but builds unused F). Since the capability targets general D ≤ N, treat as required.

Correction log: an earlier draft said the flag "allocates the feedback weight block" —
wrong; that is the reservoir's job and already done. The flag only skips internal-F.

§7.2 lists **two** required `ESN` changes (seam + `external_drive`), **zero**
Reservoir/Readout changes. No HypercubeCNN rewrite warranted.

**Recommendation:** carry both into the implementation phase; HCNN and Reservoir untouched.
