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
**Status:** RESOLVED (non-issue). The `Reservoir.cpp:50` divisibility check is
conservative: when D does not divide N, `InjectFeedback` just leaves the `≤ D−1`
remainder tail vertices unfed (`block = N/D` truncates) — a bounded, benign coverage gap,
not a correctness problem. The check can be relaxed/removed, so **D is unconstrained**
(any D ≤ N). No power-of-two restriction. §7.2 updated accordingly.

### G12. `external_drive` feedback mode is a *required* core change, not optional
**Status:** RESOLVED in the doc (§7.2 rewritten); implementation still pending

The original §7.1 called `external_drive` an optional cleanup and claimed "required
footprint is the one seam." The sources contradict this: `ESN.cpp:82` **throws unless
`num_feedback_channels == 1`** (the internal scalar-F path), and `ESN.cpp:87` eagerly
builds a scalar `feedback_readout_` that cannot drive D channels. So D-channel external
feedback is impossible without an `external_drive` mode that relaxes the guard and skips
building F. §7.2 now lists **two** required `ESN` changes (seam + `external_drive`) and
**zero** Reservoir/Readout changes. Confirmed: no HypercubeCNN rewrite is warranted.

**Recommendation:** carry both changes into the implementation task list; HCNN and
Reservoir stay untouched.
