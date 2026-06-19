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
**Status:** RESOLVED (§4.2 / §7.3). Simplified per user: gate on the **ensemble
(consensus) output** error, not per-member errors — one scalar stream, not M. The class
keeps one running estimate `consensus_err_` (EMA/window of per-step consensus-vs-target
error); `AdvanceKappa(c_out, target)` folds each step's error in and opens the ramp when
it crosses the threshold. Signal fixed; smoothing + threshold stay open (§11.2).

§4.2/§11.2 correctly mark *which* signal and *what* threshold as open. But the
**measurement** is unspecified: online training error must be accumulated somewhere
(running mean? window?) and exposed to the gate. `ProbeLoss` is single-ESN. If the
class owns the ramp (G1), it must own this accumulator.

**Recommendation:** specify that the orchestrator maintains a per-member running
training-error estimate that the gate reads, even if the exact statistic stays open.

---

## Should resolve — underspecified, would cause guesswork

### G4. Online learning-rate / weight-decay ownership
**Status:** RESOLVED (§4.2 / §7.3). Shared `lr` / `weight_decay`, owned by `EnsembleESN`
ctor config, passed verbatim into every member's `TrainLiveStepRegression` (members share
the base config, so no per-member differ). Binding constraint: **held constant (or floored)
through the κ ramp** — the ramp must stay slow *relative to* readout adaptation, which fails
if lr decays to zero mid-ramp. Annealing is optional and only after κ holds (§11.3). This
is the ESN online lr (per-step), not `ReadoutConfig`'s batch cosine fields.

  Correction note: an earlier framing ("train indefinitely → fixed lr is natural") was
  backwards — the ensemble trains finite-then-freezes, which would argue for annealing; the
  *real* constraint is the ramp interaction, not the horizon.

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

### G12. Proper external-drive feedback mode (dedicated port) — design decided
**Status:** RESOLVED in the doc (§7.2 rewritten to the chosen architecture); impl pending

**Reframed (2026-06-19).** The user flagged that the existing feedback machinery was never
exercised and may be wrong, and chose to reshape Reservoir/ESN properly rather than patch
around them ([[feedback-machinery-untested]]). Architecture selected: a **dedicated
feedback port** (not "feedback = input channels", not a new dense projection).

**All feedback is external — no ESN internal feedback policy** (user, 2026-06-19). The
earlier 3-mode enum is dropped: there is no `InternalPolicy`, no quarantine. The internal-F
apparatus is **deleted outright**.

Target design (§7.2):
- **Reservoir** — reuse the sound substrate (block-partition `InjectFeedback`, own weight
  block, `feedback_scaling`, SR-exclusion at `:139`). Touch-ups: optional vector-form
  inject; relax the `D|N` throw (`:50`) to any `D ≤ N`; **verify the path end-to-end**
  (it was never exercised).
- **ESN** — **delete** the internal learned-F apparatus (the `if` block at `ESN.cpp:80-97`,
  the `StepLive` F branch at `:108-121`, `InjectFeedbackClamped`, `ProbeLoss` /
  `TrainFeedbackCycle`, telemetry buffers, `Get/SetFeedback*`). Redefine
  `num_feedback_channels` as the external-channel count (`0` = none, `D > 0` = D external
  channels); drop the `== 1` guard (allow any `D ≤ N`). Trim `StepLive` to input-only; add
  `StepLiveExternalFeedback(inputs, φ)` as the only feedback entry point. (Readout init is
  no longer a concern — see G14: readouts are born ready, so `ESN::InitOnline` is warm-up
  only and the ensemble doesn't call it.)
- **Readout / HypercubeCNN** — untouched by the *feedback* redesign (the G14 eager-build is
  a separate cleanup).

The ESN gets *simpler* (one fewer feature). Supersedes the earlier boolean `external_drive`
flag and the 3-mode enum alike.

**Recommendation:** carry into the implementation phase; sequence the substrate
verification first.

### G14. Readout eager-build — kill the lazy-init dance — DONE
**Status:** IMPLEMENTED + verified (commit `2e8c26a`).

Root cause: the readout CNN was built lazily (deferred to `Train`/`InitOnline`) only
because warm-up was historically treated as a readout-disengaged phase. But
`build_architecture()` needs only config — no data, no warm-up. Fix: build eagerly in the
`Readout` ctor (`build_architecture` + ADAM + `PrepareBuffers` + sizes); `net_` is now a
non-null invariant. **Deleted** `Readout::InitOnline()`; the triplicated
build+optimizer+buffers sequence collapses to one; `rebuild_from_blob` just `SetWeights`;
`Train()` trains in place (no re-randomize — build-once-train-once identical). ESN: dropped
the dead `readout_.InitOnline()` (`:168`) and `feedback_readout_->InitOnline()` (`:92`);
`ESN::InitOnline` is warm-up only.

Verified: all 23 targets compile; BasicPrediction batch R2=1.0; Python smoke R2=1.0 with
eager ctor build across DIM 5-12. Superseded the no-arg `ESN::InitOnline()` plan
(unnecessary — members are born ready).
