# Ensemble ESN — Design Gap Analysis

> Tracking doc for gaps found in [ensemble_esn_feedback.md](ensemble_esn_feedback.md).
> The mechanism itself is sound and internally consistent; nearly every gap is about
> the **boundary of the `EnsembleESN` class** and the **lifecycle around the
> steady-state `Step()`**, not about the coupling mechanism.
>
> Worked one at a time. Each entry: what's missing → recommendation → status.
>
> **NOTE (2026-06-19, post-resolution):** the design doc was later revised — the
> "Falsifiable thesis and experiments" section (then §8) was **removed**, and the
> following sections renumbered (Risks §9 → §8, Open questions §10 → §9). Closed
> entries below are kept verbatim as the historical record, so their section
> references (e.g. the §8 experiments that shaped the G6 accessor list, the §10.x
> open-question cross-refs) reflect the doc **at resolution time**, not its current
> numbering. The resolved outcomes still hold — the §7.4 diagnostic surface remains.
>
> **NOTE (2026-06-23, decision reversed):** the **"class owns the ramp" resolution
> of G1/G2/G3 was later reversed.** The competence-gated κ ramp (EMA consensus-error
> gate → linear ramp to `kappa_target`), the `AdvanceKappa` step, the ramp/gate
> config fields, *and* the class-owned washout were all **removed** as premature —
> κ is now a bare scalar the **caller** manages via `SetKappa` (starts at 0). There is
> no reset entry point either: a fresh sequence is just a caller-driven warm-up (step
> with `target = nullptr`), which washes out any leftover state by the echo state
> property — so no explicit cold-clear is needed.
> `GateOpen()` is gone from the diagnostic surface (now §7.3); `Kappa()`/`CurrentStep()`
> remain. See
> the current [ensemble_esn_feedback.md](ensemble_esn_feedback.md) §4.2/§7.1 for the
> live design. Entries below that recommend or assume the class-owned ramp/gate are
> **historical** — they record the design as it stood, not the shipped code.

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
members `ClearReservoir()` together?

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
**Status:** RESOLVED (§5) — and the resolution simplified the *Reservoir* itself.

The fix went deeper than a doc sentence. `Reservoir` previously spun up four RNG streams
two different ways: three forked off `seed` by ad-hoc additive offsets
(`seed + 0x9E3779B9` for input+feedback, `seed + 12345` for the SR probe) plus a wholly
separate `bias_seed` config field. The offsets are folklore (mt19937 from nearby integers
is not guaranteed independent) and `bias_seed` existed only because there was no clean
primitive for forking a stream.

Replaced (committed) with a single **SplitMix64 mix** inside `Reservoir::Initialize`: one
master `seed` fans into labelled, statistically-independent substreams
(`Recurrent`/`Input`/`Feedback`/`Bias`/`SrProbe`) via `seed_for(role) =
mix64(seed ^ GOLDEN*role)`. Labels (not a sequential SplitMix stream) so adding a role
later doesn't perturb existing ones. `bias_seed` **deleted** from `ReservoirConfig`, the
ctor init, the member, and `GetConfig`; input and feedback no longer share a stream.

Consequence for the ensemble: a member is now fully determined by **one** seed, so G5's
"derive 2M role-tagged seeds" collapses to one deterministic line —
`seed_i = mix64(ensemble_seed ^ GOLDEN*(i+1))` — with the role-tagging pushed down into
the reservoir where it's reused by everyone. Verified: 23 C++ targets, pytest 57/57,
batch + smoke R2=1.0. (One-time cost: a given `seed` now realizes a *different* reservoir
than before; saved checkpoints are unaffected since weights are serialized, but tuned
seeds in comments would need re-finding.)

**Recommendation:** DONE.

### G6. Diagnostic accessors
**Status:** RESOLVED (§7.4) — preceded by a §8 sufficiency audit that reshaped the list.

Auditing §8 first (per the resolution order) showed the experiment set was
under-specified, so two experiments were **added** and the accessor list re-derived from
the expanded set:
- **Decorrelation axis** — `ρ̄` (mean pairwise correlation of member errors) reported
  alongside every κ-curve; the thesis's "decorrelating regime" qualifier is now *measured*,
  not an escape hatch. Promotes per-member outputs from "nice for the baseline" to
  **load-bearing**.
- **Replication + decisiveness protocol** — K ≥ 5 independent ensemble seeds, mean ± spread,
  "beats" = margin beyond seed noise (form committed; concrete margin task-specific).

Three further controls I proposed (κ sign-symmetry, gate-threshold tuning, common-mode
bias) were **rejected as out of scope** by the user — explicitly not worth the effort.
Dropping the gate-tuning sweep is what keeps the surface lean (no `consensus_err_` getter).

Committed §7.4 surface (each traced to a kept experiment): `MemberOutput(i, out)` +
`AllMemberOutputs(MxD)` (decorrelation, single-member baseline), `Kappa()` (intensity
sweep + ramp trace), `GateOpen()` + `CurrentStep()` (annotate the ramp ablation). **Not**
exposed: per-member error (G3 — gate reads consensus error only; outputs suffice) and the
raw `consensus_err_` signal (only the dropped gate-tuning sweep wanted it). All `const`,
no mechanism change.

**Recommendation:** DONE.

---

## Polish — structural / minor

### G7. Section §10 is missing entirely
**Status:** RESOLVED. Renumbered "Open questions" §11 → §10 (no empty "Related work"
placeholder restored — there's no drafted content for it, so renumbering is the honest
fix). Updated the two in-text cross-refs §11.2 → §10.2 and §11.3 → §10.3; verified no §11
reference remains in the doc.

**Recommendation:** DONE.

### G8. `|κ|` notation inconsistent with the κ > 0 convention
**Status:** RESOLVED. Dropped the bars in both §8 occurrences (thesis "monotone in `κ`",
intensity sweep "accuracy-vs-`κ`"); the §10.4 open question already wrote plain κ. Since
§4.1 fixes κ > 0, `|κ| = κ`.

**Recommendation:** DONE.

### G9. Parallelism note (optional)
**Status:** RESOLVED (§5, M paragraph). Noted that members interact only through the
shared consensus (one reduction per tick), splitting each tick into two member-parallel
regions (read all `y_i`; then train/inject/step); the stepper is serial today and would
parallelize with plain C++ standard threads. Not a design driver — one note, placed beside
the per-step cost discussion.

**Recommendation:** DONE.

### G10. Confirm D = `NumOutputs()` = `num_feedback_channels`
**Status:** RESOLVED (§3). Added a "One D, three roles" note where D is first introduced,
stating `D = NumOutputs() = num_feedback_channels` explicitly (readout output count =
consensus/deviation dimension = feedback-channel count; no separate sizing knob).

**Recommendation:** DONE.

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
**Status:** RESOLVED + IMPLEMENTED (2026-06-19). The ESN internal-F apparatus is deleted and
the external-feedback port landed. Code changes:
- **ESN** — deleted `FeedbackConfig`, the `ESNConfig::feedback` field, the entire feedback-
  training orchestration (`TrainFeedbackCycle`×2, `ValidateClosedLoop`, `ProbeLoss`,
  `TrainFeedbackCycleImpl`, `RequireFeedbackTraining`, the MSE/CE loss helpers), the whole
  telemetry layer (`FeedbackCycleInfo/Record`, `FeedbackTelemetry`, the ring + counters,
  `GetFeedbackTelemetry/History`), `feedback_readout_` (F), `InjectFeedbackClamped`,
  `MakeFeedbackReadoutConfig`, `HasFeedback`, `LastFeedbackRaw`, `Get/SetForceZeroFeedback`,
  `Get/SetFeedbackState`. `StepLive` trimmed to input-only; added
  `StepLiveExternalFeedback(inputs, φ)` (raw inject, no clamp). Added `NumFeedbackChannels()`.
- **Reservoir** — relaxed the `D|N` divisibility throw to any `D ≤ N`; added vector-form
  `InjectFeedback(const float* φ, count)`. (`feedback_scaling` etc. untouched — the substrate
  was already sound.)
- **Upstream** — deleted the dead `NARMAFeedback` example (+ CMake target); rewrote `main.cpp`
  to the reservoir snapshot/restore fidelity driver (now exercises the D≤N feedback path with
  D=3). ZERO Readout/HCNN changes; Python bindings untouched (feedback was never exposed).
- **Verified:** 20 C++ targets compile; `main.cpp` fidelity suite passes incl. the +feedback
  (fb=3, non-dividing) config; BasicPrediction batch R2=1.0; Python pytest 57/57.
- **Orphaned docs purged:** deleted `FeedbackTrainingMethodology.md` and
  `NARMAFeedbackCampaign.md` (documented removed code); scrubbed their references from
  `README.md` (the "Closed-Loop Feedback" section + Documentation-table row) and the
  `Reservoir.h` RestoreSnapshot doc comment.

Prior design state (kept for context):

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
  inject; relax the `D|N` throw (`:50`) to any `D ≤ N`.
  - *Substrate verification (was G13) — folded in, no standalone harness.* The gather
    (`:259-266`) is a line-for-line **twin of the input port**, which is exercised on every
    prediction; only the feedback weight-block offset + `fb_rng` init are feedback-specific.
    A standalone exact-math probe would need new weight-accessor API for a weak assertion —
    poor value. Decision (2026-06-19, user): **skip the harness; the first ensemble feedback
    run (Lorenz) is the end-to-end verification** — any gather bug shows up immediately as
    garbage dynamics. G13 closed as won't-do.
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
