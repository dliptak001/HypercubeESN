# Feedback Training Methodology — Gap Analysis

> Status: **analysis snapshot** (June 2026). A gap analysis of
> `FeedbackTrainingMethodology.md` performed before the ESN-side
> implementation began, cross-checked against the actual interfaces
> (`ESN.h`, `Readout.h`, `Reservoir.h`) and against
> `feedback_mechanisms.md`. Section references (§) point into
> `FeedbackTrainingMethodology.md` unless stated otherwise.
>
> Overall verdict: the design is internally consistent and nothing here
> invalidates it. The gaps cluster into four groups: **6 spec gaps an
> implementer must resolve before coding (A), 5 secondary
> underspecifications (B), 4 doc inconsistencies (C), and 2 conceptual
> risks §7 does not yet flag (D).**
>
> **Update (June 2026): all six A-items are resolved** — each carries an
> inline "Resolved" block recording the decision and where it was
> codified (§6.2 commit-time corollary, §6.13–§6.17). All B and C items
> and D1 are resolved inline below; D2 remains open.

---

## A. Blocking spec gaps (decisions needed before ESN implementation)

### A1. What feedback value does the Pass-1 commit inject after an accepted cycle?

The doc is ambiguous on the single most-executed line of the whole scheme.
Step g trains `F` *before* step i's restore-and-commit, so by commit time
`F` has moved:

```
Pass 2 (accepted +ε):   step g trains F one step  →  F′
step i: restore Sx
Pass 1 commit: inject f = ???
   option A: f* = Sf+ε    the value the probes actually verified
   option B: F′(Sx)       live post-update output — never probed,
                          lands somewhere between Sf and f*  (§2.1/§6.2 imply this)
   option C: Sf           stale pre-update output — known inferior
```

§4 Pass-1 step 1 says "inject `f = F(x)`," which after step g means
option B — a value no probe ever tested. The difference is ≤ ε so it is
nearly harmless, but A (commit what you verified) and B (stay strictly
on-policy, consistent with §6.2) are both defensible and the doc must
pick one.

> **Resolved (June 2026): option B.** The commit injects the live
> post-update `F′(Sx)`, codified in §4 (Pass-1 step 1 and step i), §5,
> and §6.2 (commit-time corollary, with option A as the recorded
> rejected alternative). Rationale: the per-cycle certificate option A
> honors is ~2/3 chance at zero margin (§6.6), while option B preserves
> the on-policy property §6.2 stakes the design on and gives one uniform
> commit path for accepted and rejected cycles. The `Sf` cache is
> probes-only; the commit does a fresh `F` forward.

### A2. The `f ≡ 0` kill-switch has no defined mechanism

§9.1 requires "feedback configured but `f ≡ 0` forced" and §7.6 mandates
a kill-switch comparison "in every experiment" — but the
`enable_feedback` toggle was deliberately removed (`cde5d2f`), and
nothing in the doc says where the force-zero now lives (an ESN-level
flag? a debug hook at the §6.11 clamp seam?). The verification plan
depends on a capability that no §8 tracker row builds.

> **Resolved (June 2026): the Reservoir toggle stays gone; two
> mechanisms at the right layers, codified in new §6.13** (cross-refs
> added in §7.6, §8, §9.1, §9.4). The §9.1 "configured but `f ≡ 0`" arm
> needs no new capability at all: `feedback_scaling = 0` yields a
> bit-identical realization with the feedback block exactly zeroed
> (weights are drawn then scaled — `Reservoir.cpp` `Initialize`), every
> probe returns exactly equal errors, and the strict-`<` accept means
> `F` never trains — a compute-matched control for free. The §7.6
> per-experiment lesion is a new ESN-level `force_zero_feedback` runtime
> flag at the §6.11 clamp seam (rides on A3's config surface; tracked in
> the §8 closed-loop-stepping row), enabling post-hoc lesioning of a
> trained system.

### A3. No config surface or defaults for `F` and the probe hyperparameters

`ESNConfig` has exactly one `readout`. The doc never specifies:

- where `F`'s `ReadoutConfig` lives;
- what its capacity defaults are (§7.7 says "small — e.g.
  `num_layers = 1`, modest channels" but that is a hint, not a spec);
- whether its `seed` must differ from `P`'s;
- **no default or even starting range for ε anywhere in the document**,
  nor for `F`'s learning rate, nor a recommended `feedback_scaling`.

§8 lists them as "hyperparameters" and stops. The margin is the only
knob with a stated default (0).

> **Resolved (June 2026): new §6.14.** Config surface is a nested
> `ESNConfig::feedback` block — `F`'s full `ReadoutConfig` plus
> `epsilon`, `margin`, `lr`, and the §6.13 `force_zero` flag — with
> enablement still gated solely on `num_feedback_channels > 0` (v1
> validates `== 1`) and the ESN forcing `dim`/`num_outputs`/`task` on
> `F` exactly as it does for `P`. Defaults: `num_layers = 1`,
> `conv_channels = 8`, `seed = 43` (no must-differ constraint —
> determinism only), `weight_decay = 0` (telemetry-gated), **ε = 0.05**
> (sane range 0.01–0.2; ε sets creep rate and locality, not noise
> immunity — probes are deterministic replays), **`F` lr = 2e-4
> constant** (no horizon for cosine; ~7× slower than `P` per §7.3;
> tuned via a new step-realization-fraction telemetry item in §8), and
> `feedback_scaling` stays at 0.5. Explicitly flagged as
> defaults-with-tuning-signals, not measured values. Side effect: the
> `F` half of B3 (lr schedule in streaming mode) is resolved; `P`'s
> streaming lr policy remains open.

### A4. Warmup / `InitOnline` semantics under the closed loop are unaddressed

`ESN::InitOnline` drives a warmup phase before any training. §6.8/§6.9
establish that `F` drives the loop from step one — so presumably warmup
is also closed-loop with frozen `F` — but the doc never says so. There
is also a code-level corollary: `Readout::InitOnline` is what *builds*
the CNN, so `F` must be initialized before its first `PredictRaw` or the
deployed loop's very first `f` evaluation has no network to call. When
`F.InitOnline()` happens (construction? `ESN::InitOnline`?) is
unspecified.

> **Resolved (June 2026): new §6.15** (cross-refs in §6.8 and both §8
> ESN rows). Three decisions: (1) `F`'s CNN is built **eagerly at ESN
> construction** whenever `num_feedback_channels > 0` — no "built yet?"
> state, the before-first-step constraint holds by construction; stated
> consequence: the random `F` is checkpoint-worthy immediately
> (`IsTrained()` set), which is correct since §6.8's random `F` is the
> live policy. (2) **Warmup runs closed-loop with `F` frozen** — the
> §6.9 distribution-jump argument telescoped back one boundary; the
> §6.13 `force_zero` flag applies during warmup too. (3) `P`'s init is
> unchanged (after warmup) — only `F` has the ordering constraint. Plus
> the tracker inversion made explicit: closed-loop warmup is the first
> consumer of the closed-loop step driver, so that capability precedes
> all training orchestration.

### A5. The probe error metric is only defined for regression `P`

§3 defines the probe metric as squared error of `P`'s prediction.
Nothing restricts `P` to `ReadoutTask::Regression`, and for
classification "squared error" of logits is undefined behavior,
spec-wise. Either scope v1 to regression tasks explicitly or define the
classification probe loss (e.g. cross-entropy). Related: for
multi-output regression, MSE-across-outputs should be stated.

> **Resolved (June 2026): both tasks, multi-output, from v1 — new §6.16
> plus a rewritten §3 probe-loss definition.** The probe loss `L` is
> per-task: regression = MSE across outputs; classification = softmax
> cross-entropy of the target label from `P`'s logits (log-sum-exp
> stabilized) — not 0/1 accuracy, which is a step function at ε scale
> (accepts would never fire and the feedback path would be structurally
> dead for classification). `F` is untouched by `P`'s task — always
> `Regression`/1-output per §6.14. The exact-equality arguments (§6.13
> kill-switch, §9.3 ε = 0 sanity) survive for either loss. §6.9's
> convergence criterion generalized (NRMSE / validation CE), §9.3 runs
> per task, and §9.4 gains a classification arm (StreamingText
> next-char). Regression-specific wording generalized in §4 and §6.1.

### A6. Validation evaluation breaks stream continuity; the doc doesn't say how to avoid it

§6.9's primary convergence criterion is held-out validation NRMSE — but
evaluating a validation stream means driving the reservoir through it,
destroying the training-trajectory state. The obvious fix (snapshot
before validation, restore after — the primitive exists) is never
stated, and validation runs closed-loop with the current `F`, which
should also be said explicitly.

> **Resolved (June 2026): new §6.17** (cross-refs in §6.9 and the §8
> orchestration row). Snapshot-bracketed validation at cycle boundaries
> only: `TakeSnapshot` → `ResetReservoirOnly` → closed-loop washout of
> `W` unscored steps (`W` reuses `InitOnline`'s warmup count — no new
> hyperparameter) → closed-loop scoring with `F` and `P` frozen
> (`force_zero` respected, same principle as §6.15 warmup) →
> `RestoreSnapshot`. The zero-reset is the non-obvious part: entering
> from the live training state would make successive evaluations differ
> in both weights and entry state, putting noise in the very series the
> plateau criterion watches; resetting makes scores differ only because
> `F`/`P` changed. Fixed held-out segment, cadence `N_val` (default
> 1000 cycles). Only the reservoir needs protecting — validation is
> forward-only, so readout weights and Adam moments are untouched by
> construction.

## B. Secondary underspecifications

- **B1. Serialization:** `ESN::GetReadoutState`/`SetReadoutState`
  handles one readout. Two-readout checkpointing is not in the §8
  tracker at all. *(Resolved June 2026: §8 gains a two-readout
  checkpointing row — `GetFeedbackState`/`SetFeedbackState` mirroring
  the existing pair; `F` is persist-worthy from construction per §6.15;
  Adam moments are not serialized for either readout — a stated,
  accepted v1 limitation.)*
- **B2. Pass-1 cadence left as an either/or:**
  "(`TrainLiveStepRegression`, or mini-batch accumulation across
  cycles)" — v1 should commit to one (per-step online is the natural
  pair to §6.10). *(Resolved June 2026: §4 Pass-1 step 2 commits v1 to
  per-step online — symmetry with `F`'s zero-staleness stance, §7.3
  freshness, no new hyperparameter; mini-batch accumulation recorded as
  a non-v1 variant.)*
- **B3. `P`'s lr schedule in streaming mode:** `CosineLR` needs a
  horizon; an open-ended stream has none. Both the pre-train phase and
  the alternation phase need a stated lr policy for `P`. *(Resolved June
  2026 — `F`: constant 2e-4, §6.14. `P`: §6.9 — pre-train cosine over a
  declared `pretrain_steps` budget (default 10 000, doubling as the
  phase backstop), annealing into a constant alternation lr `p_lr`
  (default 5e-4; the conventional cosine floor 1.5e-5 would invert
  §7.3's timescale ordering vs. `F`'s 2e-4), no lr discontinuity at the
  phase boundary. Both knobs live in `ESNConfig::feedback`.)*
- **B4. Tie-breaking and boundary equality:** `E+ == E−` (pick which
  sign?) and `min(E+, E−) == E0` exactly (reject, per strict `<`?) —
  trivial, but a deterministic spec matters for reproducibility, and
  §6.11 notes deep saturation produces exactly these equalities.
  *(Resolved June 2026: every exact equality rejects — §4 step f and a
  new §6.6 paragraph. The strict `<` is load-bearing for the §6.13
  kill-switch; a direction tie has no verified direction, and a fixed
  tie-break sign would reintroduce the asymmetry bias §6.6 already
  rejects. In the saturation regime, `tanh` monotonicity collapses
  direction ties into baseline equality anyway.)*
- **B5. Telemetry plumbing and thresholds:** §8 lists *what* to log but
  not how it is exposed (return struct per cycle? callback? stdout?),
  over what window the §7.4 variance is computed, or what raw `|F(x)|`
  level counts as "saturation" for the §6.11 watch. The A/B benchmark
  (§9.4) should also state whether it controls for the ~4× compute
  overhead. *(Resolved June 2026 in the §8 telemetry row: POD
  `FeedbackTelemetry` accumulated by the ESN, const-accessor exposure,
  no callbacks/stdout in v1; ring buffer and §7.4 variance window =
  1000 cycles, aligned with `N_val`; saturation = raw `|F(x)| > 2.0`
  (probe lever `1 − tanh² ≈ 0.07`), weight-decay responder suggested
  past 50% of the window. The compute-overhead control was already
  closed by A2 — §6.13's scaling-zero arm, noted in §9.4.)*

## C. Doc inconsistencies (against the code or itself)

- **C1. §3's `Sx` definition contradicts the implementation:** it says a
  snapshot captures "the slice-ring rotation." The implemented capture
  is **canonical / rotation-free** — rotation is deliberately *not*
  captured; restore re-homes the ring — exactly as §8's tracker row and
  `Reservoir.h` describe. §3 should match. *(Resolved June 2026: §3 now
  states the canonical capture, the re-homing restore, and that
  bit-exactness is a property of the restored dynamics, not the buffer
  layout.)*
- **C2. §1's commit range is stale:** "(commits `a8a350a..cf33018`)" is
  cited for the complete mechanism list, but that list includes
  `TakeSnapshot`/`RestoreSnapshot`, which landed later in `696d762`.
  *(Resolved June 2026: `696d762` is the immediate next Reservoir commit
  after `cf33018`, so the range extends contiguously — §1 now cites
  `a8a350a..696d762`, covering every bullet in the list.)*
- **C3. §2's table specifies `ReadoutActivation::TANH` for `F` with no
  cross-reference to §6.11.** Since the entire point of §6.11 is that
  this field does *not* bound the output (it is per-Conv-layer), citing
  it un-annotated in the architecture table invites exactly the
  misreading the doc elsewhere dismantles. *(Resolved June 2026: the §2
  row now carries the inline annotation — per-Conv-layer only, output
  bound is the §6.11 seam clamp — matching the §6.14 defaults table.)*
- **C4. The single-channel broadcast collapse is not carried over.**
  `feedback_mechanisms.md` explicitly flags that a broadcast scalar
  makes every vertex's dim-neighbor gather collapse (the effective
  injection becomes `f ×` a per-vertex weight-sum). v1 is exactly that
  case — one channel driving all N vertices — and the methodology doc
  never mentions it. It is not fatal (a scalar times a fixed random
  projection is the intended mechanism, and it is precisely the rank-1
  structure §2.3 compares against), but a known documented risk of the
  mechanism should appear in the methodology that deploys it.
  *(Resolved June 2026: §2.2 now carries the collapse note — effective
  injection `f × (Σᵢ w_{v,i})`, expressivity = one fixed random
  N-vector of per-vertex gains, intended-not-defect, with
  `num_feedback_channels > 1` named as the collapse-breaking headroom.)*

## D. Conceptual risks §7 does not yet flag

### D1. Self-referential target drift

Every accepted cycle sets `F`'s target to *its own current output* ± ε.
§6.10 covers staleness, but not the fixed-point question: on
autocorrelated streams, chance accepts can streak in one direction, and
nothing bounds the random walk of `Sf` except the indirect §6.11
saturation watch and an optional weight decay mentioned in passing. The
dynamics of "a regressor chasing its own output plus noise" deserve a
bullet in §7 — it is a different failure mode from the §7.4 variance
collapse (drift vs. flatline).

> **Resolved (June 2026): new §7.8.** Records the mechanism (sign
> streaks × `F`'s generalization on autocorrelated streams = a
> compounding walk; nothing intrinsic bounds raw `Sf` — the clamp caps
> the effect, not the walk), the separation from §7.4 (drift vs.
> flatline; mean-watch vs. variance-watch), and the responders — weight
> decay ~1e-4 (§6.11's knob, second job) and a positive margin (§6.6's
> knob). Windowed mean of `F(x)` and accept-sign balance added to the
> §8 telemetry row, both free derivatives of the B5 ring buffer.

### D2. The effective perturbation shrinks before saturation is "deep"

Probes act in post-clamp space with lever `≈ ε·(1 − tanh²(Sf))`, while
the creep step in pre-clamp space stays a constant ε. So the *tested*
effect size and the *trained* step size diverge continuously as `|Sf|`
grows — not just at the §6.11 "deep saturation" extreme. The accept-rate
and saturation telemetry would eventually catch it, but the doc treats
it as a binary cliff when it is a smooth degradation; an adaptive-ε note
in §6.7's upgrade list would close it.

---

## Bottom line

Nothing here invalidates the design. A1–A6 are the items that would
actually stall an implementer mid-task (A1 and A2 especially — one is a
semantic fork in the inner loop, the other is a verification dependency
with no owner); B and C are an afternoon of doc edits; D1/D2 are two
paragraphs in §7.

**As of June 2026 the blocking layer is clear: A1–A6 are all resolved**
(decisions codified in the methodology — §6.2 commit-time corollary and
§6.13–§6.17). What stands between this analysis and a clean sheet is
the afternoon of B/C doc edits and the two D paragraphs.
