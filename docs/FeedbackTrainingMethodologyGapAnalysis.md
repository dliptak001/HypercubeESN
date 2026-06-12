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

### A4. Warmup / `InitOnline` semantics under the closed loop are unaddressed

`ESN::InitOnline` drives a warmup phase before any training. §6.8/§6.9
establish that `F` drives the loop from step one — so presumably warmup
is also closed-loop with frozen `F` — but the doc never says so. There
is also a code-level corollary: `Readout::InitOnline` is what *builds*
the CNN, so `F` must be initialized before its first `PredictRaw` or the
deployed loop's very first `f` evaluation has no network to call. When
`F.InitOnline()` happens (construction? `ESN::InitOnline`?) is
unspecified.

### A5. The probe error metric is only defined for regression `P`

§3 defines the probe metric as squared error of `P`'s prediction.
Nothing restricts `P` to `ReadoutTask::Regression`, and for
classification "squared error" of logits is undefined behavior,
spec-wise. Either scope v1 to regression tasks explicitly or define the
classification probe loss (e.g. cross-entropy). Related: for
multi-output regression, MSE-across-outputs should be stated.

### A6. Validation evaluation breaks stream continuity; the doc doesn't say how to avoid it

§6.9's primary convergence criterion is held-out validation NRMSE — but
evaluating a validation stream means driving the reservoir through it,
destroying the training-trajectory state. The obvious fix (snapshot
before validation, restore after — the primitive exists) is never
stated, and validation runs closed-loop with the current `F`, which
should also be said explicitly.

## B. Secondary underspecifications

- **B1. Serialization:** `ESN::GetReadoutState`/`SetReadoutState`
  handles one readout. Two-readout checkpointing is not in the §8
  tracker at all.
- **B2. Pass-1 cadence left as an either/or:**
  "(`TrainLiveStepRegression`, or mini-batch accumulation across
  cycles)" — v1 should commit to one (per-step online is the natural
  pair to §6.10).
- **B3. `P`'s lr schedule in streaming mode:** `CosineLR` needs a
  horizon; an open-ended stream has none. Both the pre-train phase and
  the alternation phase need a stated lr policy for `P`.
- **B4. Tie-breaking and boundary equality:** `E+ == E−` (pick which
  sign?) and `min(E+, E−) == E0` exactly (reject, per strict `<`?) —
  trivial, but a deterministic spec matters for reproducibility, and
  §6.11 notes deep saturation produces exactly these equalities.
- **B5. Telemetry plumbing and thresholds:** §8 lists *what* to log but
  not how it is exposed (return struct per cycle? callback? stdout?),
  over what window the §7.4 variance is computed, or what raw `|F(x)|`
  level counts as "saturation" for the §6.11 watch. The A/B benchmark
  (§9.4) should also state whether it controls for the ~4× compute
  overhead.

## C. Doc inconsistencies (against the code or itself)

- **C1. §3's `Sx` definition contradicts the implementation:** it says a
  snapshot captures "the slice-ring rotation." The implemented capture
  is **canonical / rotation-free** — rotation is deliberately *not*
  captured; restore re-homes the ring — exactly as §8's tracker row and
  `Reservoir.h` describe. §3 should match.
- **C2. §1's commit range is stale:** "(commits `a8a350a..cf33018`)" is
  cited for the complete mechanism list, but that list includes
  `TakeSnapshot`/`RestoreSnapshot`, which landed later in `696d762`.
- **C3. §2's table specifies `ReadoutActivation::TANH` for `F` with no
  cross-reference to §6.11.** Since the entire point of §6.11 is that
  this field does *not* bound the output (it is per-Conv-layer), citing
  it un-annotated in the architecture table invites exactly the
  misreading the doc elsewhere dismantles.
- **C4. The single-channel broadcast collapse is not carried over.**
  `feedback_mechanisms.md` explicitly flags that a broadcast scalar
  makes every vertex's dim-neighbor gather collapse (the effective
  injection becomes `f ×` a per-vertex weight-sum). v1 is exactly that
  case — one channel driving all N vertices — and the methodology doc
  never mentions it. It is not fatal (a scalar times a fixed random
  projection is the intended mechanism, and it is precisely the rank-1
  structure §2.3 compares against), but a known documented risk of the
  mechanism should appear in the methodology that deploys it.

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
