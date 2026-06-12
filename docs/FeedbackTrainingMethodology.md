# Feedback Training Methodology

> Status: **design finalized** (June 2026). This document is the training-scheme
> design for integrating the Reservoir's feedback path into the `ESN` class.
> The design decisions are settled and recorded in §6, each with its rationale,
> the alternatives considered, and the telemetry that would trigger a
> data-driven rethink. Conceptual criticism of the scheme is in §7.
> Implementation is in progress — see the §8 tracker (the Reservoir-level
> mechanics of §1 and the snapshot/restore primitive are done; the ESN-side
> work is pending).

---

## 1. What exists today (Reservoir level)

The Reservoir already has the complete feedback *mechanism* (branch `feedback`,
commits `a8a350a..cf33018`):

- `ReservoirConfig::num_feedback_channels` (0 = path disabled, no allocation) and
  `ReservoirConfig::feedback_scaling` (DIM-invariant `1/sqrt(dim)` fan-in
  normalization, mirroring `input_scaling`).
- A dedicated feedback weight block (`n * dim` weights) drawn from its own RNG
  stream, sitting between the input block and the recurrent block in
  `vtx_weight_`.
- `InjectFeedback(channel, value)` — stages a scalar onto the contiguous vertex
  block for that channel, exactly like `InjectInput`. Consumed and **cleared by
  every `Step`**, so it must be re-injected each timestep.
- `TakeSnapshot`/`RestoreSnapshot` — bit-exact capture and restore of the
  persistent dynamical state (the branch-point primitive for the training
  probes of §4; see §8).
- The feedback weights sit **outside** the spectral-radius estimate; closing the
  loop is not covered by the SR target (see §7.6).

The `ESN` class is currently strictly open-loop. This document defines how the
loop gets closed and trained.

## 2. Architecture (v1 scope)

One feedback channel only (`num_feedback_channels = 1`), driven by a **dedicated
feedback readout** `F`:

| Component | Role | Config |
|---|---|---|
| Primary readout `P` | The task readout, exactly as today | `cfg.readout` (task-specific) |
| Feedback readout `F` | Computes the scalar feedback signal from reservoir state | `ReadoutTask::Regression`, `num_outputs = 1`, `ReadoutActivation::TANH`, same `dim` as `P` (the stride-subsampled geometry) |

Both readouts consume the **same stride-subsampled feature vector**
(`NumOutputVerts()` floats) the ESN already produces via `CopyLiveState`.

### 2.1 Deployed (inference-time) loop

Per-step contract, following the `y(t-1)` convention documented on
`Reservoir::InjectFeedback`:

```
loop over t:
    InjectInput(u(t))
    InjectFeedback(0, f(t))      where f(t) = tanh(F(x(t-1)))   // previous state, clamped (§6.11)
    Step()
    x(t) = subsampled Outputs()
    y(t) = P(x(t))               // task prediction
    f(t+1) = tanh(F(x(t)))       // feedback for the next step
```

`F` maps the current reservoir state to the feedback drive for the *next* step.
This is the only causally consistent wiring: the feedback consumed by `Step` at
time `t` must be computable before `x(t)` exists.

### 2.2 What this is, conceptually

`f(t) = F(x(t-1))` injected through a fixed random weight block is a **learned,
state-dependent, low-rank recurrent augmentation** of the reservoir — a single
trained nonlinear scalar pathway wrapped around 2^dim fixed ones. It is *not*
Jaeger-style output feedback (`W_fb` feeds the task output back; here the fed-back
signal is a free auxiliary control signal with no predefined semantics — its only
job is to reduce primary task error).

### 2.3 Differentiation from the nearest prior art

The nearest prior art is Ehlers, Nurdin & Soh (2025), "Improving the
performance of echo state networks through state feedback," *Neural Networks*
184:107101 (<https://arxiv.org/abs/2312.15141>) — trained state feedback
through the input pathway, with reported performance gains on standard
benchmarks; see also `feedback_mechanisms.md` §Related-work. Their mechanism:
the input is augmented as `u_k → u_k + Vᵀx_k`, so the reservoir update becomes
`x_{k+1} = g((A + BVᵀ)x_k + Bu_k)` — the feedback flows through the *shared*
input weights `B`, and the closed loop is exactly a **rank-1 modification of
the recurrent matrix**. `V` is trained by batch gradient descent (their
Appendix A).

The two designs agree on the headline mechanism — *a trained, state-dependent
scalar fed back as an input-like drive* — and differ on every implementation
axis:

| | Ehlers, Nurdin & Soh (2025) | This design |
|---|---|---|
| Fed-back signal | scalar | scalar |
| Function computing it | **linear** gain `Vᵀx` | **nonlinear** CNN readout `F(x)` (tanh-clamped, §6.11) |
| Injection path | shared input weights `B` (`A → A + BVᵀ`) | dedicated feedback weight block, separate from the input path (§1) |
| Training signal | gradient descent on `V` against an explicit cost | zeroth-order ±ε perturbation probes scored by the primary readout's error (§4) — no gradient through the reservoir |
| Closed-loop form | `A + BVᵀ`: a rank-1 linear modification, directly analyzable | no closed form: nonlinear `F` through its own block |

The trade is expressiveness vs. analyzability. This design is the more
expressive on every differing axis — a nonlinear state-dependent drive, its own
weight block, no need for any differentiable path — but it gives up the clean
`A + BVᵀ` story: there is no linear-algebraic handle on the closed-loop
dynamics, which is why stability is treated as an empirical question with
telemetry guards rather than a property to be engineered in (§7.6). Their
published gains are read here as evidence that the *mechanism* (state→input
scalar feedback) has legs (§7.5); whether the extra expressiveness and the
gradient-free training earn their cost is exactly what the §9 A/B benchmark
exists to answer.

## 3. Definitions

- **Example** — one training example is a **single timestep**: one input sample
  `u(t)` with its target `y(t)` (`H = 1`; rationale in §6.1, which also records
  the `H`-step window generalization as the data-driven fallback).
- **`Sx`** — a full reservoir state snapshot: `vtx_state_`, **all `M` history
  slices**, and the slice-ring rotation — captured and restored bit-exactly via
  `Reservoir::TakeSnapshot`/`RestoreSnapshot` (§8). The per-step `vtx_input_` /
  `vtx_feedback_` staging buffers are cleared by every `Step` and are not part
  of a snapshot; `RestoreSnapshot` additionally clears them.
- **`Sf`** — the scalar feedback operating point of a Pass-2 cycle:
  `Sf = F(Sx)`, the feedback readout's own output at the decision-time state,
  evaluated once per cycle and reused across the cycle's probes (§6.2).
- **`ε`** — the probe perturbation magnitude (hyperparameter, possibly adaptive).
- **Probe error metric** — **MSE** of the primary readout on the probe example;
  with `H = 1` this is simply the squared error of the one prediction. All
  probes within one cycle score the *same target(s)*, so the NRMSE
  normalization (`1/σ(targets)`) is a common constant and MSE orders the probes
  identically at lower cost — and for a single timestep NRMSE is not even
  well-defined (one target has zero variance). NRMSE is still the metric
  *reported* for validation and benchmarks (§9); only the internal probe
  comparisons use MSE.

## 4. Training procedure

Training alternates between two passes over the (sequential) training stream.
One **cycle** = one Pass 1 + one Pass 2.

### Pass 1 — primary readout training (feedback held fixed)

1. The reservoir is at committed state `S0`. Present the next training example:
   inject the input `u` *and* the current feedback signal `f = F(x)` (§2.1;
   from cycle 0 this is `F`'s live output, §6.8), then `Step`. `f = F(x)` is
   always the **live output of the current `F`** — after an accepted Pass-2
   cycle that is the *post-update* readout, recomputed at commit time; it is
   never the probe winner `f*` and never the cached `Sf` (§6.2).
2. Train `P` on the resulting (state, target) pair exactly as today
   (`TrainLiveStepRegression`, or mini-batch accumulation across cycles).
3. `F` is not touched. The reservoir state advances for real (no restore).

### Pass 2 — feedback readout training (primary held fixed)

Let `Sx` = snapshot of the reservoir state after Pass 1, and `Sf = F(Sx)` = the
current feedback operating point (§6.2). Let `(u', y')` = the next training
example.

| Step | Action |
|---|---|
| a | Snapshot `Sx`; evaluate `Sf = F(Sx)` once. |
| b | **Baseline probe:** inject `u'` and feedback `Sf`; `Step`; forward-pass `P`; **no training**. |
| c | `E0` = squared error of `P`'s prediction against `y'`. |
| d | Restore `Sx`. **Positive probe:** same input `u'` with feedback `Sf + ε`; `E+`. |
| e | Restore `Sx`. **Negative probe:** same input with `Sf − ε`; `E−`. Both directions are always probed (§6.6). |
| f | **Accept/reject:** if `min(E+, E−) < E0 − margin`, the winning perturbation direction is a locally verified improvement. Set `f* = Sf ± ε` (winning sign). |
| g | **Train `F` (one regression step):** input = subsampled features of **`Sx`** (the decision-time state, *not* any post-step probe state — §6.4), target = `f*`. |
| h | If neither probe beats baseline by the margin, train nothing this cycle. |
| i | Restore `Sx`. Return to Pass 1, which consumes `(u', y')` *for real* (§6.5 — the probe example is re-presented as the next Pass-1 example, preserving stream continuity). The feedback that commit injects is `F`'s live post-update output, not `f*` (§6.2). |

Per §6.11, every injected feedback value passes through the `tanh` clamp at
the seam; `Sf`, `±ε`, and the training target `f*` all live in the pre-clamp
space (the table above writes the pre-clamp values).

Repeat cycles until the convergence criteria (§6.9) are met. This is the
**streaming-mode** formulation — it is inherently interactive (the probes
re-execute the reservoir from a snapshot), and v1 is streaming-only (§6.12).

### 4.1 Why perturbation search at all

There is no teacher for `F`. Jaeger-style teacher forcing works when the fed-back
signal *is* the task output (the teacher is the target series). Here the feedback
signal has no ground truth — the only available training signal is "did the
primary task error go down." The reservoir is not differentiated through (and the
HCNN readout's gradient stops at the readout input), so a zeroth-order /
finite-difference probe is the cheapest honest way to extract a training signal.
The cost of that honesty is quantified — unflatteringly — in §7.1.

## 5. Worked cycle (concrete, H = 1)

```
state: S0 (committed)        stream: ... | (u_k, y_k) | (u_k+1, y_k+1) | ...

PASS 1 on example k:    inject u_k, f = F(x); Step; train P on (state, y_k).  → state S1
PASS 2 on example k+1:
    snapshot S1, Sf = F(S1)                       (on-policy operating point, §6.2)
    probe E0:  S1 --inject u_k+1, f=Sf----> Step, P forward: e = 0.0412
    probe E+:  S1 --inject u_k+1, f=Sf+ε--> Step, P forward: e = 0.0398   <- improvement
    probe E-:  S1 --inject u_k+1, f=Sf-ε--> Step, P forward: e = 0.0421
    accept +ε:  train F one step:  F(subsampled S1) → Sf + ε     (F becomes F′)
    restore S1
PASS 1 on example k+1:  inject u_k+1, f = F′(S1)  (live post-update output,
                        §6.2 — not Sf+ε); re-presented, committed for real   → state S2
PASS 2 on example k+2:  ...
```

Each example is consumed twice: once as a Pass-2 probe target, once as a Pass-1
training example. Reservoir state continuity is never broken. A full cycle
costs 4 reservoir steps and 4 `P`-forwards (3 probes + 1 commit) plus one
`P`-train and, on accept, one `F`-train — a ~4× per-timestep overhead versus
plain streaming training.

---

## 6. Design decisions

Each subsection records one settled decision: what was chosen, why, the
alternatives considered, and where applicable the telemetry that would trigger
a data-driven rethink.

### 6.1 Example granularity: one timestep (H = 1)

An example is a single timestep, the probe error is a single squared error, and
there is no window hyperparameter to tune — the simplest mechanism that can
work. Any move away from `H = 1` must be **data-driven** (triggered by the
telemetry below), not by untested hypothesis.

`H = 1` is more defensible than the noise concern suggests:

- **The probe is exact.** With one step, the feedback injected during the probe
  *is* the decision-time feedback. The "held constant across the window"
  approximation that plagues `H > 1` (deployed `F` re-evaluates per step;
  probes cannot) vanishes entirely — `H = 1` is the only choice with zero
  probe/deployment mismatch.
- **Noise averages out in `F`'s regression — and may help.** A single-sample
  comparison is a noisy one-bit direction estimate, but chance accepts are
  symmetric (`+ε` and `−ε` win equally often under pure noise), producing
  zero-mean jitter in `F`'s targets; a real effect produces systematic drift.
  That is exactly the regime stochastic approximation (SPSA-style) is built
  for, and the jitter doubles as exploration — the same role noise injection
  plays in classical output-feedback ESN training.
- **First-order credit is captured.** Feedback injected at step `t` enters the
  very next state update, which `P` reads. One step suffices for the direct
  causal path; only the recurrent echoes go uncredited (§7.2 — true at any `H`).

Costs accepted with eyes open, and the telemetry that triggers a rethink:

- Per-state information is one noisy bit; `F` must smooth across visited
  states. *Watch:* variance of `F(x)` collapsing to a constant (§7.4).
- At zero margin the pure-chance accept rate is **~2/3**, not 1/2 (accept fires
  when the baseline is not the smallest of three exchangeable values), so many
  accepted cycles will be chance wins. Harmless in expectation (zero-mean
  jitter), but the accept-rate telemetry must be read against that floor, not
  against 50%. *Watch:* accept rate indistinguishable from the chance floor
  over a long run.
- *Fallback, recorded but not to be built yet:* `H`-step probe windows (8–32),
  MSE over the window, feedback held constant during probes, `F` trained on the
  window's first decision state only. Adopt only if validation NRMSE shows no
  gain AND the telemetry above is flat.

### 6.2 The operating point is on-policy: Sf = F(Sx)

The feedback signal is, by definition, the single-channel regression output of
the feedback readout — at training time as well as deployment. The Pass-2
operating point is therefore the current policy's own output at the decision
state, `Sf = F(Sx)`; the probe explores ±ε *around the policy*, and an accepted
probe nudges the policy output at that state in the winning direction. This is
a 1-D policy-improvement step (ES/SPSA-flavored), the training data is
generated under the same policy that runs at deployment, and `Sf` needs no
separate bookkeeping — it is recomputable from `Sx`. Operationally, `F(Sx)` is
evaluated once per cycle and reused across the three probes.

*Alternative considered and rejected:* `Sf` as a free-standing carried scalar
that the accepted perturbations random-walk, with `F` distilled from the walk
and taking over only at deployment. Rejected because a slowly-walking global
scalar paired with fast-varying states teaches `F` almost no state-conditional
structure, and training-time feedback (the scalar) would mismatch
deployment-time feedback (`F(x)`).

*One nuance:* during a probe the injected feedback is briefly **not** `F`'s
live output — it is the frozen scalar `Sf ± ε`. With `H = 1` that deviation is
exactly the perturbation under test, applied for exactly one step; it is
inherent to probing.

*Commit-time corollary (after an accepted cycle):* step g trains `F` *before*
step i's restore-and-commit, so by commit time `F` has moved to `F′`. The
Pass-1 commit nevertheless injects the **live post-update output `F′(Sx)`** —
not the verified winner `f* = Sf ± ε`, and not the stale `Sf`.

*Alternative considered and rejected:* committing `f*` ("commit what you
verified"). The certificate it honors is weak — at zero margin ~2/3 of accepts
are pure chance (§6.6); the scheme's correctness lives in `F`'s regression
averaging across cycles, not in any single probe outcome. Committing the live
output instead keeps the committed trajectory exactly on-policy (the property
this section stakes the design on) and keeps Pass 1 a single uniform code
path: `f = F(x)` whether the preceding cycle accepted or rejected — identical
to the deployed loop of §2.1, with no `f*`-or-nothing handoff across the pass
boundary. The risk is bounded by tested values: one update step lands
`F′(Sx)` (to first order) between the probed `Sf` and `Sf ± ε`, so at ε scale
its error sits between tested endpoints. A single Adam step can overshoot
`f*` slightly, putting `F′(Sx)` marginally outside the probed interval — an
O(ε), self-correcting excursion worth a telemetry footnote alongside the
§6.11 saturation watch, not a guard. Operationally: the per-cycle `Sf` cache
is probes-only — the commit performs a fresh `F` forward (+1 small-`F`
forward per cycle, negligible against the ~4× overhead of §5).

### 6.3 All probes replay the same example

All probes within one Pass-2 cycle replay the identical example from the
identical restored state, so the perturbation is the only varying factor.

With different examples per probe, `E+ − E0` would mix the ε-sized perturbation
effect with the O(1) difficulty difference between examples — and on
autocorrelated streams that difficulty delta can correlate with `Sx`, a
state-dependent bias that would train `F` on stream artifacts rather than
feedback quality. Same-example replay is the paired-comparison /
common-random-numbers form of the finite difference, and it preserves the
noise-symmetry property the `H = 1` decision (§6.1) leans on. It is also the
simpler variant operationally: one example serves all three probes plus the
Pass-1 commit, and stream continuity (§6.5) stays clean.

In practice the difference may well prove negligible — but a convention had to
be picked, and this one costs nothing extra.

### 6.4 F trains on the decision-time state

`F` trains on the subsampled features of **`Sx`** (the pre-step snapshot),
target `f*` — not on any post-step probe state.

The role split at deployment: `F` always reads the state on the *left* side of
a `Step` and its output is consumed *by* that `Step`; `P` always reads the
state on the right side.

```
x(t) ──F──► f ──┐
                ▼
u(t+1) ────► [Step] ──► x(t+1) ──P──► ŷ(t+1)
```

In Pass 2, the probes all branch from `Sx`:

```
baseline:  Sx ──inject u', f=Sf────► Step ──► A0 ──P──► E0
positive:  Sx ──inject u', f=Sf+ε──► Step ──► A+ ──P──► E+   ◄ wins
negative:  Sx ──inject u', f=Sf−ε──► Step ──► A− ──P──► E−
```

A winning probe establishes: "from state `Sx` with input `u'`, the answer
should have been `Sf+ε`." The decision happened at `Sx` — `F` was queried
there and answered `Sf` — so the correction is filed against that question:
`F(Sx) → Sf+ε`. **`F` is taught what it should have said; the question was
asked at `Sx`.**

Training on the winning post-step state `A+` instead would break twice:
(1) it answers a question nobody asked — at deployment, `F`'s query at `A+`
concerns the *next* step's feedback, which no probe tested, while the decision
that *was* tested gets no correction; (2) `A+` already carries the imprint of
`Sf+ε`, so `F` would learn to recognize states that already received good
feedback rather than states that need it. The post-step states `A0/A+/A−`
are outcome measurements only, and are discarded by the restore.

(If the `H > 1` fallback of §6.1 is ever adopted: the constant `f*` would be
applied at all H steps of the window, but only the first step's decision state
is `Sx` — train only the `(Sx, f*)` pair.)

### 6.5 Stream continuity: every example is probed, then committed

After Pass 2's probes and final restore, Pass 1 re-presents the probed example
as its real training example. If Pass 1 instead consumed the *following*
example, the committed reservoir trajectory would skip the probed one entirely
— the state would be wrong for everything downstream (the reservoir never
experienced it), and `P` would train on state/input alignments that never occur
at deployment.

```
stream:  ... | u_k+1 | u_k+2 | ...

SKIP after probe (broken):                RE-PRESENT (gapless):

S1 ··probe u_k+1 ×3·· restore S1          S1 ··probe u_k+1 ×3·· restore S1
S1 ──commit u_k+2──► S2                   S1 ──commit u_k+1──► S2
     ✗ S1 never experienced u_k+1;        S2 ··probe u_k+2 ×3·· restore S2
       S2 and everything after it is      S2 ──commit u_k+2──► S3
       misaligned with the stream              ✓ every example: probed, then
                                                 committed; trajectory gapless
```

Every example is used twice (probe, then commit), the committed trajectory is
gapless, and the data cost is zero extra examples.

### 6.6 Both directions are probed; the accept margin defaults to zero

Both perturbation directions are probed every cycle, and the better one is
taken (central-difference flavored). The asymmetric alternative — probe `+ε`
first and test `−ε` only on failure — saves at most one evaluation per cycle
and buys it with a systematic positive-direction bias in `F`'s targets; not
worth it.

On accepting *any* improvement: with `H = 1` (§6.1) single-sample errors differ
by luck constantly — at zero margin the pure-chance accept rate is ~2/3. Under
§6.1's noise-tolerant stance that is acceptable: chance accepts produce
zero-mean target jitter that `F`'s regression averages away, while real effects
drift. A **margin** is therefore a purity/throughput knob, not a correctness
requirement — it trades fewer chance-trained cycles against a lower accept rate
on real effects. The margin is configurable with default 0, to be promoted to a
small relative value only if telemetry shows `F` training dominated by jitter.

### 6.7 The hunt strategy: plain ±ε creep, knowingly naive

An accepted cycle trains `F` toward `f* = Sf ± ε` — a target that differs from
its current output by *at most ε*. Even with `lr` high enough to fully reach the
target, the policy moves ε per accepted cycle. If the useful feedback magnitude
is O(1) and ε = 0.01, that is hundreds of accepted cycles minimum, before
counting rejections and regression under-shoot.

v1 uses the plain ±ε probe anyway. On the record: this is a deliberately naive
hunt strategy, chosen with full awareness of how simple and limited a
fixed-step two-point probe is — not an oversight. Escalation is deferred until
data shows the creep is the bottleneck.

**Implementation seam:** structure the Pass-2 code so "discover `f*` from
(`Sx`, `u'`, `Sf`)" is one isolated, swappable step. Candidate upgrades when
the time comes, in rough order of sophistication: verified line search after a
win (re-probe `Sf ± 2ε, ±4ε…` while improvement continues — each probe cheap,
target *tested* not extrapolated); amplified targets `f* = Sf ± λε`
(extrapolated, riskier); adaptive ε (shrink as the accept rate falls);
multi-point or stochastic probe patterns. None of these change the rest of the
methodology — they only change how far `f*` lands from `Sf`.

### 6.8 No bootstrap phase: F drives the loop from cycle 0

Since the feedback signal is by definition `F`'s output, there is no separate
bootstrap value: at cycle 0 the randomly initialized `F` drives the loop with
an arbitrary output. This is acceptable — the random early output acts like
noise injection, which the output-feedback ESN literature actually *recommends*
during training, and there is no train/deploy distribution switch. Keep
`feedback_scaling` small enough that random early feedback cannot destabilize
the reservoir (its per-neuron effect is bounded by the `tanh` in `UpdateState`
regardless), and the §6.11 clamp tames the untrained output's magnitude as
well.

### 6.9 Schedule and convergence: pre-train P, then alternate

`P` trains alone to near-convergence first; only then does the Pass-1/Pass-2
alternation begin. This gives `F`'s probes a stable error surface to score
against (the §7.3 co-adaptation risk shrinks to one moving learner at a time
early on, when it matters most).

One consistency detail for the pre-training phase: `F` exists and the loop is
wired from step one (§6.8), so pre-training runs with **`F` frozen** — its
(untrained, clamped) output drives the feedback path, but no probes run and no
`F` updates happen. Pre-training is "open-loop" only in the sense of *no
feedback training*, not *no feedback signal*: `P` pre-trains on the same
closed-loop dynamics it will see when alternation starts, avoiding a
state-distribution jump at the phase boundary. (Pre-training with `f ≡ 0`
instead would hand alternation a `P` trained on dynamics that vanish the moment
`F`'s output starts flowing.)

Convergence criteria (defaults):

- Primary: NRMSE on a held-out validation stream plateaus — the usual
  criterion, applied to the alternation phase as a whole.
- Diagnostic the scheme gives for free: **probe acceptance rate** decaying to
  the chance floor (~2/3 at zero margin, §6.1) — neither direction helps
  anymore, i.e. `F` is locally optimal at the visited states. Log from day one.
- Fixed cycle budget as a backstop.

### 6.10 F updates are applied per-accept, online

Each accepted cycle's `(Sx, f*)` pair is applied immediately as a single
`TrainOnlineStepRegression` step, then discarded. No replay buffer, no batch
cadence, no eviction policy — zero added hyperparameters, and zero staleness by
construction: every target is fresh relative to the `F` that trains on it (the
creep targets `f* = Sf ± ε` are anchored to `F`'s output at discovery time, so
a *reused* pair goes stale the moment `F` reaches it).

Accepted trade-offs, on the record:

- Single-sample updates are high-variance, and ~2/3 of accepts at zero margin
  are chance wins (§6.1) — `F` relies on cancellation across successive
  updates rather than within a batch. This is the same noise-tolerance bet
  §6.1 already made.
- `Readout::InitOnline` hardwires the **Adam** optimizer, whose running
  gradient-moment estimates are at their least useful fed single noisy samples
  at irregular intervals. If this proves to be a real problem, the fallback is
  to replumb HypercubeCNN's optimizer selection to better fit this operating
  model (e.g. plain SGD for `F`) — an accepted contingency, not a v1 task.

**Implementation seam** (same pattern as §6.7): isolate "apply accepted
pair → `F`" as one swappable step, so a replay buffer with periodic
mini-batches can drop in later if telemetry shows `F` learning too slowly or
too noisily. Accepted pairs should still be **logged** (cheap), since the
distribution of discovered `f*` values feeds the §7.4 variance telemetry —
logging is independent of training from them. The accumulate-then-batch shape
lives on regardless in §6.12's batch outer-loop variant.

### 6.11 The feedback signal is clamped at the ESN seam

The code-level catch that forced the decision: in this codebase,
`ReadoutConfig::activation` is the **per-Conv-layer** activation; the regression
output is the **raw linear-layer output** (`PredictRaw`: "Regression: raw
network output"). Specifying TANH does *not* produce an `F` whose output lives
in [−1, 1]. Left alone, the injected feedback is unbounded — and an unbounded
`f` multiplied through the feedback weight block can saturate every neuron's
`tanh` (information blackout) or dominate the input drive. The reservoir-side
`tanh` caps the *effect* per neuron but not the saturation regime.

The ESN therefore clamps the feedback before injection — `f = tanh(F(x))` —
at the seam where it routes `F`'s prediction into `InjectFeedback`.
Consequences:

- `feedback_scaling` alone sets the worst-case drive; stability analysis gets a
  hard bound (`|f| < 1`) to lean on.
- The untrained `F`'s arbitrary early output (§6.8) is tamed for free.
- **Perturb and train in the *pre-clamp* space consistently:** `Sf` is the raw
  `F(Sx)`, the probes inject `tanh(Sf ± ε)`, and `F`'s regression target is the
  raw `f* = Sf ± ε`. The clamp is part of the *injection* path, not the
  learning path — `F` never has to invert it.
- *Side effect to watch:* if the raw `|F(x)|` drifts deep into tanh saturation,
  `tanh(Sf ± ε) ≈ tanh(Sf)` and the probes lose their lever — accepts stall not
  because `F` is optimal but because the clamp has flattened the
  perturbation. Add raw `|F(x)|` magnitude to the telemetry set; a little
  weight decay on `F` is the cheap counter if it occurs.

### 6.12 Scope: streaming mode only

Feedback training is offered **only in streaming mode** in v1; batch mode stays
open-loop.

The ESN trains in two modes today, and the Pass-1/Pass-2 cycle does not fit
them equally. In **streaming** mode (`InitOnline` → per-step `TrainLive*`) the
cycle maps directly: each arriving example is probed (Pass 2) and then
committed with online `P` training (Pass 1), using snapshot/restore around the
probes — §4 describes this mode essentially verbatim. **Batch** mode
(`Warmup` → `Run` collect → `Train` over many epochs) is a fundamental
mismatch, for three reasons:

1. The probes require **re-executing the reservoir** from a snapshot under
   perturbed feedback. A pre-collected state matrix is dead data — there is
   nothing to perturb.
2. With the loop closed, collected states **depend on `F`**: states collected
   under one policy go stale as soon as `F` trains, so `P`'s offline epochs
   train on a distribution that no longer matches deployment.
3. Per-example alternation has no analogue in "collect once, train 600 epochs."

The batch analogue, recorded for the future (to be built only if streaming
results justify it) — **outer-loop alternation (EM-flavored)**: freeze `F`;
drive the full training sequence closed-loop, collect states; batch-train `P`
with the mature multi-epoch path. Then freeze `P`; re-drive the sequence
interactively, running the Pass-2 probe at each step and accumulating the
accepted `(state, f*)` pairs; batch-train `F` on the accumulated set. Repeat
the outer loop until validation NRMSE plateaus. Each learner always trains
against a frozen partner — which also neutralizes most of the §7.3
co-adaptation risk — and `P` keeps the batch path the existing benchmarks rely
on. Cost: ~4 full-sequence drives per outer iteration (1 collect + 3 probe
passes). It trades the per-example alternation for phase-level alternation,
its natural batch form. (Bonus: accumulating all pairs before training `F`
subsumes the §6.10 replay-buffer question in this mode.)

Consequence of the streaming-only scope: the natural showcase benchmarks
(NARMA-30, sine; §9) currently run on the **batch** harness, so the A/B
evaluation needs either a streaming variant of those benchmarks (the readout
already supports online regression) or its first evidence from the streaming
examples instead.

### 6.13 The `f ≡ 0` kill-switch: a config arm and a runtime lesion flag

§7.6 mandates a kill-switch comparison in every experiment, and §9.1 requires
"feedback configured but `f ≡ 0` forced." The Reservoir-level
`enable_feedback_` toggle is deliberately gone (`cde5d2f`) and **stays gone**:
at that layer `num_feedback_channels > 0` is the single source of truth for
the path's existence, and a second boolean that silently deadens a configured
path is exactly the split-brain the removal fixed. The force-zero semantics
the verification plan needs — realization, weights, and loop machinery intact,
only the *value* dead — belongs where the value is computed, and is provided
by two complementary mechanisms at two layers:

```
              kill point 2 — runtime: force_zero_feedback (ESN seam)
                        │
  x ──F──► raw ──tanh──► f ──InjectFeedback──► f × W_fb ──► Σ (per vertex)
                                                    │
              kill point 1 — config: feedback_scaling = 0  ⇒  W_fb ≡ 0
```

**Config arm — `feedback_scaling = 0` (no new code; it already works).**
Feedback weights are drawn from the RNG and *then* scaled (`Reservoir.cpp`,
`Initialize`), so scaling zero with `num_feedback_channels > 0` consumes
identical RNG draws and yields a **bit-identical realization** with the
feedback block exactly zeroed — input and recurrent weights untouched, and the
SR normalization unaffected (it excludes the feedback block). Every injected
`f`, whatever its value, then contributes exactly 0.0 to every pre-activation:
literally "configured but `f ≡ 0` forced." The entire training machinery still
executes — snapshots, three probes, `F` forwards — but the post-step probe
states are bit-identical, so `E+ = E− = E0` *exactly*, the strict-`<` accept
of §4 step f never fires, and `F` never trains. The control arm therefore
costs the same ~4× overhead as the live arm with the signal provably dead —
the §9.4 A/B comparison gets a compute-matched control for free. Limitation:
build-time only; it cannot be flipped on an already-trained system.

**Runtime arm — ESN-level `force_zero_feedback` at the §6.11 clamp seam.**
One line at the seam: inject `0` instead of `tanh(F(x))`. (Injecting 0 and
skipping `InjectFeedback` are mathematically identical — the staging buffer is
cleared by every `Step` — but injecting keeps the executed code path closest
to the live arm.) This serves the ablation the config arm cannot: train
normally with feedback live, then flip the flag at evaluation to measure how
much of the trained system's performance the closed loop actually carries —
§7.6's per-experiment lesion comparison in its most informative form. If the
flag is on *during* training, every probe injects the same 0, the
exact-equality argument above applies, and `F` freezes automatically — no
separate "disable Pass 2" machinery is needed, only this documented
consequence. The flag lives in the ESN-side feedback configuration block
(§6.14: `ESNConfig::feedback.force_zero`), defaults off, and is settable
between training and evaluation.

This is not the removed toggle returning. That toggle duplicated the
Reservoir's own existence-gating one layer down; this flag overrides a value
at the single layer that computes it, and the Reservoir's
`num_feedback_channels`-gates-everything invariant stays intact.

### 6.14 Config surface and defaults

The ESN-side configuration is one nested block, `ESNConfig::feedback`,
holding `F`'s `ReadoutConfig` plus the scheme's own knobs and the §6.13 flag:

```cpp
struct FeedbackConfig {
    ReadoutConfig readout;        // F (dim / num_outputs / task forced by ESN)
    float epsilon    = 0.05f;     // probe perturbation, pre-clamp space (§6.11)
    float margin     = 0.0f;      // accept margin (§6.6)
    float lr         = 2e-4f;     // F's constant Adam learning rate
    bool  force_zero = false;     // runtime lesion flag (§6.13)
};
```

Enablement stays where it already lives: `reservoir.num_feedback_channels > 0`
(v1: the ESN ctor validates `== 1` when feedback training is used) — no second
on/off flag, the same single-source-of-truth discipline as §6.13. The ESN
forces `dim` (the shared subsample geometry, exactly as `MakeReadoutConfig`
already does for `P`), `num_outputs = 1`, and `task = Regression` on `F`'s
config; the batch-path fields (`epochs`, `batch_size`,
`lr_max`/`lr_min_frac`/`lr_decay_epochs`, `momentum`) are ignored for `F` in
streaming v1 and documented as such. `F` reuses the full `ReadoutConfig`
rather than a slimmed struct so checkpoint serialization stays uniform — the
two-readout `GetReadoutState` extension will want that.

Defaults, with the reasoning on record:

| Knob | Default | Rationale |
|---|---|---|
| `feedback.readout.num_layers` | 1 | §7.7's hint, promoted to spec. |
| `feedback.readout.conv_channels` | 8 | Half of `P`'s 16. `F` learns one scalar from sparse, autocorrelated samples (one per accepted cycle); small capacity *is* the regularizer against §7.7's memorization risk. |
| `feedback.readout.activation` | `TANH` | Matches the §2 table. Per-Conv-layer only — it does **not** bound `F`'s output (§6.11). |
| `feedback.readout.seed` | 43 | No must-differ constraint: `F` and `P` have different architectures, so identical seeds are harmless. Defaulting off `P`'s 42 avoids accidental twin-init in degenerate same-shape configs. Determinism is the only requirement. |
| `feedback.readout.weight_decay` | 0 | Telemetry-gated (§6.11, §7): promote to ~1e-4 only if the raw `\|F(x)\|` saturation watch fires. |
| `feedback.epsilon` | 0.05 (sane range 0.01–0.2) | Rationale below. |
| `feedback.lr` | 2e-4, constant | Rationale below. |
| `feedback.margin` | 0 | Already specced (§6.6). |
| `reservoir.feedback_scaling` | 0.5 (unchanged) | Matches `input_scaling`'s 0.5, so at `\|f\| ~ \|u\|` the feedback drive is commensurate with the input drive — a sane first operating point. Drop to 0.25 if the §7.6 state-norm monitor shows growth; do **not** pre-shrink it, or a null result is uninterpretable (signal too weak vs. idea wrong). |

**Why ε = 0.05 — what ε actually controls.** ε is *not* a noise-immunity
knob: the probes are deterministic same-state, same-example replays (§6.3),
so even a tiny ε yields an exact error ordering within a cycle. What ε sets
is (a) the **creep rate** — `F`'s target moves at most ε per accepted cycle —
and (b) **locality** — how far the probe linearization and the §6.11 clamp
curvature can be trusted. Sized as a fraction of the usable range: post-clamp
the signal lives in (−1, 1), and near the origin ε = 0.05 is a ~5% step —
roughly 20 net-directional accepts to traverse half the usable range, fast
enough to observe learning, slow enough for §7.3's timescale separation. The
floor of the sane range (0.01) is a float32 consideration: probe deltas must
stay above rounding noise in `P`'s forward pass. The ceiling (0.2) is where
the D-series lever mismatch (post-clamp lever `≈ ε·(1 − tanh²(Sf))` vs.
constant pre-clamp creep) and the clamp curvature stop being second-order.
Fixed in v1; adaptive ε stays on the §6.7 upgrade list.

**Why lr = 2e-4, constant — and the §6.2 interaction.** Constant because the
stream is open-ended: `CosineLR` needs a horizon that does not exist. (This
resolves `F`'s half of the streaming-lr question; `P`'s streaming lr policy
is a separate open item.) Magnitude: ~7× below `P`'s `lr_max = 0.0015`,
implementing §7.3's "make `F`'s effective timescale much slower than `P`'s."
The lr also controls what fraction of the ε step the commit realizes (§6.2
commit-time corollary): `F′(Sx)` lands lr-dependently between `Sf` and `f*`.
Too high, and one Adam step reaches the target — the commit degenerates into
committing `f*`, forfeiting the averaging that the §6.6 chance-accept
tolerance relies on; too low, and accepted improvements barely register. The
tuning signal is the **step-realization fraction** `|F′(Sx) − Sf| / ε`
(telemetry, §8) — tune lr toward ~0.1–0.3. Adam's moment estimates on single
samples at irregular intervals (§6.10's known caveat) make the realized
fraction unpredictable a priori — telemetry, not faith.

Both ε and lr are educated guesses with stated reasoning, not measured
values — nothing has run yet. They are defaults-with-tuning-signals (accept
rate, creep rate, realization fraction), not blessed constants.

---

## 7. Critique of the concept itself

Stated bluntly. None of these are fatal; several bound what should be expected
from v1.

### 7.1 Sample efficiency is poor by construction

Each cycle costs ~3 probe evaluations (baseline + two perturbations) plus the
re-presented Pass-1 example, and yields **at most one bit** of training signal
(direction of improvement at one state) embodied in one ε-sized regression
target. A backprop-based scheme would extract a full gradient per sample;
genuine SPSA would at least perturb *in parameter space* and update all of `F`'s
weights per probe pair. This scheme spends reservoir evaluations to discover a
1-D signal-space direction, then spends a regression step transferring it into
`F`. It is the price of having no teacher and no gradient through the reservoir
(§4.1) — but be clear-eyed: training `F` will be slow, and most of the wall-clock
will be probe evaluations. Mitigations exist on the §6.7 upgrade list (the
verified line search amortizes probes into bigger tested steps) and in
§6.10 (a replay buffer reuses what was paid for).

### 7.2 The probes are myopic — and the reservoir's memory makes that worse

A feedback perturbation at `Sx` affects the reservoir for far longer than the
single probed step: with `spectral_radius ≈ 0.92–0.99` and an M-deep delay
line, injected signal echoes persist for tens of steps. With `H = 1`
(§6.1) the probe credits only the immediate next-step error — the direct
causal path is captured (feedback enters the very next state update, which `P`
reads), but every downstream echo goes uncredited. A perturbation that helps
the next step yet poisons the state 50 steps later scores as a win; the
converse scores as a loss. The greedy criterion optimizes a proxy, and nothing
in the scheme detects when the proxy diverges from the true objective. The
`H`-window fallback would widen the credited horizon, but buys it with the
constant-`Sf` probe approximation — the two errors trade against each other
and no `H` fixes both. This is the scheme's most fundamental limitation; v1
accepts it and lets the A/B benchmark (§9) say whether next-step credit is
good enough.

### 7.3 Two learners co-adapting on shifting ground

`P` trains against the state distribution induced by the current `F`; `F`'s
probes are scored by the current `P`. Both move every cycle. This is the classic
actor-critic instability shape, in miniature. Risks: oscillation (each pass
undoes the other's assumptions), and `F` chasing improvements that exist only
relative to a half-trained `P` (early in training, `E0` is dominated by `P`'s
own error, and probe wins may reflect noise in `P`'s error surface rather than
genuinely better reservoir states). Mitigations: pre-train `P` to
near-convergence before starting Pass-2 cycles (§6.9); make `F`'s
effective timescale much slower than `P`'s (small ε, accept margin); monitor
validation NRMSE for oscillation.

### 7.4 The discovered signal may be a glorified bias

The honest null hypothesis: after training, `F` outputs a near-constant value —
a learned global bias drive — because the per-state signal is too noisy to learn
and a constant is the noise-floor solution of the regression. That outcome isn't
a blow-up (a small constant drive is harmless and occasionally even useful), but
it would mean the entire probe machinery bought nothing a bias term couldn't.
**Telemetry to detect it:** track the variance of `F(x)` across states; if it
collapses, the state-conditional claim is unsupported. This is the cheapest
falsification test of the whole idea and should be in the first experiment.

### 7.5 What it plausibly *can* do

For balance: trained state feedback through the input pathway has published
evidence of real gains (Ehlers et al. 2025; the full differentiation from this
design is in §2.3). Mechanistically, a
state-dependent scalar drive can modulate the reservoir's effective operating
point — pushing it toward/away from saturation depending on context, a knob the
open-loop ESN simply does not have. Tasks where error correlates with the
reservoir's excitation regime (long-memory NARMA variants, regime-switching
signals) are the plausible win cases. Pure short-memory tasks likely show
nothing — choose benchmarks accordingly.

### 7.6 Stability is unguarded

Restating from `feedback_mechanisms.md` with the training scheme in view: the
echo-state property is an open-loop guarantee, and the feedback weight block sits
**outside** `EstimateSpectralRadius`. Closing the loop with a *trained, state-
dependent* signal voids the configured SR semantics. The guards available:
small `feedback_scaling`, bounded `f` (§6.11), and the per-neuron `tanh`.
None of them is a proof; watch for state-norm growth during closed-loop runs and
keep a kill-switch comparison (`f ≡ 0`) in every experiment — both mechanisms
are specified in §6.13 (config arm: `feedback_scaling = 0`; runtime lesion
flag: `force_zero_feedback`).

### 7.7 Smaller observations

- The HCNN forward is the cost driver of a probe. With 3 probes + 1 commit, a
  cycle is ~4× the per-timestep cost of plain streaming training (§5). Budget
  accordingly.
- The scheme trains `F` only at committed-trajectory states (one per cycle).
  `F` generalizes from sparse, temporally correlated samples; the CNN may
  memorize. Capacity of `F` should be small — specced as defaults in §6.14
  (`num_layers = 1`, `conv_channels = 8`).
- Snapshot/restore must be exact (bit-for-bit), or probe comparisons inherit a
  systematic offset. Restoring must also re-home the slice-ring rotation, not
  just the buffer contents. (Both properties hold and are tested — see §8.)

---

## 8. New capability required (implementation tracker)

| Capability | Where | Status | Notes |
|---|---|---|---|
| State snapshot/restore | `Reservoir` | **DONE** (`696d762`) | `TakeSnapshot`/`RestoreSnapshot`: canonical (rotation-free) capture of `vtx_state_` + all M history slices; restore re-homes the ring and clears staged drives. Bit-exact — verified by the §9.2 diagnostics in `main.cpp`. |
| Second readout instance | `ESN` | pending | `F` constructed from a `feedback ReadoutConfig` (Regression, 1 output) sharing the subsample geometry. |
| Closed-loop stepping | `ESN` | pending | A step driver that evaluates `F`, applies the §6.11 clamp, calls `InjectFeedback`, then `Step`. Includes the `force_zero_feedback` runtime override at the clamp seam (§6.13). |
| Training orchestration | `ESN` | pending | The Pass-1/Pass-2 cycle of §4 (streaming mode — §6.12), with hyperparameters: `ε`, accept margin (default 0 — §6.6), `F` learning rate, schedule (§6.9). `H` is fixed at 1 (§6.1). Config surface and defaults — §6.14 (`ESNConfig::feedback`). |
| Telemetry | `ESN` | pending | Probe acceptance rate, `E0/E+/E−` traces, variance of `F(x)` (§7.4), raw `|F(x)|` magnitude (§6.11 saturation watch), state-norm monitor (§7.6), step-realization fraction `|F′(Sx) − Sf|/ε` (§6.14 lr tuning). |

## 9. Verification plan (when implemented)

1. **No-op regression:** `num_feedback_channels = 0` paths byte-identical to
   `main`; with feedback configured but `f ≡ 0` forced — each §6.13 mechanism
   in turn: `feedback_scaling = 0`, and the `force_zero_feedback` flag —
   results match open-loop.
2. **Snapshot fidelity:** snapshot → N steps → restore → N steps reproduces the
   identical trajectory bit-for-bit. *(Implemented and passing — `main.cpp`
   diagnostics: restore+replay memcmp-identical, Take→Restore→Take identity,
   staged-injection isolation, size-mismatch throw; three configs incl.
   2-channel feedback, Release build.)*
3. **Probe sanity:** with `ε = 0`, all three probes return the identical
   squared error (this also exercises snapshot fidelity through the full probe
   path).
4. **A/B benchmark:** NARMA-30 and sine prediction, feedback-trained vs.
   open-loop, multiple seeds — the same harness as the existing M-sweep. Report
   NRMSE deltas *with* the §7.4 variance telemetry, so a win can be attributed
   (state-dependent signal vs. learned bias). The `feedback_scaling = 0` arm
   of §6.13 doubles as the compute-matched control (same ~4× probe overhead,
   signal provably dead). These benchmarks run on the
   batch harness today; with v1 streaming-only (§6.12), they need streaming
   variants — the readout already supports online regression.
5. **Stability soak:** long closed-loop free runs; assert bounded state norms.
