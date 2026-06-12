# Feedback Training Methodology

> Status: **design finalized — ready for implementation planning** (June 2026).
> This document articulates the training scheme for integrating the Reservoir's
> feedback path into the `ESN` class. All design decisions ([OPEN-1]–[OPEN-12],
> §6) are **resolved**; each entry records the decision, rationale, and where
> applicable the rejected alternative and the telemetry that would trigger a
> rethink. Conceptual criticism of the scheme is in §7. Implementation is in
> progress — see the §8 tracker (Reservoir-level mechanics §1 and
> snapshot/restore are done; the ESN-side work is pending).

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
    InjectFeedback(0, f(t))      where f(t) = tanh(F(x(t-1)))   // previous state, clamped ([OPEN-11])
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
job is to reduce primary task error). The nearest prior art is Ehlers, Nurdin &
Soh (2025) — trained state feedback through the input pathway — which reported
real performance gains; see `feedback_mechanisms.md` §Related-work. The novelty
here is **how `F` is trained**: not by closed-form optimization, but by local
perturbation search (§4).

## 3. Definitions

- **Example** — one training example is a **single timestep**: one input sample
  `u(t)` with its target `y(t)` (`H = 1`; **[OPEN-1]**, resolved). The
  generalization to `H`-step windows is recorded in [OPEN-1] as the data-driven
  fallback, to be adopted only if single-step probes demonstrably fail.
- **`Sx`** — a full reservoir state snapshot: `vtx_state_`, **all `M` history
  slices**, and the slice-ring rotation. (The per-step `vtx_input_` /
  `vtx_feedback_` staging buffers are always clear between steps and need not be
  captured.) Requires a new `Reservoir` snapshot/restore capability (§8).
- **`Sf`** — the scalar feedback operating point at the start of a Pass-2 cycle:
  `Sf = F(Sx)`, the feedback readout's own output at the decision-time state.
  ("Preserve the current feedback signal" means: evaluate `F(Sx)` once and reuse
  that value across the cycle's probes.) See **[OPEN-2]** (resolved) for the
  rejected alternative reading.
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
   from cycle 0 this is `F`'s live output, [OPEN-8]), then `Step`.
2. Train `P` on the resulting (state, target) pair exactly as today
   (`TrainLiveStepRegression`, or mini-batch accumulation across cycles).
3. `F` is not touched. The reservoir state advances for real (no restore).

### Pass 2 — feedback readout training (primary held fixed)

Let `Sx` = snapshot of the reservoir state after Pass 1, and `Sf = F(Sx)` = the
current feedback operating point ([OPEN-2], resolved). Let `(u', y')` = the next
training example.

| Step | Action |
|---|---|
| a | Snapshot `Sx`; evaluate `Sf = F(Sx)` once. |
| b | **Baseline probe:** inject `u'` and feedback `Sf`; `Step`; forward-pass `P`; **no training**. |
| c | `E0` = squared error of `P`'s prediction against `y'`. |
| d | Restore `Sx`. **Positive probe:** same input `u'` with feedback `Sf + ε`; `E+`. |
| e | Restore `Sx`. **Negative probe:** same input with `Sf − ε`; `E−`. *(The spec probes `−ε` only if `+ε` fails; §6 [OPEN-6] argues for always probing both.)* |
| f | **Accept/reject:** if `min(E+, E−) < E0 − margin`, the winning perturbation direction is a locally verified improvement. Set `f* = Sf ± ε` (winning sign). |
| g | **Train `F` (one regression step):** input = subsampled features of **`Sx`** (the decision-time state, *not* any post-step probe state — see [OPEN-4]), target = `f*`. |
| h | If neither probe beats baseline by the margin, train nothing this cycle. |
| i | Restore `Sx`. Return to Pass 1, which consumes `(u', y')` *for real* (see [OPEN-5] — the probe example is re-presented as the next Pass-1 example, preserving stream continuity). |

Per [OPEN-11], every injected feedback value passes through the `tanh` clamp at
the seam; `Sf`, `±ε`, and the training target `f*` all live in the pre-clamp
space (the table above writes the pre-clamp values).

Repeat cycles until the convergence criteria ([OPEN-9]) are met. This is the
**streaming-mode** formulation — it is inherently interactive (the probes
re-execute the reservoir from a snapshot). Batch-mode integration is a
structurally different question: see **[OPEN-12]**.

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
    snapshot S1, Sf = F(S1)                       ([OPEN-2], resolved: on-policy)
    probe E0:  S1 --inject u_k+1, f=Sf----> Step, P forward: e = 0.0412
    probe E+:  S1 --inject u_k+1, f=Sf+ε--> Step, P forward: e = 0.0398   <- improvement
    probe E-:  S1 --inject u_k+1, f=Sf-ε--> Step, P forward: e = 0.0421
    accept +ε:  train F one step:  F(subsampled S1) → Sf + ε
    restore S1
PASS 1 on example k+1:  (re-presented, committed for real)                    → state S2
PASS 2 on example k+2:  ...
```

Each example is consumed twice: once as a Pass-2 probe target, once as a Pass-1
training example. Reservoir state continuity is never broken. A full cycle
costs 4 reservoir steps and 4 `P`-forwards (3 probes + 1 commit) plus one
`P`-train and, on accept, one `F`-train — a ~4× per-timestep overhead versus
plain streaming training.

---

## 6. Design decisions (all resolved)

Originally the open-issues list; all twelve are now resolved. Each entry keeps
its full reasoning — decision, rationale, rejected alternatives, and the
telemetry that would trigger a data-driven rethink.

### [OPEN-1] What is an "example"? — **RESOLVED: one timestep (`H = 1`) for v1**

Decision: keep the mechanism as simple as possible at the onset — an example is
a single timestep, the probe error is a single squared error, and there is no
window hyperparameter to tune. Any move away from `H = 1` must be **data-driven**
(triggered by the telemetry below), not by untested hypothesis.

Why `H = 1` is more defensible than the noise concern suggests:

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
- *Fallback, recorded but NOT to be built yet:* `H`-step probe windows (8–32),
  MSE over the window, feedback held constant during probes, `F` trained on the
  window's first decision state only. Adopt only if validation NRMSE shows no
  gain AND the telemetry above is flat.

### [OPEN-2] Where does `Sf` come from? — **RESOLVED: `Sf = F(Sx)` (on-policy)**

The feedback signal is, by definition, the single-channel regression output of
the feedback readout — at training time as well as deployment. So the Pass-2
operating point is the current policy's own output at the decision state,
`Sf = F(Sx)`; the probe explores ±ε *around the policy*, and an accepted probe
nudges the policy output at that state in the winning direction. This is a 1-D
policy-improvement step (ES/SPSA-flavored), the training data is generated under
the same policy that runs at deployment, and `Sf` needs no separate bookkeeping —
it is recomputable from `Sx`. The spec's "preserve the current feedback signal"
means: evaluate `F(Sx)` once and reuse it across the three probes.

*Rejected reading, recorded for posterity:* `Sf` as a free-standing carried
scalar that the accepted perturbations random-walk, with `F` distilled from the
walk and taking over only at deployment. Rejected because a slowly-walking
global scalar paired with fast-varying states teaches `F` almost no
state-conditional structure, and training-time feedback (the scalar) would
mismatch deployment-time feedback (`F(x)`).

*Residual nuance:* during the probe windows themselves the injected feedback is
briefly **not** `F`'s live output — it is the frozen scalar `F(Sx) ± ε` held
constant across the window. That deviation is inherent to probing and is one of
the reasons `H` must stay modest ([OPEN-1]).

### [OPEN-3] Probe example — **RESOLVED: all probes replay the *same* example**

All probes within one Pass-2 cycle replay the identical example from the
identical restored state, so the perturbation is the only varying factor. (The
spec's literal wording — "present a fresh training example" at each repeat —
read as fresh examples per probe; that reading was considered and dropped.)

Rationale, briefly: with different examples per probe, `E+ − E0` mixes the
ε-sized perturbation effect with the O(1) difficulty difference between
examples, and on autocorrelated streams that difficulty delta can correlate
with `Sx` — a state-dependent bias that would train `F` on stream artifacts
rather than feedback quality. Same-example replay is the paired-comparison /
common-random-numbers form of the finite difference, and it preserves the
noise-symmetry property the `H = 1` decision ([OPEN-1]) leans on. It is also
the simpler variant operationally: one example serves all three probes plus the
Pass-1 commit, and the [OPEN-5] stream continuity stays clean.

(Honest caveat from review: the practical difference may well be negligible —
but one variant had to be picked, and this one costs nothing extra.)

### [OPEN-4] Which state is `F` trained on? — **RESOLVED: `Sx`, the decision-time state**

`F` trains on the subsampled features of **`Sx`** (the pre-step snapshot),
target `f*`. Not on any post-step probe state.

Why — the role split at deployment: `F` always reads the state on the *left*
side of a `Step` and its output is consumed *by* that `Step`; `P` always reads
the state on the right side.

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

Training on the winning post-step state `A+` instead (the other reading of the
spec's "run that Reservoir output through the feedback Readout") breaks twice:
(1) it answers a question nobody asked — at deployment, `F`'s query at `A+`
concerns the *next* step's feedback, which no probe tested, while the decision
that *was* tested gets no correction; (2) `A+` already carries the imprint of
`Sf+ε`, so `F` would learn to recognize states that already received good
feedback rather than states that need it. The post-step states `A0/A+/A−`
are outcome measurements only, and are discarded by the restore.

(Wrinkle that applies only to the `H > 1` fallback of [OPEN-1]: the constant
`f*` would be applied at *all* H steps of the window, but only the first step's
decision state is `Sx`. Training only the `(Sx, f*)` pair is the conservative
choice. Moot under the resolved `H = 1`.)

### [OPEN-5] Stream continuity — **RESOLVED: Pass 1 re-presents the probe example**

Spec step (j) restores `Sx` and "starts over with Pass 1" on a fresh example. If
Pass 1 then consumed the example *after* the probed one, the committed reservoir
trajectory would skip the probed example entirely — the state would be wrong for
everything downstream (the reservoir never experienced it), and `P` would train
on state/input alignments that never occur at deployment.

```
stream:  ... | u_k+1 | u_k+2 | ...

SKIP after probe (broken):                RE-PRESENT (§4i, gapless):

S1 ··probe u_k+1 ×3·· restore S1          S1 ··probe u_k+1 ×3·· restore S1
S1 ──commit u_k+2──► S2                   S1 ──commit u_k+1──► S2
     ✗ S1 never experienced u_k+1;        S2 ··probe u_k+2 ×3·· restore S2
       S2 and everything after it is      S2 ──commit u_k+2──► S3
       misaligned with the stream              ✓ every example: probed, then
                                                 committed; trajectory gapless
```

**Resolution (as written in §4i):** Pass 1 re-presents the probe example as its
real training example. Every example is used twice (probe, then commit), the
committed trajectory is gapless, and the data cost is zero extra examples.

### [OPEN-6] Probe order — **RESOLVED: always probe both directions; margin default 0**

The spec probed `+ε` and accepted immediately if it beat baseline, testing `−ε`
only on failure. Decision: probe **both** directions every cycle and take the
better (central-difference flavored). It costs at most one extra evaluation per
cycle and removes a systematic positive-direction bias from `F`'s targets.

On accepting *any* improvement: with `H = 1` ([OPEN-1]) single-sample errors
differ by luck constantly — at zero margin the pure-chance accept rate is ~2/3.
Under [OPEN-1]'s noise-tolerant stance that is acceptable: chance accepts
produce zero-mean target jitter that `F`'s regression averages away, while real
effects drift. A **margin** is therefore a purity/throughput knob, not a
correctness requirement — it trades fewer chance-trained cycles against a lower
accept rate on real effects.

**Decision:** always probe both directions; margin configurable with default 0,
promoted to a small relative value only if telemetry shows `F` training
dominated by jitter.

### [OPEN-7] Hunt strategy — **RESOLVED: plain ±ε creep for v1, knowingly naive**

An accepted cycle trains `F` toward `f* = Sf ± ε` — a target that differs from
its current output by *at most ε*. Even with `lr` high enough to fully reach the
target, the policy moves ε per accepted cycle. If the useful feedback magnitude
is O(1) and ε = 0.01, that is hundreds of accepted cycles minimum, before
counting rejections and regression under-shoot.

**Decision:** v1 uses the plain ±ε probe anyway. On the record: this is a
deliberately naive hunt strategy, chosen with full awareness of how simple and
limited a fixed-step two-point probe is — not an oversight. Escalation is
deferred until data shows the creep is the bottleneck.

**Implementation seam:** structure the Pass-2 code so "discover `f*` from
(`Sx`, `u'`, `Sf`)" is one isolated, swappable step. Candidate upgrades when
the time comes, in rough order of sophistication: verified line search after a
win (re-probe `Sf ± 2ε, ±4ε…` while improvement continues — each probe cheap,
target *tested* not extrapolated); amplified targets `f* = Sf ± λε`
(extrapolated, riskier); adaptive ε (shrink as the accept rate falls);
multi-point or stochastic probe patterns. None of these change the rest of the
methodology — they only change how far `f*` lands from `Sf`.

### [OPEN-8] Bootstrap — **RESOLVED by [OPEN-2]: `F(x)` drives from cycle 0**

Since the feedback signal is by definition `F`'s output, there is no separate
bootstrap value: at cycle 0 the randomly initialized `F` drives the loop with an
arbitrary (and unbounded — see [OPEN-11]) output. This is acceptable — the
random early output acts like noise injection, which the output-feedback ESN
literature actually *recommends* during training, and there is no
train/deploy distribution switch. Keep `feedback_scaling` small enough that
random early feedback cannot destabilize the reservoir (its per-neuron effect is
bounded by the `tanh` in `UpdateState` regardless), and note that the [OPEN-11]
clamp also tames the untrained output's magnitude.

### [OPEN-9] Schedule & convergence — **RESOLVED: pre-train `P` first, then alternate**

**Schedule (decided):** train `P` alone to near-convergence first, *then* begin
the Pass-1/Pass-2 alternation. This gives `F`'s probes a stable error surface
to score against (the §7.3 co-adaptation risk shrinks to one moving learner at
a time early on, when it matters most).

One consistency detail for the pre-training phase: `F` exists and the loop is
wired from step one ([OPEN-8]), so pre-training runs with **`F` frozen** — its
(untrained, clamped) output drives the feedback path, but no probes run and no
`F` updates happen. "Pre-train open-loop" thus means *no feedback training*,
not *no feedback signal*: `P` pre-trains on the same closed-loop dynamics it
will see when alternation starts, avoiding a state-distribution jump at the
phase boundary. (Pre-training with `f ≡ 0` instead would hand alternation a
`P` trained on dynamics that vanish the moment `F`'s output starts flowing.)

**Convergence criteria (defaults):**
- Primary: NRMSE on a held-out validation stream plateaus — the usual criterion,
  applied to the alternation phase as a whole.
- Diagnostic the scheme gives for free: **probe acceptance rate** decaying to
  the chance floor (~2/3 at zero margin, [OPEN-1]) — neither direction helps
  anymore, i.e. `F` is locally optimal at the visited states. Log from day one.
- Fixed cycle budget as a backstop.

### [OPEN-10] How `F`'s regression step is applied — **RESOLVED: per-accept online**

**Decision:** each accepted cycle's `(Sx, f*)` pair is applied immediately as a
single `TrainOnlineStepRegression` step, then discarded. No replay buffer, no
batch cadence, no eviction policy — zero added hyperparameters, and zero
staleness by construction: every target is fresh relative to the `F` that
trains on it (the creep targets `f* = Sf ± ε` are anchored to `F`'s output at
discovery time, so a *reused* pair goes stale the moment `F` reaches it).

Accepted trade-offs, on the record:
- Single-sample updates are high-variance, and ~2/3 of accepts at zero margin
  are chance wins ([OPEN-1]) — `F` relies on cancellation across successive
  updates rather than within a batch. This is the same noise-tolerance bet
  [OPEN-1] already made.
- `Readout::InitOnline` hardwires the **Adam** optimizer, whose running
  gradient-moment estimates are at their least useful fed single noisy samples
  at irregular intervals. If this proves to be a real problem, the fallback is
  to replumb HypercubeCNN's optimizer selection to better fit this operating
  model (e.g. plain SGD for `F`) — an accepted contingency, not a v1 task.

**Implementation seam** (same pattern as [OPEN-7]): isolate "apply accepted
pair → `F`" as one swappable step, so a replay buffer with periodic
mini-batches can drop in later if telemetry shows `F` learning too slowly or
too noisily. Accepted pairs should still be **logged** (cheap), since the
distribution of discovered `f*` values feeds the §7.4 variance telemetry —
logging is independent of training from them. The accumulate-then-batch shape
lives on regardless in [OPEN-12]'s batch outer-loop variant.

### [OPEN-11] Feedback signal bounding — **RESOLVED: clamp at the ESN seam**

The code-level catch that forced the decision: in this codebase,
`ReadoutConfig::activation` is the **per-Conv-layer** activation; the regression
output is the **raw linear-layer output** (`PredictRaw`: "Regression: raw
network output"). Specifying TANH does *not* produce an `F` whose output lives
in [−1, 1]. Left alone, the injected feedback is unbounded — and an unbounded
`f` multiplied through the feedback weight block can saturate every neuron's
`tanh` (information blackout) or dominate the input drive. The reservoir-side
`tanh` caps the *effect* per neuron but not the saturation regime.

**Decision:** the ESN clamps the feedback before injection —
`f = tanh(F(x))` — applied at the seam where the ESN routes `F`'s prediction
into `InjectFeedback`. Consequences:

- `feedback_scaling` alone sets the worst-case drive; stability analysis gets a
  hard bound (`|f| < 1`) to lean on.
- The untrained `F`'s arbitrary early output ([OPEN-8]) is tamed for free.
- **Perturb and train in the *pre-clamp* space consistently:** `Sf` is the raw
  `F(Sx)`, the probes inject `tanh(Sf ± ε)`, and `F`'s regression target is the
  raw `f* = Sf ± ε`. The clamp is part of the *injection* path, not the
  learning path — `F` never has to invert it.
- *Side effect to watch:* if the raw `|F(x)|` drifts deep into tanh saturation,
  `tanh(Sf ± ε) ≈ tanh(Sf)` and the probes lose their lever — accepts stall not
  because `F` is optimal but because the clamp has flattened the
  perturbation. Add raw `|F(x)|` magnitude to the telemetry set; a little
  weight decay on `F` is the cheap counter if it occurs.

### [OPEN-12] Batch vs. streaming — **RESOLVED: streaming-only for v1**

Decision: feedback training is offered **only in streaming mode** for v1; batch
mode stays open-loop. The batch outer-loop alternation (option A below) is
recorded as the future batch analogue, to be built only if streaming results
justify it.

Background — the ESN trains in two modes today, and the Pass-1/Pass-2 cycle
does not fit them equally.

**Streaming** (`InitOnline` → per-step `TrainLive*`): the cycle maps directly.
Each arriving example is probed (Pass 2) and then committed with online `P`
training (Pass 1), using snapshot/restore around the probes. §4 describes this
mode essentially verbatim.

**Batch** (`Warmup` → `Run` collect → `Train` over many epochs): fundamental
mismatch, for three reasons:

1. The probes require **re-executing the reservoir** from a snapshot under
   perturbed feedback. A pre-collected state matrix is dead data — there is
   nothing to perturb.
2. With the loop closed, collected states **depend on `F`**: states collected
   under one policy go stale as soon as `F` trains, so `P`'s offline epochs
   train on a distribution that no longer matches deployment.
3. Per-example alternation has no analogue in "collect once, train 600 epochs."

Batch options:

- **(A) Outer-loop alternation (EM-flavored).** Freeze `F`; drive the full
  training sequence closed-loop, collect states; batch-train `P` with the
  mature multi-epoch path. Then freeze `P`; re-drive the sequence
  interactively, running the Pass-2 probe at each step and accumulating the
  accepted `(state, f*)` pairs; batch-train `F` on the accumulated set. Repeat
  the outer loop until validation NRMSE plateaus. Each learner always trains
  against a frozen partner — which also neutralizes most of the §7.3
  co-adaptation risk — and `P` keeps the batch path the existing benchmarks
  rely on. Cost: ~4 full-sequence drives per outer iteration (1 collect + 3
  probe passes). It departs from the spec's per-example alternation but is its
  natural batch analogue. (Bonus: accumulating all pairs before training `F`
  subsumes the [OPEN-10] replay-buffer question in this mode.)
- **(B) Streaming-only v1.** Feedback training is offered only in streaming
  mode; batch mode stays open-loop. Simplest possible scope.

Consequence of the streaming-only decision: the natural showcase benchmarks
(NARMA-30, sine; §9) currently run on the **batch** harness, so the A/B
evaluation needs either a streaming variant of those benchmarks (the readout
already supports online regression) or its first evidence from the streaming
examples instead.

---

## 7. Critique of the concept itself

Stated bluntly, as requested. None of these are fatal; several bound what should
be expected from v1.

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
will be probe evaluations. Mitigations exist on the [OPEN-7] upgrade list (the
verified line search amortizes probes into bigger tested steps) and in
[OPEN-10] (a replay buffer reuses what was paid for).

### 7.2 The probes are myopic — and the reservoir's memory makes that worse

A feedback perturbation at `Sx` affects the reservoir for far longer than the
single probed step: with `spectral_radius ≈ 0.92–0.99` and an M-deep delay
line, injected signal echoes persist for tens of steps. With `H = 1`
([OPEN-1]) the probe credits only the immediate next-step error — the direct
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
near-convergence open-loop before starting Pass-2 cycles ([OPEN-9]); make `F`'s
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
evidence of real gains (Ehlers et al. 2025 — though theirs is a closed-form
trained linear projection, not perturbation search). Mechanistically, a
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
small `feedback_scaling`, bounded `f` ([OPEN-11]), and the per-neuron `tanh`.
None of them is a proof; watch for state-norm growth during closed-loop runs and
keep a kill-switch comparison (`f ≡ 0`) in every experiment.

### 7.7 Smaller observations

- The HCNN forward is the cost driver of a probe. With 3 probes + 1 commit, a
  cycle is ~4× the per-timestep cost of plain streaming training (§5). Budget
  accordingly.
- The scheme trains `F` only at committed-trajectory states (one per cycle).
  `F` generalizes from sparse, temporally correlated samples; the CNN may
  memorize. Capacity of `F` should be small (e.g. `num_layers = 1`, modest
  channels).
- Snapshot/restore must be exact (bit-for-bit), or probe comparisons inherit a
  systematic offset. Restoring must also re-home the slice-ring rotation, not
  just the buffer contents.

---

## 8. New capability required (implementation tracker)

| Capability | Where | Status | Notes |
|---|---|---|---|
| State snapshot/restore | `Reservoir` | **DONE** (`696d762`) | `TakeSnapshot`/`RestoreSnapshot`: canonical (rotation-free) capture of `vtx_state_` + all M history slices; restore re-homes the ring and clears staged drives. Bit-exact — verified by the §9.2 diagnostics in `main.cpp`. |
| Second readout instance | `ESN` | pending | `F` constructed from a `feedback ReadoutConfig` (Regression, 1 output) sharing the subsample geometry. |
| Closed-loop stepping | `ESN` | pending | A step driver that evaluates `F`, applies the [OPEN-11] clamp, calls `InjectFeedback`, then `Step`. |
| Training orchestration | `ESN` | pending | The Pass-1/Pass-2 cycle of §4 (streaming mode first — [OPEN-12]), with hyperparameters: `ε`, accept margin (default 0 — [OPEN-6]), `F` learning rate, schedule ([OPEN-9]). `H` is fixed at 1 ([OPEN-1]). |
| Telemetry | `ESN` | pending | Probe acceptance rate, `E0/E+/E−` traces, variance of `F(x)` (§7.4), raw `|F(x)|` magnitude ([OPEN-11] saturation watch), state-norm monitor (§7.6). |

## 9. Verification plan (when implemented)

1. **No-op regression:** `num_feedback_channels = 0` paths byte-identical to
   `main`; with feedback configured but `f ≡ 0` forced, results match open-loop.
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
   (state-dependent signal vs. learned bias). These benchmarks run on the
   batch harness today; with v1 streaming-only ([OPEN-12], resolved), they need
   streaming variants — the readout already supports online regression.
5. **Stability soak:** long closed-loop free runs; assert bounded state norms.
