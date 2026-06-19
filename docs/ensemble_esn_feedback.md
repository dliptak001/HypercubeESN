# Ensemble ESN — Design

> Status: **design (finalizing)**. Task-agnostic by intent: this specifies the
> general `EnsembleESN` capability. Demonstration examples are deliberately **out of
> scope** and will be built *after* the capability lands. No example drives this
> design.
>
> The mechanism builds on the single-reservoir feedback driver path documented in
> [reservoir_feedback_mechanism.md](reservoir_feedback_mechanism.md).

---

## 1. Scope — feedback-only, online-only (read this first)

`EnsembleESN` exists for exactly one purpose: **consensus feedback coupling of M
ESNs**. The design is built entirely around that mechanism, and the following
boundaries are binding:

- **Feedback is the whole point.** `EnsembleESN` is not a general ensemble container.
  Its members are coupled at runtime through the feedback driver path; that coupling
  is the reason the class exists.

- **Online-only.** The coupling requires live member outputs every step *and* a
  step-locked joint evolution of all members. Batch training provides neither (its
  readout is fit only after open-loop, teacher-forced state collection — there are no
  outputs during collection and no member interaction). So `EnsembleESN` runs in
  online mode only. **Batch mode is not a consideration and does not constrain this
  design.**

- **No feedback-less / averaging-only case.** A developer who wants a plain
  output-averaging ensemble (no coupling) can assemble one trivially from independent
  `ESN` instances — it needs nothing from this class. `EnsembleESN` therefore carries
  **no config flag, no code path, and no special case** for the no-feedback ensemble,
  and that use case is **not permitted to drive any design decision here.**

- **The mechanism is always engaged.** There is no "feedback on/off" switch. From the
  first online step the consensus is computed and injected; only its **intensity**
  varies over time (§4). A zero-intensity operating point exists (it falls out of the
  ramp's low-intensity start) and is used as a measurement baseline (§8), but it is a
  degenerate setting of the one mechanism, **not** a supported feedback-less
  configuration.

Everything below describes that single, feedback-centric, online machine.

---

## 2. The consensus frame — output space

The consensus is formed in **output space**, the only frame the members share. Members
differ by reservoir weight seed (and optionally more, §5); two members' N-vertex state
vectors live in incomparable coordinate systems (vertex v of member 1 has no
correspondence to vertex v of member 2). The only shared coordinates are the
D-dimensional outputs — every member's readout is trained to produce the same target
quantity. The consensus, the deviations, and the coupling drive are therefore all
defined per output channel.

The consensus statistic is **mean** (default) or **median** (§6). The consensus serves
double duty: it is both the ensemble's output and the reference each member is coupled
toward. (Combining member outputs is intrinsic to the mechanism — it is how the
coupling reference is formed — not a separate averaging feature.)

---

## 3. Mechanism

Let M = member count, D = output dimension, member i output `y_i ∈ R^D`. The coupling
is live for the entire online run — training and inference alike.

```
consensus      c   = mean_i  y_i           (per channel; or median, §6)
deviation      Δ_i = y_i − c               (Σ_i Δ_i = 0 for the mean)
coupling drive φ_i = clamp( κ · Δ_i )       (κ pulls toward consensus; clamp = guard)
```

`φ_i` (D values) is injected on member i's D feedback channels and consumed by its next
`Step`. The coupling rides the existing feedback weight block — its own weights, its own
`feedback_scaling` fan-in, and (per the reservoir contract) it sits **outside** the
spectral-radius estimate, so it does not silently inflate the open-loop stability
budget. Members are built with `num_feedback_channels = D`.

**Lockstep online step (training and inference alike):**

```
   all members at internal state x_i(t)
   ┌──────────────────────────────────────────────────────────┐
   │ 1. read outputs:   y_i(t) = readout_i( x_i(t) )           │   (PredictLiveRaw)
   │ 2. consensus:      c(t)   = mean_i y_i(t)                  │   (= ensemble output)
   │ 3. deviations:     Δ_i(t) = y_i(t) − c(t)                 │   (Σ_i Δ_i = 0)
   └───────────────────────────┬──────────────────────────────┘
                               ▼
   for each member i:
       (training only) online-update readout_i toward target(t)   (TrainLive*)
       inject task input u(t)            on the input    channels
       inject φ_i = clamp(κ(t)·Δ_i(t))   on the feedback channels
       Step  ->  x_i(t+1)                                          (the one new seam, §7)
                               │
                               ▼
                 ensemble output = c(t)
```

Standard single-step closed-loop causality: the feedback injected into the t→t+1 step
is built from outputs read at state x(t), which already exist. No delay line. The only
difference between training and inference is whether each member's readout takes an
online update this step.

---

## 4. Feedback intensity

### 4.1 The mechanism is always on; intensity is the knob

There is no "engage feedback" switch (§1). From the first online step the consensus is
computed, the deviations are formed, and `φ_i` is injected. What moves is the scalar
**feedback intensity** κ:

- **Direction (sign).** κ pulls each member *toward* the consensus — the supported
  consensus/denoise regime. Pushing members apart (repulsion) is unstable as a runtime
  signal and is out of scope (§9).
- **Magnitude `|κ|`.** The "intensity" the schedule (§4.2) ramps. At `|κ| = 0` the
  mechanism is engaged but injects nothing; as `|κ|` rises members are pulled harder
  toward the consensus.
- **Over-coupling.** Too large `|κ|` over-synchronizes members — their *outputs*
  collapse toward equality (the reservoirs stay distinct; they do not become one
  reservoir), destroying the error independence the consensus depends on. So `|κ|` has
  a useful upper bound, found by sweep (§8).

### 4.2 Intensity ramp — start low, ramp on competence

Early in online training the members are still poor, so the consensus is poor, so
coupling hard to a bad consensus would **destabilize the ensemble** — every member
chasing a meaningless average. The schedule keeps the intensity low until the members
are good enough to trust the consensus:

```
|κ|
κ*                          ┌───────────────  hold at target κ*
                            ╱
                          ╱    ramp (gradual or small steps)
 κ₀ (≈0) ─────────────────
          └─── low while training error > threshold ───┘└ ramp once it crosses ┘
```

- **Start** at `κ₀` (zero or low): the mechanism is live but barely biting.
- **Gate on competence.** Hold `κ₀` until the **training error crosses a low
  threshold** — the members have learned the task well enough that their consensus is
  meaningful. (Competence-signal options: per-member training error, ensemble/consensus
  error, or inter-member agreement; default per-member training error — open, §11.)
- **Ramp** the intensity up to target `κ*` once the gate opens — gradually
  (linear/smooth) or in small steps with dwell. The ramp should be slow relative to the
  readout's online adaptation, so the readouts track the rising coupling rather than
  being shocked.
- **Hold** at `κ*` through the rest of training and into inference.

Because feedback is never off, the readouts learn under the rising coupling and reach
inference already adapted to `κ*` — there is no train/inference mismatch.

**Parameters:** `kappa_start` (≈ 0), `kappa_target` (κ*), the competence gate
(signal + threshold), and the ramp shape/rate. The schedule acts on the **magnitude**;
the direction (toward consensus) is fixed.

### 4.3 Clamp

`φ_i` passes through a bounding nonlinearity (the existing `tanh` seam on the feedback
path) as a runaway guard. Near-linear for well-behaved (small) deviations; load-bearing
only when the intensity is pushed hard. Whether it is applied is a config choice.

---

## 5. Members and diversity

The coupling exploits error **independence** between members; identical members have
zero deviation and contribute nothing to average out. Diversity sources, by cost:

1. **Reservoir weight seed** (always) — distinct `reservoir.seed` per member. Primary,
   cheapest axis.
2. **Input scaling / spectral radius** per member — spreads operating points.
3. **Activation shape** per member (e.g. plain `tanh` vs the Lorentzian `A`) —
   different return maps and empirically decorrelated errors. Opt-in beyond the
   seed-only default; widens the member-config surface.

**M (member count).** Parameter, default 3. Variance reduction ~1/M for independent
errors; returns diminish while cost grows linearly (M reservoirs stepped per online
step). 3 is the smallest M for which a median is meaningful. No hard upper bound; small
M is the demonstration target.

**Over-coupling vs diversity.** The same coupling that aligns members erodes the
diversity it needs; member diversity must be chosen and preserved against `|κ|` — the
central trade-off, made concrete by the κ sweep (§8).

---

## 6. Consensus statistic — mean vs median

- **Mean** (default). Clean variance-reduction reading; exact conservation
  `Σ_i Δ_i = 0`, so coupling adds no net drive to the ensemble. Sensitive to one
  straying member.
- **Median** (per channel; option). Robust to a single diverging member — valuable in
  the decorrelating regimes the coupling targets. Loses exact conservation and
  smoothness; for M = 3 it is the middle value per channel.

Config choice; default mean, median for robustness studies.

---

## 7. Integration with the codebase

Verified against the current `ESN` / `Reservoir` public API. `EnsembleESN` owns M
`ESN` members (each owns a non-copyable `Reservoir`, held by `unique_ptr`), built with
`num_feedback_channels = D`. The members are trained online (`InitOnline` +
`TrainLive*`) and read via `PredictLiveRaw`.

### 7.1 The one required core change

The online step must inject the orchestrator's own D-vector `φ_i` on the feedback
channels and step. `ESN::StepLive` instead evaluates the learned feedback policy F and
injects `tanh(F(x))` on channel 0 only; `Reservoir::InjectFeedback(channel, value)` is
public but reached through the private `reservoir_`. So the seam is one `ESN` method:

```
// proposed — additive, behavior-preserving (existing StepLive untouched)
void ESN::StepLiveExternalFeedback(const float* inputs,    // NumInputs() floats (task input)
                                   const float* feedback);  // num_feedback_channels (= D) floats
//   for c in [0, D):  reservoir_->InjectFeedback(c, clamp ? tanh(feedback[c]) : feedback[c])
//   for ch in inputs: reservoir_->InjectInput(ch, inputs[ch])
//   reservoir_->Step()      // F is never evaluated
```

~10 lines, purely additive. **Unused-F wrinkle:** members built with
`num_feedback_channels = D` make the `ESN` ctor eagerly build the unused F readout —
acceptable for a first cut (one wasted CNN per member); a clean follow-up is a small
`FeedbackConfig::external_drive` flag that allocates the feedback weight block but skips
building F. Required footprint is the one seam.

### 7.2 Orchestrator sketch (design pseudocode — not for implementation yet)

```cpp
class EnsembleESN {
    size_t M_, D_;
    Combine combine_;                                // Mean (default) | Median (§6)
    float  kappa_;                                   // current feedback intensity (ramped, §4.2)
    std::vector<std::unique_ptr<ESN>> esn_;          // each built with num_feedback_channels = D

    // one lockstep online step; writes consensus c(t). target == nullptr at inference.
    void Step(const float* input, const float* target, float* c_out) {
        std::vector<std::vector<float>> y(M_, std::vector<float>(D_));
        for (size_t i = 0; i < M_; ++i) esn_[i]->PredictLiveRaw(y[i].data());
        combine(y, c_out, combine_);                 // consensus (= ensemble output)
        std::vector<float> phi(D_);
        for (size_t i = 0; i < M_; ++i) {
            if (target) esn_[i]->TrainLiveStepRegression(target, /*lr,wd*/…);   // online update
            for (size_t c = 0; c < D_; ++c) phi[c] = kappa_ * (y[i][c] - c_out[c]);
            esn_[i]->StepLiveExternalFeedback(input, phi.data());               // the §7.1 seam
        }
        // caller advances kappa_ per the competence-gated ramp (§4.2)
    }
};
```

---

## 8. Falsifiable thesis and experiments

**Baselines:**
1. **Single member** — the floor.
2. **Zero-intensity (κ = 0)** — the same `EnsembleESN` machinery held at `κ₀ = 0` (no
   coupling; the consensus is read out but never injected). This coincides with a plain
   output-averaging ensemble but requires **no separate configuration** — it is just
   the degenerate operating point of the one mechanism, used purely as a measurement
   baseline.

**Thesis.** In a regime where members decorrelate, there exists a feedback intensity
κ at which the coupled ensemble beats both the single member **and** the κ = 0 point.
If accuracy is monotone in `|κ|` with the optimum at an endpoint (κ = 0 wins, or full
sync wins), coupling buys nothing and the mechanism is falsified.

**Sweeps:**
- **Intensity:** κ across a range at fixed M and members; locate the accuracy-vs-`|κ|`
  curve and any interior optimum.
- **Ramp ablation:** the competence-gated ramp (§4.2) vs a fixed-intensity online run —
  does ramping reach a higher usable κ* by avoiding early destabilization?
- Then vary M and the consensus statistic.

**Metrics** are task-appropriate and deferred to the (future) examples; the design
commits only to these baselines and the κ sweep being decisive.

---

## 9. Risks, caveats, and out-of-scope

**Risks / caveats:**
- **Correlated errors from shared training data** cap what the consensus can average
  out; diversity (§5) is the mitigation.
- **Over-coupling collapses the benefit** (§4.1): strong `|κ|` synchronizes outputs and
  destroys the independence the consensus depends on.
- **Common-mode bias is invisible to coupling.** If all members drift the same way the
  consensus drifts with them and every Δ_i → 0; coupling controls disagreement, not
  shared bias.
- **Early-training instability** is what the ramp (§4.2) guards against; a badly-tuned
  gate/ramp can either destabilize (ramp too early/fast) or waste training (too
  late/slow).

**Out of scope (and explicitly NOT design drivers):**
- **Feedback-less / averaging-only ensembles** — trivial for a consumer to build from
  independent `ESN`s; no footprint here (§1).
- **Batch mode** — feedback cannot exist there (§1).
- **Repulsive coupling (push members apart)** — unstable runtime signal, no known-good
  setting.
- **Weighted / learned combiners (stacking)** — consensus is mean/median only in v1.
- **Large M / topologies beyond mean-field** — v1 is small-M, all-to-all.
- **Demonstration examples** — built after the capability lands.

---

## 10. Related work (UNVERIFIED — carried from concept notes; needs a lit pass)

Not re-verified; do not rest a claim on these without checking.
- *Consensus / synchronization coupling* is the grounded side: linearly coupled
  reservoir computers are reported to synchronize; reservoir ensembles combined through
  their feedback lines are reported to beat a single reservoir — both couple to
  combine/synchronize.
- *Diversity-by-repulsion* is classically a **training-time** idea
  (negative-correlation learning, the ambiguity decomposition), not a runtime feedback
  signal.

**Action:** a moderate literature pass to confirm/retire each pointer and check whether
runtime consensus coupling of an ESN ensemble through the feedback weights, with a
competence-gated intensity ramp, has a named precedent.

---

## 11. Open questions

1. **Does the coupling beat the κ = 0 point?** The decisive A/B (§8).
2. **Competence gate** — which signal (per-member error / consensus error /
   inter-member agreement) and what threshold opens the ramp.
3. **Ramp shape/rate** — gradual vs stepwise, and how slow relative to online readout
   adaptation.
4. **Intensity magnitude** — the upper bound on `|κ|` before over-synchronization.
5. **Member diversity** — how much seed (and optional activation/scaling) diversity is
   needed for informative deviations, and how coupling erodes it.
6. **Consensus statistic** — mean (conservative) vs median (robust) under divergence.
7. **Clamp form** — `tanh` on `φ_i` vs a per-channel slew/magnitude cap.
