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

The consensus is formed in **output space**, the only frame the members share. All
members run the **same base configuration** and differ only by their reservoir and bias
seeds (§5) — so they share the hypercube topology but carry **different random weight
realizations**. Vertex v of member 1 therefore has no correspondence to vertex v of
member 2, and their N-vertex state vectors are not comparable across members. The only
shared coordinates are the D-dimensional outputs — every member's readout is trained to
produce the same target quantity. The consensus, the deviations, and the coupling drive
are therefore all defined per output channel.

The consensus statistic is **mean** (default) or **median** (§6). The consensus serves
double duty: it is both the ensemble's output and the reference each member's deviation
is measured against. (Combining member outputs is intrinsic to the mechanism — it is how
the coupling reference is formed — not a separate averaging feature.)

---

## 3. Mechanism

Let M = member count, D = output dimension, member i output `y_i ∈ R^D`. The coupling
is live for the entire online run — training and inference alike.

```
consensus      c   = mean_i  y_i           (per channel; or median, §6)
deviation      Δ_i = y_i − c               (Σ_i Δ_i = 0 for the mean)
coupling drive φ_i = κ · Δ_i                (scaled deviation; injected raw, §4.1)
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
       inject φ_i = κ(t)·Δ_i(t)          on the feedback channels
       Step  ->  x_i(t+1)                                          (the seam, §7.2)
                               │
                               ▼
                 ensemble output = c(t)
```

Standard single-step closed-loop causality: the feedback injected into the t→t+1 step
is built from outputs read at state x(t), which already exist. No delay line. The only
difference between training and inference is whether each member's readout takes an
online update this step.

**What is fed back is the deviation, not the average.** Member i never receives the
consensus `c` itself — it receives its own departure from it, `Δ_i = y_i − c`. That
deviation **continually changes sign** over a run: a member is above the consensus on
some steps (Δ_i > 0) and below it on others (Δ_i < 0), so the injected signal
`φ_i = κ·Δ_i` already swings both ways regardless of κ's fixed sign. (This is the
concrete reason the sign of κ is only a scaling convention, §4.1 — there is no static
"direction" to the drive.) A member sitting exactly at the consensus receives nothing,
and `Σ_i Δ_i = 0`, so the coupling only redistributes drive among members — it never
injects the average as a net bias.

---

## 4. Feedback intensity

### 4.1 The mechanism is always on; intensity is the knob

There is no "engage feedback" switch (§1). From the first online step the consensus is
computed, the deviations are formed, and `φ_i` is injected. What moves is the scalar
**feedback intensity** κ:

- **Sign is a convention, not a regime.** Two reasons. First, the fed-back quantity is
  each member's deviation `Δ_i`, which already changes sign continually (§3), so the
  injected signal swings both ways no matter how κ's sign is fixed. Second, the drive is
  injected through each member's *fixed random* feedback weight block and a nonlinear
  reservoir, and the readouts are *trained* in its presence — so the sign carries no
  intrinsic stabilizing/destabilizing meaning (flipping κ's sign is equivalent to
  flipping the random feedback weights, a statistically identical realization; the
  readouts adapt either way). We fix **κ > 0** for consistency; a negative convention
  would work as well. Training sorts out how each member uses the signal.
- **Magnitude — the intensity.** This is what the schedule (§4.2) ramps. At `κ = 0` the
  mechanism is engaged but injects nothing; as κ rises, the deviation signal drives the
  members more strongly.
- **Over-driving.** The coupling cannot blow up (Δ_i is bounded by the members' bounded
  outputs and the reservoir's `tanh` bounds the state, §4.3), but too large a κ lets the
  injected drive dominate the dynamics — saturating neurons and swamping the task input —
  which degrades the members. So κ has a useful upper bound, found by sweep (§8). What
  the useful coupling buys, and how it degrades past the optimum, is empirical — not
  assumed from a consensus-dynamics analogy.

### 4.2 Intensity ramp — start low, ramp on competence

Early in online training the members are still poor, so the consensus is poor, so
coupling hard would **destabilize training** — every member driven by a meaningless,
noise-level deviation signal. The schedule keeps the intensity low until the members
are good enough that their deviation signal is meaningful rather than noise:

```
κ
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

**Ownership — the class drives κ, not the consumer.** The ramp is **owned by
`EnsembleESN`**. Its schedule parameters — `kappa_start` (≈ 0), `kappa_target` (κ*), the
competence gate (signal + threshold), and the ramp shape/rate — are **constructor
config**. The class advances κ internally on each `Step`: it holds the members, so it
owns the per-member competence signal the gate reads, and the consumer never computes
κ — it just feeds data and reads the ensemble output. (`EnsembleESN` is therefore a
*policy* object, not a bare lockstep stepper.) The schedule moves κ over [0, κ*]; the
sign is fixed by convention (κ > 0, §4.1).

### 4.3 No clamp on the coupling drive

The coupling drive `φ_i = κ·Δ_i` is injected **raw** — no bounding nonlinearity, for
either sign of κ. The drive cannot run away: the deviation `Δ_i` is bounded by the
members' bounded outputs, and the reservoir's own `tanh` already bounds the resulting
state. Over-large κ degrades by saturation (§4.1), not by blow-up — so there is nothing
for a clamp to guard.

---

## 5. Members — one shared config, different seeds

All M members are built from a **single shared base configuration**. They are identical
ESNs in every structural respect — same `dim`, `spectral_radius`, `leak_rate`,
`input_scaling`, `num_inputs`, `history_depth`, `history_floor`, activation
(`lorentz_gamma` / `lorentz_inv_sigma2`), `bias_scaling`, `feedback_scaling`,
`num_feedback_channels = D`, and the same readout architecture. Members are **not**
tuned to different operating points.

They differ in exactly two fields, both random seeds:

- **`seed`** — the reservoir weight seed: a different random realization of the
  recurrent / input / feedback weight blocks per member.
- **`bias_seed`** — the per-neuron bias seed: a different random bias vector per member
  (`bias_scaling` is shared, so the bias *magnitude* matches; only its realization
  differs).

That is the **entire** source of member diversity. Identical dynamics under different
random realizations decorrelate the members' errors — which is what makes the
deviations `Δ_i` informative — while the shared base config keeps the members on the
same operating point and directly comparable in output space (§2). The orchestrator
assigns the M distinct seed pairs (e.g. derived from one ensemble seed).

**M (member count).** Parameter, default 3. Variance reduction ~1/M for independent
errors; returns diminish while cost grows linearly (M reservoirs stepped per online
step). 3 is the smallest M for which a median is meaningful. No hard upper bound; small
M is the demonstration target.

---

## 6. Consensus statistic — mean vs median

- **Mean** (default). Clean variance-reduction reading; exact conservation
  `Σ_i Δ_i = 0`, so coupling adds no net drive to the ensemble. Sensitive to one
  straying member.
- **Median** (per channel; option). Robust to a single diverging member — valuable in
  the decorrelating regimes the coupling targets. Loses exact conservation and
  smoothness; for M = 3 it is the middle value per channel.

Config choice; default mean, median for robustness studies.  We will want to explore the effects of both.

---

## 7. Integration with the codebase

Verified against the current `ESN` / `Reservoir` public API. `EnsembleESN` owns M
`ESN` members (each owns a non-copyable `Reservoir`, held by `unique_ptr`), built with
`num_feedback_channels = D`. The members are trained online (`InitOnline` +
`TrainLive*`) and read via `PredictLiveRaw`.

### 7.1 Online lifecycle — one loop, two scheduled knobs

`EnsembleESN` runs in a **single execution mode** from the first step to the last. There
is no separate warm-up mode, no reservoir-only path, and no `if (warming_up)` branch in
the member loop. Every step — warm-up, training, inference — goes through the **same**
`StepLiveExternalFeedback` seam (§7.2). What changes over the run is not the code path
but the value of **two scheduled knobs**.

**Why warm-up is not a separate mode.** A reservoir warm-up exists only to wash the
arbitrary initial state `x(0) = 0` out of the dynamics before the readout's outputs are
used. In the single-`ESN` online API this is bundled into `InitOnline`, which steps the
reservoir `warmup_count` times and *then* builds the readout. Crucially, the readout
build (`readout_.InitOnline()`) **takes no arguments and consumes no warm-up state** — it
is pure network construction (allocate weights, init Adam moments). So the ordering "warm
up, then build readout" is **incidental, not a data dependency**: nothing about the
readout needs the warm-up to have happened first. Running the readout during warm-up
would merely produce an output we ignore — one cheap HCNN forward per member per step —
which is not worth a dedicated execution mode. (Verified: `ESN::InitOnline` at
`ESN.cpp:162` calls `Warmup(...)` then the argument-free `readout_.InitOnline()`.)

`EnsembleESN` therefore **builds every member's readout up front** and folds warm-up into
the normal loop:

- **Construction.** Each member is built with `num_feedback_channels = D` (§7.2) and
  initialized with `InitOnline(input, /*warmup_count =*/ 0)` — this builds the readout and
  runs **zero** internal warm-up steps (the `Warmup` loop body never executes). The
  ensemble owns the washout itself, through its own loop, on the external-feedback seam —
  so warm-up is a clean, **input-only** reservoir washout with no internal-F dribble on
  channel 0 (contrast `StepLive`, whose internal-F path we never touch).

- **The two scheduled knobs.** From step 0 the loop runs `Step()` (§7.3) unchanged. Two
  scalars move on a schedule the class owns (§4.2):

  ```
    ONE loop, same seam throughout. Two scheduled flips, nothing else:

    step:   0 ............. W ..................... competence ............→
    κ:      0 ───────────────────────────────────┐ ramp ┌──── κ*
    train:  off ──────────┐ on ───────────────────────────────────────────
                          ▲                        ▲
                    enable training          open the κ ramp once the
                    after the W-step         competence gate fires
                    reservoir washout        (§4.2 / G3)

    "warm-up" = steps [0, W): a normal Step() with κ = 0 and training not yet enabled.
  ```

  1. **`train` enable** flips on at step `W` (the washout length). During `[0, W)` the
     reservoir is driven by the task input and the zero-deviation feedback only (κ = 0, so
     `φ_i = 0`), and **no readout update is taken** — we do not fit the readout on
     transient states that still remember `x(0) = 0`. `W` plays exactly the
     transient-killing role of the single-`ESN` `warmup_count`.
  2. **κ ramp** holds at `κ₀ ≈ 0` until the competence gate fires, then ramps to `κ*`
     (§4.2). Because κ = 0 across `[0, W)` anyway, the consensus is computed and read out
     but injected as ~nothing — warm-up is the natural **left edge of the κ schedule**,
     not a thing bolted on beside it.

- **Inference.** Same loop, same seam. The readout update is simply not taken
  (equivalently `target == nullptr`); κ holds at `κ*`. There is no mode switch between
  training and inference — only whether the readout update runs this step.

- **Sequence boundaries / reset.** When a fresh, independent sequence begins, every member
  resets together via `ResetReservoirOnly()` (clears reservoir state only; trained readout
  weights are preserved). The κ schedule and the `train`-enable knob are **not** rewound
  on such a reset — competence already achieved is not un-learned. Whether to re-impose a
  short washout (hold κ, suppress the readout update for a few steps while the dynamics
  re-settle) at each sequence boundary is a config choice; **default: yes, a short one.**

This collapses the lifecycle to **one run mode plus a two-knob schedule** — that is the
entire state machine. (This supersedes the earlier framing of warm-up as a distinct
"uncoupled phase that predates the readouts": with readouts built up front, warm-up is
just the schedule's left edge.)

### 7.2 Required core changes (two; no Reservoir or Readout change)

The ensemble rides almost entirely on **existing** machinery. Verified against the
sources, the required footprint is **two small, additive `ESN` changes** — and **zero**
changes to `Reservoir` or the HCNN `Readout`.

**(1) The external-feedback step seam.** The online step must inject the orchestrator's
own D-vector `φ_i` on the feedback channels and step. `ESN::StepLive` instead evaluates
the learned feedback policy F and injects `tanh(F(x))` on channel 0 only;
`Reservoir::InjectFeedback(channel, value)` is public but reached through the private
`reservoir_`. So we add one `ESN` method:

```
// proposed — additive, behavior-preserving (existing StepLive untouched)
void ESN::StepLiveExternalFeedback(const float* inputs,    // NumInputs() floats (task input)
                                   const float* feedback);  // num_feedback_channels (= D) floats
//   for c in [0, D):  reservoir_->InjectFeedback(c, feedback[c])   // injected raw, no clamp
//   for ch in inputs: reservoir_->InjectInput(ch, inputs[ch])
//   reservoir_->Step()      // F is never evaluated
```

~10 lines, purely additive.

**(2) A feedback mode that uses the reservoir's feedback channels *without* the internal
learned-F policy.** The current ESN couples **two independent things** under one knob,
`cfg.reservoir.num_feedback_channels > 0`:

- *(Reservoir — automatic.)* The feedback **substrate** — the `vtx_feedback_` buffer and
  the `n_·dim_` feedback weight block — is built inside `Reservoir::Create(cfg.reservoir)`
  (`ESN.cpp:66`), sized and block-partitioned by `num_feedback_channels`, **independent of
  anything the ESN does**. This is exactly what the ensemble needs, at D channels, and it
  already works (§7.2 verification above). Nothing to add here.
- *(ESN — the `if` block at `ESN.cpp:80-97`.)* The internal **learned-F apparatus** — the
  scalar `feedback_readout_`, its eager `InitOnline`, and the decision/prediction/telemetry
  buffers (`:94-96`) — gated by a guard that **throws unless `num_feedback_channels == 1`**
  (the message reads verbatim *"feedback training (v1) supports exactly 1 feedback
  channel"*). This is single-channel by construction, and the ensemble wants **none** of it
  — it drives the channels externally through the seam (1).

So the change is a small `FeedbackConfig::external_drive` flag that, when set, makes the
ESN ctor **skip the entire internal-F `if` block**: no F readout, no telemetry, and no
`!= 1` guard. It **allocates nothing** — the reservoir already built the D-channel
substrate. The mode simply **decouples** "having feedback channels" from "having an
internal F policy." `ESN`/config change only — no Reservoir change, no HCNN change.

**Required for D > 1; a cleanup for D = 1.** With `num_feedback_channels = D > 1` the
current ctor *throws*, so the flag is **mandatory** for any multi-output task. At D = 1
the ctor *succeeds* but builds the full internal-F apparatus the ensemble never touches
(harmless but wasteful), so there the flag is an **optimization**. Because the capability
targets general D ≤ N, treat it as required.

**Why no Reservoir change.** The `Reservoir` is already D-channel-native: it sizes the
feedback weight block at `n_*dim_` independent of channel count (`Reservoir.cpp:64`), and
`InjectFeedback(channel, value)` partitions the N vertices into `num_feedback_channels`
equal blocks and broadcasts each value to its block (`Reservoir.cpp:269-277`). Our D
deviation components map one-to-one onto D vertex blocks with no new code.

**D is unconstrained (verified against the sources).** `Reservoir.cpp:50` *throws*
unless `num_feedback_channels` divides N evenly, but the guard is conservative, not
load-bearing. `InjectFeedback` (`Reservoir.cpp:269-277`) writes channel c to vertices
`[c·block, (c+1)·block)` with `block = floor(N / D)`; when D does not divide N, the
`N mod D` (`≤ D−1`) tail vertices are never written and hold their reset value of 0
(full-N zeroing at `Reservoir.cpp:199/311/341`). Those vertices then act only as **zero
feedback *sources*** — they still *receive* coupling, since state update gathers each
vertex's neighbors' feedback (`Reservoir.cpp:244`), so nothing is cut off and no index
runs out of range (`v ^ NearestMask(i)` is always in `[0, N)`, and `block ≥ 1` for any
`D ≤ N`). Removing the check is therefore safe for **any D ≤ N**. The only effect is a
dropped feedback-source fraction of `(N mod D) / N`: negligible in the ensemble's regime
of small D vs large N = 2^dim (e.g. D = 3, N = 64 → 1 vertex), and material only if D
approaches N (outside the use case). No power-of-two restriction on D.

### 7.3 Orchestrator sketch (design pseudocode — not for implementation yet)

```cpp
class EnsembleESN {
    size_t M_, D_;
    size_t t_ = 0, W_;                               // step counter; W_ = washout length (§7.1)
    Combine   combine_;                              // Mean (default) | Median (§6)
    RampConfig ramp_;                                // kappa_start/target, gate, shape — ctor config (§4.2)
    float     kappa_;                                // current intensity, advanced INTERNALLY (§4.2)
    std::vector<std::unique_ptr<ESN>> esn_;          // each built with num_feedback_channels = D
                                                     // and external_drive (§7.2), via InitOnline(.,0)

    void AdvanceKappa(const float* target);          // class-owned competence-gated ramp (§4.2)

    // one lockstep online step; writes consensus c(t). target == nullptr at inference.
    void Step(const float* input, const float* target, float* c_out) {
        std::vector<std::vector<float>> y(M_, std::vector<float>(D_));
        for (size_t i = 0; i < M_; ++i) esn_[i]->PredictLiveRaw(y[i].data());
        combine(y, c_out, combine_);                 // consensus (= ensemble output)
        const bool train = target && (t_ >= W_);     // suppress fitting during the [0,W) washout (§7.1)
        std::vector<float> phi(D_);
        for (size_t i = 0; i < M_; ++i) {
            if (train) esn_[i]->TrainLiveStepRegression(target, /*lr,wd*/…);    // online update
            for (size_t c = 0; c < D_; ++c) phi[c] = kappa_ * (y[i][c] - c_out[c]);
            esn_[i]->StepLiveExternalFeedback(input, phi.data());               // the §7.2 seam
        }
        AdvanceKappa(target);    // class drives κ via the competence-gated ramp (§4.2) — not the caller
        ++t_;
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
- **Over-driving collapses the benefit** (§4.1): too large a κ lets the injected drive
  dominate the dynamics (saturation, swamped input), degrading members and the
  independence the consensus depends on.
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
- **Weighted / learned combiners (stacking)** — consensus is mean/median only in v1.
- **Large M / topologies beyond mean-field** — v1 is small-M, all-to-all.
- **Demonstration examples** — built after the capability lands.

---

## 11. Open questions

1. **Does the coupling beat the κ = 0 point?** The decisive A/B (§8).
2. **Competence gate** — which signal (per-member error / consensus error /
   inter-member agreement) and what threshold opens the ramp.
3. **Ramp shape/rate** — gradual vs stepwise, and how slow relative to online readout
   adaptation.
4. **Intensity magnitude** — the useful range of κ and where over-driving begins to
   degrade members (the sign is fixed by convention, §4.1).
5. **Common-mode bias** — out of scope here (the consensus is blind to it, §9), but if
   ever pursued, **member heterogeneity** (mixed activations first — tanh vs `A` are
   known to decorrelate — then operating points / bagged data) is the lever that shrinks
   the shared-error floor; only an external reference can remove it.
