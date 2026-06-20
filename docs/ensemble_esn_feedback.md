# Ensemble ESN — Design

> Status: **implementing**. Design finalized; the `EnsembleESN` orchestrator
> (§7.3) is now being built in code (`EnsembleESN.{h,cpp}`). Task-agnostic by
> intent: this specifies the
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
  ramp's low-intensity start), but it is a degenerate setting of the one mechanism,
  **not** a supported feedback-less
  configuration.

Everything below describes that single, feedback-centric, online machine.

---

## 2. The consensus frame — output space

The consensus is formed in **output space**, the only frame the members share. All
members run the **same base configuration** and differ only by their reservoir `seed`
(§5) — so they share the hypercube topology but carry **different random weight
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

**One D, three roles.** A member's output dimension, its readout output count, and its
feedback-channel count are the **same number**: `D = NumOutputs() = num_feedback_channels`.
The readout emits D values, the consensus/deviation live in `R^D`, and the coupling drive
`φ_i` is injected on exactly D feedback channels — there is no separate sizing knob. Members
are built with `num_feedback_channels = D` (§7.2).

```
consensus      c   = mean_i  y_i           (per channel; or median, §6)
deviation      Δ_i = y_i − c               (Σ_i Δ_i = 0 for the mean)
coupling drive φ_i = κ · Δ_i                (scaled deviation; injected raw, §4.1)
```

`φ_i` (D values) is injected on member i's D feedback channels and consumed by its next
`Step`. The coupling rides the existing feedback weight block — its own weights, its own
`feedback_scaling` fan-in, and (per the reservoir contract) it sits **outside** the
spectral-radius estimate, so it does not silently inflate the open-loop stability
budget.

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
  which degrades the members. So κ has a useful upper bound, found empirically. What
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
          └─── low while consensus error > threshold ──┘└ ramp once it crosses ┘
```

- **Start** at `κ₀` (zero or low): the mechanism is live but barely biting.
- **Gate on competence — measured on the ensemble output.** Hold `κ₀` until the
  **ensemble's training error crosses a low threshold**. The competence signal is the
  error of the **consensus output** `c` against the target — **not** per-member errors:
  `c` is the quantity the ensemble actually delivers, and (being an average) it reaches
  competence at least as cleanly as any single member. This keeps the gate to **one
  scalar error stream** instead of M. The class maintains **one running estimate** of
  this error (e.g. an EMA or windowed mean of the per-step consensus error); the gate
  reads it. The exact smoothing and threshold are tunable (§9 Q2).
- **Ramp** the intensity up to target `κ*` once the gate opens — gradually
  (linear/smooth) or in small steps with dwell. The ramp should be slow relative to the
  readout's online adaptation, so the readouts track the rising coupling rather than
  being shocked.
- **Hold** at `κ*` through the rest of training and into inference.

Because feedback is never off, the readouts learn under the rising coupling and reach
inference already adapted to `κ*` — there is no train/inference mismatch.

**Ownership — the class drives κ, not the consumer.** The ramp is **owned by
`EnsembleESN`**. Its schedule parameters — `kappa_start` (≈ 0), `kappa_target` (κ*), the
competence gate (threshold), and the ramp shape/rate — are **constructor
config**. The class advances κ internally on each `Step`: it forms the consensus, so it
owns the single consensus-error signal the gate reads, and the consumer never computes
κ — it just feeds data and reads the ensemble output. (`EnsembleESN` is therefore a
*policy* object, not a bare lockstep stepper.) The schedule moves κ over [0, κ*]; the
sign is fixed by convention (κ > 0, §4.1).

**Readout learning rate (the members' online `lr` / `wd`) — held constant through the
ramp.** This is separate from κ: it governs how fast each member's *readout* adapts via
`TrainLiveStepRegression`, not how hard the members couple. One **shared** `lr` /
`weight_decay` is constructor config (§7) and is passed verbatim to every member's online
step — members share the base config (§5), so there is no reason to differ them. The
binding rule is the ramp interaction above: the κ ramp must stay slow *relative to* readout
adaptation, which fails if the readouts stop adapting — so **`lr` is held effectively
constant (or floored) through the ramp**, never annealed toward zero while κ is still
moving. Annealing `lr` is optional and only *after* κ reaches κ* and holds (a convergence
refinement); that schedule is tunable (§9 Q3). Note this is the ESN *online* `lr` passed
per step, not `ReadoutConfig`'s batch cosine fields, which the online path ignores.

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

They differ in exactly **one** field — the reservoir `seed`. A `Reservoir` fans its
single `seed` internally, via a SplitMix64 mix, into independent labelled substreams
(recurrent, input, feedback, **bias**, SR-probe), so one integer fully determines a
member's entire random realization — recurrent/input/feedback weights *and* the
per-neuron bias vector alike. (`bias_scaling` is shared, so the bias *magnitude*
matches across members; only its realization differs.) There is no separate `bias_seed`.

That is the **entire** source of member diversity. Identical dynamics under different
random realizations decorrelate the members' errors — which is what makes the
deviations `Δ_i` informative — while the shared base config keeps the members on the
same operating point and directly comparable in output space (§2).

The orchestrator derives the M member seeds from one **ensemble seed** by the same
mixer the reservoir uses internally:

```
seed_i = mix64(ensemble_seed ^ (GOLDEN * (i + 1)))      // 0 <= i < M
```

Equal ensemble seeds reproduce the ensemble exactly; distinct `i` yield independent
reservoir realizations. Because the reservoir self-fans one seed into all of its
streams, the ensemble needs to derive only **one** seed per member, not a role-tagged
pair — the seed-derivation problem reduces to a single deterministic line.

**M (member count).** Parameter, default 3. Variance reduction ~1/M for independent
errors; returns diminish while cost grows linearly (M reservoirs stepped per online
step). 3 is the smallest M for which a median is meaningful. No hard upper bound; small
M is the demonstration target. The per-step cost also **parallelizes** cleanly: members
interact only through the shared consensus, a single reduction per tick. That splits each
tick into two member-parallel regions — read all `y_i`, then (after the consensus reduce)
train/inject/step each member — with no member-to-member dependence inside either. The
project already links OpenMP.

---

## 6. Consensus statistic — mean vs median

- **Mean** (default). Clean variance-reduction reading; exact conservation
  `Σ_i Δ_i = 0`, so coupling adds no net drive to the ensemble. Sensitive to one
  straying member.
- **Median** (per channel; option). Robust to a single diverging member — valuable in
  the decorrelating regimes the coupling targets. Loses exact conservation and
  smoothness; for M = 3 it is the middle value per channel.

Config choice; default mean, median for robustness studies. Both are worth exploring.

---

## 7. Integration with the codebase

Verified against the current `ESN` / `Reservoir` public API. `EnsembleESN` owns M
`ESN` members (each owns a non-copyable `Reservoir`, held by `unique_ptr`), built with
`num_feedback_channels = D`. The ensemble drives every member through its own loop on the
`StepLiveExternalFeedback` seam (it does **not** call `ESN::InitOnline` — it owns the
washout itself, §7.1), trains them online via `TrainLiveStepRegression`, and reads them
via `PredictLiveRaw`.

### 7.1 Online lifecycle — one loop, two scheduled knobs

`EnsembleESN` runs in a **single execution mode** from the first step to the last. There
is no separate warm-up mode, no reservoir-only path, and no `if (warming_up)` branch in
the member loop. Every step — warm-up, training, inference — goes through the **same**
`StepLiveExternalFeedback` seam (§7.2). What changes over the run is not the code path
but the value of **two scheduled knobs**.

**Why warm-up is not a separate mode.** A reservoir warm-up exists only to wash the
arbitrary initial state `x(0) = 0` out of the dynamics before the readout's outputs are
used. Historically the readout build was *deferred* until after warm-up — but that build
takes no arguments and consumes no warm-up state; it is pure network construction
(allocate weights, init Adam moments). So the ordering "warm up, then build readout" was
**incidental, not a data dependency**: nothing about the readout needs the warm-up to have
happened first. That observation is now **realized in code (G14)**: the readout CNN is
built eagerly in the `Readout` ctor, the lazy-init dance (`Readout::InitOnline`) is gone,
and members are born ready. Warm-up is therefore just normal stepping with the early
outputs ignored — no dedicated execution mode.

`EnsembleESN` therefore **builds every member's readout up front** and folds warm-up into
the normal loop:

- **Construction.** Each member is built with `num_feedback_channels = D` (§7.2). The
  readout CNN is built **eagerly in the `Readout` ctor** (landed — G14), so members are
  **born ready**: there is no readout-init step of any kind, and the ensemble simply runs
  its unified loop from step 0.
  The ensemble owns the washout itself, through its own loop, on the external-feedback seam,
  with the coupling drive held at `φ = 0` (κ = 0). So warm-up is a clean, **input-only**
  reservoir washout — there is no internal feedback path to interfere (the ESN has none,
  §7.2).

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

### 7.2 Core changes — external-only feedback

> **Decision (2026-06-19), landed (G12, c355c8e).** The existing feedback machinery had
> never been exercised and was not treated as a fixed constraint. **All feedback is
> external; the ESN has no internal feedback policy.** The reservoir's sound feedback
> *substrate* was reused as the external-drive port, and the unfinished internal-F *policy*
> was **deleted** from the ESN. Footprint: a focused `ESN` refactor + small `Reservoir`
> touch-ups. The HCNN `Readout` was **untouched**.

**The substrate is sound and is reused (Reservoir).** Per-vertex state update already adds
a feedback term identical in form to the input term —
`Σ_i vtx_feedback_[v ^ NearestMask(i)] · fw[i]` (`Reservoir.cpp:266-270`) — fed by an
independent `n_·dim_` weight block (`:68`), scaled by `feedback_scaling` (`:118`), and
**excluded from the spectral-radius rescale** (`:164`, exactly as input is).
`InjectFeedback(channel, value)` (`:294-303`) block-partitions the N vertices into
`num_feedback_channels` regions and broadcasts each value to its block. This is precisely
the D-channel external drive the ensemble needs. Reservoir touch-ups that landed alongside:

- *Ergonomics:* a vector-form `InjectFeedback(const float* φ, size_t count)` that loops
  the per-channel call — the D-channel external-drive entry point.
- *Relaxed the divisibility throw* (`:50`) to permit any **D ≤ N**. The guard had been
  conservative, not load-bearing: when D ∤ N only the `N mod D` (`≤ D−1`) tail
  vertices go unwritten — they hold reset-zero and act as zero feedback *sources* while
  still *receiving* coupling via the neighbor gather (`:269`); no index leaves `[0, N)`
  and `block ≥ 1` for any `D ≤ N`. Dropped fraction `(N mod D)/N` is negligible for the
  ensemble's small-D / large-N = 2^dim regime. No power-of-two restriction on D.

**The ESN change — internal feedback removed entirely.** Binding decision: **all feedback
is external; the ESN has no internal feedback policy.** The ESN had built an internal
learned-F apparatus whenever `num_feedback_channels > 0` (the internal-F `if` block plus
the `StepLive` F-injection branch): `feedback_readout_`, its eager `InitOnline`,
`InjectFeedbackClamped`, the `ProbeLoss` / `TrainFeedbackCycle` machinery, the
decision/prediction/telemetry buffers, and the `Get/SetFeedback*` accessors — gated by a
guard that threw unless `num_feedback_channels == 1`. **All of it was deleted.** There is
no mode enum and no quarantined path, because there is no second kind of feedback to
distinguish. After removal:

- `num_feedback_channels` means exactly one thing: **how many external-drive feedback
  channels the reservoir carries.** `0` = no feedback; `D > 0` = D externally-supplied
  channels. No learned policy, no F readout, no telemetry — ever.
- The `== 1` guard is gone; any **D ≤ N** is allowed (the substrate handles it, §above).
- `ESN::StepLive(inputs)` is **input-only** — the `tanh(F(x))`-on-channel-0 branch was
  removed with F. It serves the no-feedback case.
- `ESN::StepLiveExternalFeedback(inputs, φ)` (below) is the **only** way feedback enters.
- **Readout init — already done (G14).** The readout CNN is now built **eagerly in the
  `Readout` ctor**, so there is no readout-init step at all: members are born ready and the
  no-arg `InitOnline()` once planned here is unnecessary. `ESN::InitOnline(inputs, count)`
  survives as warm-up-only sugar for single-ESN users; the ensemble does not call it.

The result is a *simpler* ESN — one fewer feature, not more complex.

**The step seam (ESN).**

```
// the only feedback entry point; StepLive(inputs) is the input-only (no-feedback) path
void ESN::StepLiveExternalFeedback(const float* inputs,    // NumInputs() floats (task input)
                                   const float* feedback);  // D floats (the coupling drive φ_i)
//   for c in [0, D):  reservoir_->InjectFeedback(c, feedback[c])   // raw, no clamp (§4.3)
//   for ch in inputs: reservoir_->InjectInput(ch, inputs[ch])
//   reservoir_->Step()
```

**Net footprint (landed).** Deleted the ESN internal-F apparatus; redefined
`num_feedback_channels` as the external-channel count and dropped the `== 1` guard; trimmed
`StepLive` to input-only and added the `StepLiveExternalFeedback` seam. `Reservoir`
touch-ups: the vector inject and the relaxed divisibility throw. **Zero** `Readout` /
HypercubeCNN changes. Net effect: the ESN shed a half-baked feature and gained one clean,
external, well-defined feedback path.

### 7.3 Orchestrator sketch (the implementation blueprint — landed in `EnsembleESN.{h,cpp}`)

```cpp
class EnsembleESN {
    size_t M_, D_;
    size_t t_ = 0, W_;                               // step counter; W_ = washout length (§7.1)
    Combine   combine_;                              // Mean (default) | Median (§6)
    RampConfig ramp_;                                // kappa_start/target, gate, shape — ctor config (§4.2)
    float     kappa_;                                // current intensity, advanced INTERNALLY (§4.2)
    float     lr_, wd_;                              // shared readout online lr / weight-decay — ctor config;
                                                     // held constant through the ramp (§4.2 / G4)
    float     consensus_err_;                        // running estimate (EMA/window) of the consensus-vs-
                                                     // target error — the competence signal (§4.2 / G3)
    std::vector<std::unique_ptr<ESN>> esn_;          // each: num_feedback_channels = D (external feedback,
                                                     // §7.2), readout born ready (built in ctor, §7.1)

    // class-owned competence-gated ramp (§4.2): fold this step's consensus error
    // into consensus_err_, and once it crosses the gate threshold, step kappa_ → κ*.
    void AdvanceKappa(const float* c_out, const float* target);

    // one lockstep online step; writes consensus c(t). target == nullptr at inference.
    void Step(const float* input, const float* target, float* c_out) {
        std::vector<std::vector<float>> y(M_, std::vector<float>(D_));
        for (size_t i = 0; i < M_; ++i) esn_[i]->PredictLiveRaw(y[i].data());
        combine(y, c_out, combine_);                 // consensus (= ensemble output)
        const bool train = target && (t_ >= W_);     // suppress fitting during the [0,W) washout (§7.1)
        std::vector<float> phi(D_);
        for (size_t i = 0; i < M_; ++i) {
            if (train) esn_[i]->TrainLiveStepRegression(target, lr_, wd_);      // shared online lr/wd (§4.2/G4)
            for (size_t c = 0; c < D_; ++c) phi[c] = kappa_ * (y[i][c] - c_out[c]);
            esn_[i]->StepLiveExternalFeedback(input, phi.data());               // the §7.2 seam
        }
        AdvanceKappa(c_out, target);  // class drives κ from the consensus error (§4.2/G3) — not the caller
        ++t_;
    }
};
```

### 7.4 Diagnostic surface (read-only)

A consumer often needs more than the consensus `c_out` that `Step` hands back — the
per-member outputs behind it, the current coupling intensity, and where the schedule
stands. The design commits to this **minimal** read-only surface; all are `const` getters
over existing private state and none change the mechanism.

```cpp
// Member outputs behind the consensus — e.g. to gauge inter-member agreement or read
// a single member in isolation. AllMemberOutputs fills an M×D row-major buffer in one
// call (a pairwise comparison forms an M×M table every step, so M calls would waste it).
void   MemberOutput(size_t i, float* out) const;   // member i: D floats (y_i)
void   AllMemberOutputs(float* out_MxD)  const;     // all members: M×D, row-major

float  Kappa()       const;   // current intensity — the operating point of the κ schedule (§4.2)
bool   GateOpen()    const;   // has the competence ramp triggered? — marks where coupling began
size_t CurrentStep() const;   // t_ — aligns traces to the §7.1 schedule (e.g. the ramp's trigger step)
```

Deliberately **not** exposed:
- **Per-member training error.** G3 made the competence gate read the *consensus* error
  only; any per-member error a consumer wants is computable externally from `MemberOutput`
  + the target. Keeping it off the surface avoids implying the gate consumes it.
- **The raw gate signal (`consensus_err_`).** `GateOpen()` — the boolean outcome — is the
  observable that matters; exposing the smoothed scalar would only invite coupling to a
  knob whose smoothing and threshold are still open (§9 Q2).

---

## 8. Risks, caveats, and out-of-scope

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

## 9. Open questions

1. **Does the coupling beat the κ = 0 point?** The decisive A/B the whole mechanism rests on.
2. **Competence gate** — the signal is fixed: the running error of the **ensemble
   (consensus) output** vs target (§4.2). Still open: the smoothing (EMA factor vs
   window length) and the threshold value that opens the ramp.
3. **Ramp shape/rate** — gradual vs stepwise, how slow relative to online readout
   adaptation, and whether/how to anneal the readout `lr` *after* κ holds (it is held
   constant through the ramp, §4.2/G4).
4. **Intensity magnitude** — the useful range of κ and where over-driving begins to
   degrade members (the sign is fixed by convention, §4.1).
5. **Common-mode bias** — out of scope here (the consensus is blind to it, §8), but if
   ever pursued, **member heterogeneity** (mixed activations first — tanh vs `A` are
   known to decorrelate — then operating points / bagged data) is the lever that shrinks
   the shared-error floor; only an external reference can remove it.
