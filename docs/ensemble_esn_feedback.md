# Ensemble ESN — Design

> Status: **implemented**. The `EnsembleESN` orchestrator (§7.2) is built in code
> (`EnsembleESN.{h,cpp}`), with Python bindings and persistence. The class is
> deliberately a **thin lockstep stepper**: each step it computes the consensus
> and injects each member's scaled deviation, but the **coupling-intensity
> schedule is the caller's policy** — the class only stores the current κ and
> applies it. An earlier design folded a competence-gated κ ramp and an initial
> washout *into* the class; both were removed as premature for this stage of
> development (see §4.2 and §7.1). Task-agnostic by intent: this specifies the
> general `EnsembleESN` capability. Demonstration examples are deliberately **out
> of scope** and will be built *after* the capability lands. No example drives
> this design.
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
  **no config flag and no special code path** for the no-feedback ensemble, and that
  use case is **not permitted to drive any design decision here.** (Uncoupled
  operation is still reachable — it is just the κ = 0 operating point, not a separate
  mode; see below.)

- **The consensus is always computed; intensity is the knob.** There is no
  "feedback on/off" switch in the code. From the first online step the consensus is
  formed and each member's deviation is available to inject; what scales that
  injection is the scalar **intensity** κ (§4). κ starts at 0 (members uncoupled)
  and is moved by the **caller** via `SetKappa`. A run that leaves κ = 0 throughout
  is the degenerate, uncoupled operating point of the one mechanism — not a separate
  feedback-less configuration.

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
are built with `num_feedback_channels = D` (§7).

```
consensus      c   = mean_i  y_i           (per channel; or median, §6)
deviation      Δ_i = y_i − c               (Σ_i Δ_i = 0 for the mean)
coupling drive φ_i = κ · Δ_i                (scaled deviation; injected raw, §4.1)
```

`φ_i` (D values) is injected on member i's D feedback channels and consumed by its next
`ReservoirStep`. The coupling rides the existing feedback weight block — its own weights, its own
`feedback_scaling` fan-in, and (per the reservoir contract) it sits **outside** the
spectral-radius estimate, so it does not silently inflate the open-loop stability
budget.

**Lockstep online step (training and inference alike):**

```
   all members at internal state x_i(t)
   ┌──────────────────────────────────────────────────────────┐
   │ 1. read outputs:   y_i(t) = readout_i( x_i(t) )           │   (Predict)
   │ 2. consensus:      c(t)   = mean_i y_i(t)                  │   (= ensemble output)
   │ 3. deviations:     Δ_i(t) = y_i(t) − c(t)                 │   (Σ_i Δ_i = 0)
   └───────────────────────────┬──────────────────────────────┘
                               ▼
   for each member i:
       (training only) online-update readout_i toward target(t)   (TrainStep)
       inject task input u(t)            on the input    channels
       inject φ_i = κ·Δ_i(t)             on the feedback channels
       Step  ->  x_i(t+1)                                          (the ReservoirStep seam)
                               │
                               ▼
                 ensemble output = c(t)
```

Standard single-step closed-loop causality: the feedback injected into the t→t+1 step
is built from outputs read at state x(t), which already exist. No delay line. The only
difference between training and inference is whether each member's readout takes an
online update this step. κ is whatever the caller has set; the class does not move it
(§4.2).

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

### 4.1 The consensus is always formed; intensity is the knob

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
- **Magnitude — the intensity.** This is the knob the caller sets via `SetKappa`
  (§4.2). At `κ = 0` the mechanism is engaged but injects nothing; as κ rises, the
  deviation signal drives the members more strongly.

### 4.2 Intensity is caller-managed

κ is a **single scalar the class stores and applies** — nothing more. The caller sets
it with `SetKappa(κ)` and reads it with `Kappa()`; the class holds it fixed between
calls and uses it verbatim in `φ_i = κ·Δ_i` on every `Step`. κ starts at **0**
(uncoupled), so a freshly constructed ensemble runs its members independently until the
caller raises it.

**Why the schedule lives with the caller, not the class.** Early in online training the
members are still poor, so the consensus is poor, so coupling hard immediately would
**destabilize training** — every member driven by a meaningless, noise-level deviation
signal. The sensible remedy is to keep κ low until the members are competent and then
raise it gradually. But *when* to raise, *how fast*, and *what competence signal* to
gate on are **policy**, not mechanism — and they depend on the task and the training
regime. So the class exposes only the mechanism (set κ, apply κ) and leaves the schedule
to the caller, who can implement any policy by calling `SetKappa` between `Step`s:

```
κ
κ*                          ┌───────────────  caller holds at its target κ*
                            ╱
                          ╱    caller ramps (any shape) once it judges the
 0 ───────────────────────     readouts competent
   └─ caller keeps κ = 0 while the consensus is still poor ─┘
```


**Readout learning rate (the members' online `lr` / `wd`).** Separate from κ: it governs
how fast each member's *readout* adapts via `TrainStep`, not how hard the members couple.
Both are **caller-managed at runtime** (`SetLr` / `SetWeightDecay`), not constructor
config — they start at 0 and are applied verbatim and identically to every member (members
share the base config, §5, so there is no reason to differ them). If the caller ramps κ,
that ramp should stay slow *relative to* readout adaptation (so the readouts track the
rising coupling rather than being shocked), which means not annealing `lr` toward zero
while κ is still moving — the caller owns that coordination and has the handles for it.
Unlike κ, `lr`/`wd` are *not* part of @ref State: a reload starts them back at 0. Note
this is the ESN
*online* `lr` passed per step, not `ReadoutConfig`'s batch cosine fields, which the online
path ignores.

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
`ReservoirStep(inputs, φ)` feedback seam, trains them online via `TrainStep`, and reads
them via `Predict`.

### 7.1 Online lifecycle — one loop

`EnsembleESN` runs in a **single execution mode** from the first step to the last. There
is no separate warm-up mode, no reservoir-only path, and no `if (warming_up)` branch in
the member loop. Every step goes through the **same** `ReservoirStep(inputs, φ)` feedback
seam. Only two things vary across the run, and both are decided per call, not by
an internal schedule:

```
   ONE loop, same seam throughout. Two per-call inputs decide everything:

     • target == nullptr ?   → no, fit each readout this step;  yes, inference only
     • κ (caller-set)        → scales the injected deviation φ_i = κ·Δ_i

   step:  0 .................................................→
   train: governed solely by whether `target` is passed each step
   κ:     0 by default; whatever the caller last set via SetKappa
```

- **Construction.** Each member is built with `num_feedback_channels = D` (§7); its
  readout is ready immediately, so the ensemble simply runs its unified loop from step 0.
  κ starts at 0.

- **Warm-up is the caller's concern, not the class's.** A reservoir warm-up exists only
  to wash the arbitrary initial state `x(0) = 0` out of the dynamics before the readout's
  outputs are trusted. The class no longer owns one. A caller that wants a washout simply
  drives the first few steps with `target = nullptr` (the reservoir is driven by the task
  input; no readout update is taken) — exactly the transient-killing role of a single
  `ESN`'s `warmup_count` — and only then starts passing targets. Leaving κ = 0 over those
  steps keeps the warm-up input-only.

- **Training vs inference.** The same loop, the same seam. Whether a step trains is
  decided by one thing: was a `target` supplied? `target == nullptr` is an inference step
  (no readout update). There is no mode switch.

- **Sequence boundaries / reset.** When a fresh, independent sequence begins, the caller
  calls `ResetReservoirStates()`: every member's reservoir state is cleared
  (`ReservoirClear()`), while trained readout weights and κ are **preserved** —
  competence already achieved is not un-learned. The reservoirs are cold
  afterward, so the caller re-warms (step with `target = nullptr`) before trusting outputs
  if the sequence break warrants it.

This collapses the lifecycle to **one run mode with two per-call inputs** — that is the
entire state machine.

### 7.2 Orchestrator sketch (the implementation blueprint — landed in `EnsembleESN.{h,cpp}`)

```cpp
class EnsembleESN {
    size_t M_, D_;
    Combine combine_;                                // Mean (default) | Median (§6)
    float   kappa_ = 0.0f;                           // coupling intensity — caller-managed (§4.2)
    float   lr_ = 0.0f, wd_ = 0.0f;                  // shared readout online lr / weight-decay (§4.2);
                                                     // runtime-only, caller-set (start at 0)
    std::vector<std::unique_ptr<ESN>> esn_;          // each: num_feedback_channels = D (external feedback),
                                                     // readout born ready (built in ctor, §7.1)
    std::vector<float> y_flat_, phi_;                // pre-allocated per-tick scratch: M*D and D (decision #5)

    void SetKappa(float k)        { kappa_ = k; }    // caller's handle on coupling intensity (§4.2)
    void SetLr(float lr)          { lr_ = lr; }      // caller's handle on readout lr / wd (§4.2)
    void SetWeightDecay(float wd) { wd_ = wd; }

    // one lockstep online step; writes consensus c(t). target == nullptr at inference.
    void Step(const float* input, const float* target, float* c_out) {
        for (size_t i = 0; i < M_; ++i)
            esn_[i]->Predict(y_flat_.data() + i*D_);  // zero-alloc read into pre-sized scratch
        combine(y_flat_, c_out, combine_);            // consensus (= ensemble output)
        const bool train = (target != nullptr);       // fit iff a target is supplied (§7.1)
        for (size_t i = 0; i < M_; ++i) {
            if (train) esn_[i]->TrainStep(target, lr_, wd_);      // shared online lr/wd
            const float* y_i = y_flat_.data() + i*D_;
            for (size_t c = 0; c < D_; ++c) phi_[c] = kappa_ * (y_i[c] - c_out[c]);
            esn_[i]->ReservoirStep(input, phi_.data());            // the ReservoirStep feedback seam
        }
    }
};
```

The class is intentionally tiny: it forms the consensus, injects the scaled deviation,
and steps the members. It never moves κ (or `lr`/`wd`) on its own and owns no washout —
all are decided by the caller (§4.2, §7.1).

### 7.3 Diagnostic surface (read-only)

A consumer often needs more than the consensus `c_out` that `Step` hands back — the
per-member outputs behind it and the current coupling intensity. The design commits to
this **minimal** read-only surface; all are `const` getters over existing private state
and none change the mechanism.

```cpp
// Member outputs behind the consensus — e.g. to gauge inter-member agreement or read
// a single member in isolation. AllMemberOutputs fills an M×D row-major buffer in one
// call (a pairwise comparison forms an M×M table every step, so M calls would waste it).
void   MemberOutput(size_t i, float* out) const;   // member i: D floats (y_i)
void   AllMemberOutputs(float* out_MxD)  const;     // all members: M×D, row-major

float  Kappa()       const;   // current intensity (whatever the caller last set, §4.2)
float  Lr()          const;   // current shared readout lr / weight-decay (§4.2)
float  WeightDecay() const;
```

Deliberately **not** exposed:
- **Per-member training error.** Any per-member error a consumer wants is computable
  externally from `MemberOutput` + the target; the class does not track it.

---

## 8. Open questions

1. **Common-mode bias** — out of scope here (the consensus is blind to it, §8), but if
   ever pursued, **member heterogeneity** (mixed activations first — tanh vs `A` are
   known to decorrelate — then operating points / bagged data) is the lever that shrinks
   the shared-error floor; only an external reference can remove it.
