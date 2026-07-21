# Design (temporary): Internal Full-State Linear Feedback

> **Status: temporary design — not landed.**  
> Working title for an expansion of the reservoir’s drive paths. This is the
> place to argue shape before code. When the feature lands (or is rejected),
> fold the durable parts into
> [reservoir_feedback_mechanism.md](reservoir_feedback_mechanism.md) /
> [Reservoir.md](Reservoir.md) / the SDKs and **delete this file**.
>
> Related but **not** the same document:
> [full_state_linear_feedback.md](full_state_linear_feedback.md) is an earlier
> *policy-on-external-port* sketch (Ehlers et al. mapping, outer-loop training).
> This design **revises** that sketch: application of full-state feedback becomes
> a **dedicated internal subsystem** of the reservoir; the external feedback port
> stays as-is for caller-owned loops (e.g. Lorenz free-run).
>
> ### Decided for first land
>
> | Choice | Decision |
> |--------|----------|
> | Subsystem shape | Private members + thin private helpers on `Reservoir` (no public FSF class / no extra public header) |
> | Drive path | Dedicated **B_fsf** (third optional drive port) — not same-B on input, not overload of ext-feedback |
> | Enable | **Construction-only** via config; off ⇒ **zero** FSF allocation (no buffer, weights, or V) |
> | Channels | **Scalar φ only, forever** — φ = V · x (one float); **V has length N**. Multi-channel FSF is **out of scope permanently** — do not design, stub, or plan for D>1 |
> | **V** | **Fixed parameter** — init 0; `Set`/`Get` only; **not** trained by the library |
> | Training V | **Deferred** |
> | `fsf_seed` | Seeds **only** `B_fsf` weights; independent of `reservoir.seed`; **ignored when FSF off**; default a fixed constant (not derived from reservoir seed) |
> | Set/Get when off | **Throw** (fail loud) |
> | Hot path | When enabled, always stage φ and gather (no special “V all zero” skip in v1) |
> | FSF weight draw | U(−1,1) from **`fsf_seed`**, then × `fsf_scaling/√dim` (mirror ext-feedback geometry) |
> | Config names | `full_state_feedback`, `fsf_seed` (default **1**, still a config field), `fsf_scaling` (0.5) |
> | ESN surface | **First-class in the same land** — not a follow-up wrapper (see §5.3) |
> | Smoke | Small C++ smoke: FSF off / V=0 / V≠0; C++ only; no V persistence; no Python yet |
> | Multi-channel FSF | **Never** |
> | Existing feedback port rename | **A — full rename** to *external* feedback (no aliases). In-tree only; no external API freeze. Same PR as FSF preferred |

---

## 1. Problem / goal

### What we already have

The reservoir exposes two per-step **external** drive ports:

| Port | Staging API | Who supplies the values | Role |
|------|-------------|-------------------------|------|
| Input | `InjectInput` | Caller | Task drive `u` |
| External feedback | `InjectFeedback` | Caller | Arbitrary closed-loop policy (ŷ feedback, metrics, …) |

See [reservoir_feedback_mechanism.md](reservoir_feedback_mechanism.md). That port is
source-agnostic by design: the reservoir does not invent the drive. ESN’s only
closed-loop seam is `ReservoirStep(inputs, feedback)`.

That correctly supports **external** feedback. It is a poor home for **full-state
linear feedback** (FSF): a gain vector **V** with fixed policy φ = Vᵀ x every
step. Pushing FSF through the external port forever means every warm-up, run,
free-run, and replay path must remember to inject φ — easy to get wrong, and it
conflates two different contracts.

### What we want (v1)

1. **Keep** all current functionality (open-loop, external feedback, Lorenz-style
   wiring, snapshots, SR story).
2. **Add** a subsystem **dedicated** to full-state linear feedback:
   - Completely **internal** application: each `Step` computes and applies the
     drive when enabled; the caller does not stage φ for FSF.
   - Mechanically a **twin** of the external feedback path (buffer + weight block
     + XOR gather + clear), not a rewire of topology or SR.
   - **V is a fixed parameter** (like bias): constructed to 0, changed only via
     explicit `Set` / load — never updated inside `Step`, `Train`, or any
     reservoir hot path.
3. **Do not** overload `InjectFeedback` for FSF. External feedback remains the
   only caller-owned closed-loop hook.

### Non-goals (v1)

- **Any** library training of V (nested GD, FD, online adaptive, `Train` hooks).
  Deferred entirely; the mechanism is *training-ready* (settable V) without
  being *trainable* yet.
- Same-B realization (adding Vᵀx onto `u` and reusing `B_in`) as the default.
- **Multi-channel FSF** (D>1, multi-column V, regional broadcast of several
  projections). **Permanently out of scope** — not a v2 item, not a stub, not a
  config hook. (External multi-channel *feedback* remains a separate, existing
  port and is unchanged.)
- Baking FSF into the spectral-radius secant solve.
- Changing HypercubeCNN or the readout.
- First-class V persistence format (nice-to-have after Set/Get; not a gate for
  mechanism smoke).

---

## 2. Conceptual model — three drive paths

After this work, the reservoir has **up to three** optional drive paths:

```
  weight layout (conceptual)

  [ input | external feedback | FSF | recurrent ]
     B_in        B_ext           B_fsf    A  (SR-rescaled only)
```

| Path | Owner of values each step | Parameters | In SR rescale? |
|------|---------------------------|------------|----------------|
| Input | Caller | input weights, `input_scaling` | No |
| External feedback | Caller | feedback weights, `feedback_scaling` | No |
| **FSF (new)** | **Reservoir** (φ = Vᵀ x_k) | FSF weights, `fsf_scaling`, **V** | No |
| Recurrent | Fixed at construction | recurrent block | **Yes** |

Effective closed-loop map when FSF is on (scalar channel, schematic):

```
  x_{k+1} = g( A x_k  +  B_in u_k  +  B_ext φ_ext,k  +  B_fsf (V · x_k)  + … )
```

Paper form is A′ = A + B Vᵀ with a single B. Our default is the same *structure*
with a **dedicated** B_fsf (decoupled from task input and from external feedback).
Document that explicitly; do not claim bit-identity with Ehlers same-B theorems
unless a same-B experiment is added later.

### Revised binding decision (docs must update when landed)

**Before:** all feedback is external; no internal feedback policy.

**After:**

- **External feedback** — only caller-owned closed-loop hook (`InjectFeedback`).
- **Full-state linear feedback** — optional reservoir-owned drive with fixed
  policy φ = Vᵀ x (application internal). **V is a settable fixed parameter in
  v1**; learning V is a later concern and never lives inside `UpdateState`.

Reservoir still does **not** know about the readout. FSF is not output feedback.

---

## 3. Lifecycle and causality

φ is computed from the **published pre-step state** (same causal pattern as the
paper and as any external drive that uses x_k):

```
  have x_k   (= Outputs() / SliceAt(0) after previous Step, or zeros after Clear)
     │
     ├─ caller may stage u and/or φ_ext   (InjectInput / InjectFeedback)
     ├─ if FSF enabled: stage φ_fsf = Vᵀ x_k internally
     └─ Step:
          UpdateState (gather all staged drives + recurrent history)
          age history ring
          clear staged drives (input, ext-feedback, FSF buffer)
          ──► x_{k+1} published
```

Rules:

1. **No extra lag** unless we deliberately study a delayed variant. Default is
   simultaneous use of x_k in the recurrence and in φ.
2. **Warm-up / run under the same V** used for training and deploy. Because FSF
   is applied inside `Step`, `ESN::ReservoirWarmup` / `ReservoirRun` /
   `ReservoirStep` stay consistent without caller changes.
3. **V = 0** (or FSF disabled) ⇒ no FSF contribution. Disabled path should be
   bit-identical to today’s open-loop for a fixed seed and input sequence when
   external feedback is also unused.
4. **Coexistence:** external feedback and FSF may both be on. Independent drives.

---

## 4. What is state vs parameters

| Quantity | Kind | Clear | Snapshot | Seed-rebuild |
|----------|------|-------|----------|--------------|
| vertex state + history | dynamical state | zeros | yes | n/a |
| staged u / φ_ext / φ_fsf | per-step scratch | cleared every Step | no | n/a |
| input / ext-fb / FSF / recurrent weights | construction params | untouched | no | yes (seed + config) |
| bias | construction params | untouched | no | yes |
| **V** | **fixed params** (set/load; not learned in v1) | untouched | **no** | **no** — must persist if non-zero in a shipped model |

V is a thin **gain vector**, not dynamical state and not (yet) an optimized
layer. Seed + config rebuild `B_fsf` weights; they do **not** rebuild a
caller-set V.

---

## 5. Config / API surface

### 5.1 `ReservoirConfig` (FSF fields)

| Field | Default | Meaning |
|-------|---------|---------|
| `full_state_feedback` | `false` | Construction-only master enable; **false ⇒ zero FSF allocation** |
| `fsf_seed` | `1` | **Config parameter** (user may set any `uint64_t`). Seeds **only** `B_fsf`. Default is the fixed constant `1`, **not** derived from `seed`. Ignored when FSF is off |
| `fsf_scaling` | `0.5f` | DIM-invariant drive (`× fsf_scaling/√dim`), same story as input / external feedback |

No channel-count field. FSF is one scalar φ forever.

When `full_state_feedback == false`: allocate nothing FSF-related.

When true: allocate FSF weight block `N·dim`, per-vertex staging buffer, gain V
length **N** init **0**. Weight draw: U(−1,1) from an RNG seeded by `fsf_seed`
only, then fan-in scale.

Staging each `Step`: φ = V · x → write φ on every vertex of the FSF buffer →
XOR-gather through `B_fsf`.

Weight layout:

```
  [ input | external-feedback (if D_ext>0) | FSF (if enabled) | recurrent ]
```

SR rescale: recurrent block only (after all drive blocks).

### 5.2 `Reservoir` API

```cpp
// Construction-only enable via config. Set/Get throw if FSF was not enabled.

bool FullStateFeedbackEnabled() const;

void SetFullStateFeedbackGain(const float* v, size_t n);       // requires n == N
void GetFullStateFeedbackGain(float* v_out, size_t n) const; // requires n == N

// No public InjectFsf — staging is internal to Step.
```

`GetConfig()` must round-trip the three FSF config fields (enable / seed /
scaling). **V is not part of config** — it is separate settable parameter
state (like readout weights, not like `spectral_radius`).

`Step()` (pseudocode):

```cpp
void Reservoir::Step() {
    if (fsf_enabled_)
        StageFsfFromState(Outputs());  // φ = V · x; always gather when enabled
    // UpdateState all v; rotate history; clear input / ext-fb / fsf buffers
}
```

Private helpers only (no public FSF type, no extra public header).

### 5.3 `ESN` surface — first-class, same land

ESN is the normal user-facing type (examples, SDK). FSF must be complete there
in the first implementation, not a Reservoir-only prototype with wrappers later.

#### Config path

FSF knobs live on `cfg.reservoir` (already nested in `ESNConfig`):

```cpp
ESNConfig cfg;
cfg.reservoir.full_state_feedback = true;
cfg.reservoir.fsf_seed            = 1;      // or any experiment seed
cfg.reservoir.fsf_scaling         = 0.5f;
ESN esn(cfg);
// V is still 0 until SetFullStateFeedbackGain
```

`ESN::GetConfig()` returns those fields as stored at construction (same pattern
as other reservoir knobs). **Caveat for docs:** the stock comment “reservoir is
fully determined by config + seed” is **false for V** when FSF is on and V was
set — config + seeds rebuild weights/`B_fsf`, not a non-zero V. Update that
comment when landing; V requires Get/Set (and later persistence if shipped).

#### Accessors (mirror Reservoir; thin delegates)

```cpp
/// @brief True if this ESN was built with reservoir.full_state_feedback.
[[nodiscard]] bool FullStateFeedbackEnabled() const;

/// @brief Set the full-state gain V (length ReservoirNeuronCount()).
/// @throws if FSF not enabled, or n != N.
void SetFullStateFeedbackGain(const float* v, size_t n);

/// @brief Copy current V into v_out (length n == N).
/// @throws if FSF not enabled, or n != N.
void GetFullStateFeedbackGain(float* v_out, size_t n) const;
```

No extra readouts for `fsf_seed` / `fsf_scaling` beyond `GetConfig()` — same as
`input_scaling` / `seed` today (some are also mirrored in Python; C++ uses
GetConfig).

#### Drive path — no new `ReservoirStep` argument

```cpp
void ReservoirStep(const float* inputs, const float* external_feedback = nullptr);
//                                     ^^^^^^^^^^^^^^^^ name: see §5.4
```

| Drive | How it enters each step |
|-------|-------------------------|
| Input | `inputs` (always) |
| External feedback | optional pointer; caller-owned; D channels |
| **FSF** | **automatic inside `Reservoir::Step`** when enabled; uses current V |

Implications (document on Warmup / Run / Step):

- `ReservoirWarmup` / `ReservoirRun` already call open-loop `ReservoirStep`
  (no external feedback). **They still apply FSF** if enabled — correct and
  required so collect/train see the same closed map as live steps.
- Callers never pass φ for FSF. Passing a non-null second arg is **only**
  external feedback.
- `ESN::Train` / readout paths unchanged: they never touch V.

#### What ESN deliberately does **not** expose (v1)

| Omitted | Why |
|---------|-----|
| `TrainFullStateFeedback` / any V optimizer | Training deferred |
| Save/load of V | Deferred; Get/Set enough to experiment |
| Inject-FSF or override φ | Would break “internal policy” |
| Runtime enable/disable | Construction-only |
| Python bindings for FSF | C++ first |

#### Naming on ESN (consistency)

Prefer full words aligned with existing style (`NumFeedbackChannels`,
`CopyReservoirState`):

- `FullStateFeedbackEnabled`
- `SetFullStateFeedbackGain` / `GetFullStateFeedbackGain`

Avoid a parallel `Fsf*` short API unless we rename everything to short forms
(we should not).

### 5.4 Rename existing “feedback” → “external feedback” (**decided: A**)

No external API freeze — **full rename, no deprecated aliases**, ideally in the
same change set as FSF so docs teach three peer ports once.

| Old | New |
|-----|-----|
| `num_feedback_channels` | `num_external_feedback_channels` |
| `feedback_scaling` | `external_feedback_scaling` |
| `InjectFeedback` | `InjectExternalFeedback` |
| `NumFeedbackChannels` | `NumExternalFeedbackChannels` |
| `ReservoirStep(inputs, feedback)` | `ReservoirStep(inputs, external_feedback)` |
| docs / CLAUDE “feedback port” | “external feedback port” (or “external drive port”) |
| internals e.g. `vtx_feedback_`, `num_feedback_*` | prefer `vtx_ext_feedback_`, `num_ext_feedback_*` (or equivalent) for consistency |

**Behavior unchanged** — multi-channel block broadcast, outside SR, clear after
`Step`, caller-owned values. Names only.

Touch list (non-exhaustive): `Reservoir.h/.cpp`, `ESN.h/.cpp`, Lorenz example +
README, `reservoir_feedback_mechanism.md` (title/prose), `Reservoir.md`,
`CPP_SDK.md` / CLAUDE as needed, design cross-links.

### 5.5 Training of V — deferred (not v1)

v1 only **applies** whatever V is currently set. No outer loop, no gradient, no
coupling to `ESN::Train`. Reservoir must not depend on Readout.

---

## 6. Interaction matrix

| Mode | Expected behavior |
|------|-------------------|
| FSF off, no ext-fb | Today’s open-loop (golden) |
| Ext-fb only (Lorenz, etc.) | Unchanged |
| FSF on, V = 0 | Bit-identical to FSF off (same seed, inputs, no ext-fb) if weights still allocated but φ=0; prefer also a “disabled” path with zero allocation |
| FSF on, V ≠ 0 | States diverge from V = 0; ‖x‖ finite on long rollouts |
| FSF + ext-fb both on | Both drives contribute; caller still owns φ_ext only |
| Snapshot restore | Restores state/history; clears staged drives; **V unchanged** |
| `Clear` | Zeros dynamics; **V unchanged** |

---

## 7. Relation to Ehlers et al. and the older FSF doc

| Topic | Paper / old `full_state_*.md` | This design (v1) |
|-------|------------------------------|------------------|
| Core idea | u′ = u + Vᵀx or φ = Vᵀx; A′ = A + B Vᵀ | Same idea |
| Where φ is applied | Caller policy on input or ext port | **Internal** dedicated B_fsf |
| External ŷ feedback | Separate concept | Unchanged external port |
| Training V | Nested GD + linear W,C | **Out of scope for v1** — V is fixed/settable only |
| HCNN readout | Nested or freeze-V then train | Unchanged; readout training unrelated to V |
| “Do not bake V into Reservoir” | Old guidance | **Application** in Reservoir; **learning** still not in Reservoir |

Update `full_state_linear_feedback.md` when this lands so the two docs do not fight
(or retire the old “policy only” framing into a short “theory mapping” section).

---

## 8. Implementation phases (suggested)

Each phase should be independently testable. **v1 ends at P2** (mechanism +
settable V). Training is explicitly later.

| Phase | Deliverable | Pass bar |
|-------|-------------|----------|
| **P0** | Config + alloc + `Step` FSF path + Reservoir Set/Get; **ESN delegates** same API; docs on auto-apply in Warmup/Run | Build; FSF off golden |
| **P1** | Smoke: off / V=0 / fixed V≠0; ext-fb regression with FSF off | F0–F3 |
| **P1b** | External-feedback identifier rename + Lorenz/docs (may merge with P0) | All in-tree consumers compile; behavior identical |
| **P2** | Fold durable prose into substrate / Reservoir / SDK notes; trim this temp doc | Docs match code |
| **P3+** (**deferred**) | Train V | Separate pass |

P0 includes the ESN surface — not a later wrapper.

---

## 9. Verification plan (engineering bar)

| ID | Check | Pass criterion |
|----|-------|----------------|
| F0 | FSF disabled | Bit-identical states vs current main (fixed seed, input) |
| F1 | FSF on, V = 0 | Bit-identical to F0 (or documented ε if layout differs — prefer true identity via early-out when V is zero **or** skip gather when disabled) |
| F2 | Fixed V ≠ 0 | Trajectory differs; finite ‖x‖ |
| F3 | Ext-fb regression | Lorenz-style path (or existing feedback smoke) unchanged with FSF off |
| F4 | FSF + ext-fb | Both on does not crash; drives superpose |
| F5 | Snapshot | Restore + replay inputs under fixed V reproduces trajectory |
| F6 | Clear | Does not wipe V |

Training-related checks (F7, paper-style NMSE, …) wait for the deferred train
phase. Science extras later: same-B ablation, dim scaling, HCNN A/B with a
hand-set or eventually trained V*.

---

## 10. Risks and open questions

**Risks**

- **Stability:** FSF outside SR; oversized V can blow up dynamics. Mitigate in
  trainer (‖V‖ cap, reject steps), not by folding V into SR.
- **Doc drift:** substrate doc still says “external only” until updated.
- **Allocation / golden:** if FSF-on with V=0 still runs an extra gather of zeros,
  float paths may still match; prefer explicit “no FSF contribution” when
  disabled or V all-zero if bit-identity is required under enable-with-default.
- **API creep:** no training of V in `Reservoir` or `ESN` in v1 (agreed).

**Open questions for v1:** none material. Ready to implement when you say go.

---

## 11. Explicit non-decisions / deferred

- **How to train V** (FD vs RTRL vs coordinate descent) — deferred with training.
- Whether production demos ship FSF on by default (default remains **off**).
- Same-B experimental path.
- Interaction studies with `A_lorentz` / leak / history_depth (orthogonal knobs).
- V persistence and Python FSF bindings.

---

## 12. Summary

| Keep | Add / change (v1) |
|------|-------------------|
| Input port | Dedicated FSF weight block + buffer (`B_fsf`) |
| Caller closed-loop **behavior** | Rename to **external** feedback (A); internal φ = V · x |
| SR on recurrent block only | **V fixed settable** (init 0); full ESN Get/Set |
| Reservoir ↔ Readout decoupling | `fsf_seed` config param (default 1) |
| Lorenz behavior | Docs: three optional drive ports |

**One-liner:** third optional drive port applies full-state linear feedback from
a fixed gain V (config + SetV; no library training); external caller feedback
stays; train V later if at all.

---

## Reference

- Ehlers, Nurdin & Soh — *Improving the Performance of Echo State Networks
  Through Feedback*, arXiv:2312.15141.
- [reservoir_feedback_mechanism.md](reservoir_feedback_mechanism.md) — landed
  external port.
- [full_state_linear_feedback.md](full_state_linear_feedback.md) — prior
  policy-only mapping (to be reconciled).
- [Reservoir.md](Reservoir.md) — dynamics reference (update when landed).
