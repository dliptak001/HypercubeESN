# Closed-loop - Areas of Exploration

A running list of exploratory ideas — not approved work, just captured directions
worth investigating.

## 1. Free Running

_Closed-loop / autonomous generative operation — the readout drives the reservoir
forward with no external input (e.g. Lorenz free-run)._

## 2. General

### Ensemble consensus feedback

**Maturity:** _Novel_ — ESN ensembles (averaging members) are established, but
feeding each member its deviation from the ensemble mean back through the
**feedback driver path** as a coupling signal is not standard procedure.

Run an ensemble of exactly **three** ESNs in parallel on the same task, each with
differently-seeded weights. At each step compute the ensemble mean of their outputs
`ȳ = (1/3)(y_1 + y_2 + y_3)`, then feed each member _i_ a scaled copy of its own
deviation from that mean, `Δ_i = y_i − ȳ`. A member running above consensus gets a
positive deviation, one below a negative one, one exactly at the mean gets zero.
The deviations are conservative — `Σ_i Δ_i = 0` by construction — so the coupling
only redistributes drive among members, never adds net drive to the ensemble.

For _D_-dimensional output, average each output channel independently and feed each
member its per-channel deviation: _D_ output signals → a _D_-vector deviation per
member → _D_ feedback channels per member.

**Why:** The deviation couples the reservoirs through their disagreement. The
consensus mean ȳ is a lower-variance estimate of the underlying trajectory than any
single member (its error variance is ~1/3 of a single member's when the three are
independent), so routing
each member its departure from ȳ is a self-correcting error signal with **no
external teacher** — exactly what free-running lacks once the input is removed. The
coupling is mean-field / all-to-all: every member sees the same ȳ, which is the
complete-graph (K_3, a triangle) diffusive coupling of classical consensus
dynamics. Its sign
sets the regime (below).

**Mapping to the existing feedback path.** This needs no new reservoir mechanism —
it reuses the closed-loop driver already in `Reservoir`:
- Each member is an `ESN` built with `num_feedback_channels = D` (one feedback
  channel per output dimension; channels already map to contiguous vertex blocks).
- The cross-member mean/deviation is computed by an **ensemble orchestrator** that
  owns the three members and drives them in lockstep — exactly as `ESN` already owns
  `InjectInput` / `InjectFeedback` around `Step`. The reservoir stays ignorant of
  the ensemble, preserving the existing decoupling.
- `Δ_i` is staged via `InjectFeedback(c, κ·Δ_i,c)` before each member's `Step`. A
  `coupling_scaling` knob κ plays the role `feedback_scaling` plays for output
  feedback; the `tanh` clamp seam and the `state_rms` stability monitor already in
  the feedback path apply unchanged.

**Timing — one-step delay, same causality as output feedback.** `Δ_i(t)` is built
from outputs that only exist *after* each member Steps at _t_, so it is injected at
_t+1_, mirroring the established `y(t-1)` output-feedback delay:

```
   ┌──── all three members step at t ────┐
   │   ESN_1 ─► y_1(t)                    │   ȳ(t)   = (y_1 + y_2 + y_3) / 3
   │   ESN_2 ─► y_2(t)                    │   Δ_i(t) = y_i(t) − ȳ(t)
   │   ESN_3 ─► y_3(t)                    │   (Δ_1 + Δ_2 + Δ_3 = 0)
   └──────────────────┬──────────────────┘
                      ▼
   ESN_1 ◄─ InjectFeedback(c, κ·Δ_1,c(t)) ┐
   ESN_2 ◄─ InjectFeedback(c, κ·Δ_2,c(t)) ├─►  each Steps at t+1
   ESN_3 ◄─ InjectFeedback(c, κ·Δ_3,c(t)) ┘
                      │
                      ▼
             ensemble output = ȳ(t)
```

**Dynamical regimes (the sign of κ).** The two signs are **not** on equal footing.

- **−κΔ_i (pull toward the mean) — established direction.** Negative / consensus
  coupling: the linear surrogate `ẋ_i = −κ(x_i − x̄)` is contractive on the
  deviation subspace, so every member converges to the common mean ȳ at rate κ and
  the ensemble acts as a **denoiser**. Synchronizing reservoir computers through
  linear coupling is a studied effect (Hu et al. 2022, below), so this rests on
  known dynamics.
- **+κΔ_i (push away from the mean) — speculative, and not a clean mirror.** The
  *goal* — keep members diverse so ȳ keeps averaging something — is well motivated:
  ensemble diversity provably lowers error (Krogh & Vedelsby's ambiguity
  decomposition) and is actively promoted by **negative correlation learning**
  (Liu & Yao 1999). But NCL induces diversity through a **training-time loss
  penalty**, not a runtime feedback signal, and the naive sign-flip is
  **dynamically unstable**: `ẋ_i = +κ(x_i − x̄)` has positive eigenvalues on the
  deviation subspace, so deviations grow exponentially and the ensemble diverges
  unless the `tanh` clamp seam or an explicit renormalization bounds it. So "+κ for
  diversity" is a research question, not a knob with a known-good setting — strictly
  exploratory, and the load-bearing experiments below concern the −κ regime.

κ = 0 is the plain uncoupled averaging ensemble — the baseline either coupling must
beat.

**Caveat — depends on the rest of the closed-loop stack, and over-coupling collapses
the benefit.** The variance-reduction argument assumes the members' errors are at
least partly *independent*; strong consensus coupling over-synchronizes them into a
single effective reservoir, at which point ȳ averages nothing and the scheme
degenerates to one member. Member diversity (distinct `seed` / `input_scaling`) must
be preserved against the very coupling that erodes it. Like every output-feedback
ESN this is trainable only with **state noise** during training (now implemented —
see [`Reservoir.md`](Reservoir.md) training noise) and benefits from a per-channel
dv/dt slew-rate cap as a runaway guardrail. There is also a train/run
fork: training each member's readout independently (teacher-forced, coupling off)
then enabling coupling only at free-run is the conservative path; training *with*
the coupling live is more faithful but couples the members' learning too.

**Prior art (brief pass).** The pieces exist separately; the specific construction
here — runtime coupling of an ESN ensemble by feeding each member its
deviation-from-mean back through the **feedback weights** — was not found as a named
method in a moderate-effort search.
- _Consensus side (−κ) is grounded._ Linearly coupled reservoir computers
  synchronize: Hu, Zhang, Ma, Dai & Yang, "Synchronization between two linearly
  coupled reservoir computers," _Chaos, Solitons & Fractals_ 157:111882 (2022). And
  ensembles of time-delay reservoirs coupled through their feedback lines (outputs
  linearly combined) beat the single reservoir: "Reservoir Computing with an
  Ensemble of Time-Delay Reservoirs," _Cognitive Computation_ (2017). Both couple to
  **synchronize / combine**, not to repel.
- _Diversity side (+κ) is a training-time idea, not a runtime one._ Negative
  correlation learning (Liu & Yao, "Ensemble learning via negative correlation,"
  _Neural Networks_ 12(10):1399–1404, 1999) and the ambiguity decomposition (Krogh &
  Vedelsby, NIPS 1995) promote diversity via a **loss penalty during training**; in
  the ESN setting NCL has been applied to the readouts of reservoir sub-groups with
  lateral inhibition. None feeds a runtime repulsion signal back into the dynamics.
- _Closest dynamical analog._ Repulsive / anti-phase coupling in oscillator networks
  (splay and chimera states) shows runtime repulsion does produce desynchronization —
  but as a route to pattern formation, not ensemble prediction, and with the
  bounding nonlinearity doing the stabilizing work.

**Positioning.** The consensus (−κ) regime is on solid ground and is where to start.
The diversity (+κ) regime is genuinely under-explored as a runtime mechanism and
carries the linear-instability caveat above — pursue it only after −κ is characterized,
and only with the clamp/renormalization in place.

Open questions:
1. **Coupling sign & strength** — consensus (sync) vs diversity (repulsion), and the
   stability bound on κ before over-synchronization (sync) or blow-up (diversity).
2. **κ = 0 baseline** — the coupled ensemble must beat plain output averaging; that
   A/B (κ = 0 vs κ > 0, same members and seeds) is the load-bearing experiment.
3. **Member diversity** — identical reservoirs collapse to ȳ trivially;
   differently-seeded `W` / input scaling makes the deviations informative, and the
   coupling must not erase that diversity.
4. **Training protocol** — independent teacher-forced readouts with coupling added
   only at free-run, vs training with the coupling loop live.
5. **Readout topology** — shared single readout vs per-member readouts; final output
   taken as the ensemble mean either way.
6. **Interaction with state noise** — both inject perturbations during the run;
   whether consensus coupling can partly substitute for (or must be tuned against)
   the Jaeger state-noise level is open.
