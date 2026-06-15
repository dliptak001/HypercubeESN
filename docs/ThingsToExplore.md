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

Run an ensemble of _N_ ESNs (start with three) in parallel on the same task, each
with differently-seeded weights. At each step compute the ensemble mean of their
outputs `ȳ = (1/N) Σ y_i`, then feed each member _i_ a scaled copy of its own
deviation from that mean, `Δ_i = y_i − ȳ`. A member running above consensus gets a
positive deviation, one below a negative one, one exactly at the mean gets zero.
The deviations are conservative — `Σ_i Δ_i = 0` by construction — so the coupling
only redistributes drive among members, never adds net drive to the ensemble.

For _D_-dimensional output, average each output channel independently and feed each
member its per-channel deviation: _D_ output signals → a _D_-vector deviation per
member → _D_ feedback channels per member.

**Why:** The deviation couples the reservoirs through their disagreement. The
consensus mean ȳ is a lower-variance estimate of the underlying trajectory than any
single member (error variance falls ~1/N when members are independent), so routing
each member its departure from ȳ is a self-correcting error signal with **no
external teacher** — exactly what free-running lacks once the input is removed. The
coupling is mean-field / all-to-all: every member sees the same ȳ, which is the
complete-graph (K_N) diffusive coupling of classical consensus dynamics. Its sign
sets the regime (below).

**Mapping to the existing feedback path.** This needs no new reservoir mechanism —
it reuses the closed-loop driver already in `Reservoir`:
- Each member is an `ESN` built with `num_feedback_channels = D` (one feedback
  channel per output dimension; channels already map to contiguous vertex blocks).
- The cross-member mean/deviation is computed by an **ensemble orchestrator** that
  owns the _N_ members and drives them in lockstep — exactly as `ESN` already owns
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
   ┌── members step at t ──┐
   │  ESN_1 ─► y_1(t)      │
   │  ESN_2 ─► y_2(t)      │   ȳ(t)   = mean_i y_i(t)
   │   ...                 │   Δ_i(t) = y_i(t) − ȳ(t)        (Σ_i Δ_i = 0)
   │  ESN_N ─► y_N(t)      │
   └──────────┬────────────┘
              ▼
   each member i:  InjectFeedback(c, κ·Δ_i,c(t))  ──►  Step at t+1
              │
              ▼
   ensemble output = ȳ(t)
```

**Dynamical regimes (the sign of κ).** Injecting **−κΔ_i** (pull each member toward
the mean) is negative / consensus coupling: the linear surrogate `ẋ_i = −κ(x_i − x̄)`
drives every member to the common mean at rate κ, so the ensemble acts as a
**denoiser** whose output is ȳ. Injecting **+κΔ_i** (push away from the mean) is
**diversity** coupling: members repel to cover more of the dynamics, useful when a
single attractor estimate is too narrow. κ = 0 is the plain uncoupled averaging
ensemble — the baseline any coupling must beat.

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

Open questions:
1. **Coupling sign & strength** — consensus (sync) vs diversity (repulsion), and the
   stability bound on κ before over-synchronization (sync) or blow-up (diversity).
2. **κ = 0 baseline** — the coupled ensemble must beat plain output averaging; that
   A/B (κ = 0 vs κ > 0, same members and seeds) is the load-bearing experiment.
3. **Member diversity** — identical reservoirs collapse to ȳ trivially;
   differently-seeded `W` / input scaling makes the deviations informative, and the
   coupling must not erase that diversity.
4. **Ensemble size _N_** — three is the minimum for a meaningful mean; diminishing
   returns vs N× compute beyond that.
5. **Training protocol** — independent teacher-forced readouts with coupling added
   only at free-run, vs training with the coupling loop live.
6. **Readout topology** — shared single readout vs per-member readouts; final output
   taken as the ensemble mean either way.
7. **Interaction with state noise** — both inject perturbations during the run;
   whether consensus coupling can partly substitute for (or must be tuned against)
   the Jaeger state-noise level is open.
