# Closed-loop - Areas of Exploration

A running list of exploratory ideas — not approved work, just captured directions
worth investigating.

## 1. Free Running

_Closed-loop / autonomous generative operation — the readout drives the reservoir
forward with no external input (e.g. Lorenz free-run)._

### Cap output dv/dt

**Maturity:** _Heuristic_ — an engineering stabilization trick, not part of the
canonical ESN recipe.

Clamp the step-over-step change of the generated output to the largest dv/dt the
output should physically experience before feeding it back into the reservoir.

**Why:** In closed-loop free-running the readout feeds its own output back, so a
single spurious spike can compound and blow up the trajectory. Bounding the
per-step rate of change keeps the generated signal on a plausible manifold and
buys stability without retraining.

**How to apply:** Compute the largest |dv/dt| seen in the training target and use
it as a cap; before feeding y(t) back, clamp `y(t) - y(t-1)` to that bound. Hook
it into the closed-loop step where the output is routed back as the next driver.

Open questions:
1. **Bound estimation** — largest training |dv/dt| vs a percentile (robust to
   outliers) vs a hand-set physical limit.
2. **Hard vs soft clamp** — hard clip introduces kinks; a soft (tanh/saturating)
   limiter is smoother but adds a nonlinearity to tune.
3. **Per-channel vs global** — independent caps per output dimension vs one
   shared bound.

## 2. General

### Add noise

**Maturity:** _Standard_ — canonical ESN practice (Jaeger); essentially mandatory
for output-feedback / generative reservoirs. Adopt, don't just explore.

Inject small random perturbations into the reservoir state each step during
training (and optionally warmup) — a standard RC regularizer (Jaeger's state
noise).

**Why:** In closed-loop free-running the readout is trained on teacher-forced
(clean) states but must run on its own slightly-wrong predictions, so tiny errors
compound. Injecting state noise during training forces the readout to learn a
contraction back toward the attractor from off-trajectory states — improving
stability and generalization. It is the mechanism that makes output-feedback ESNs
trainable, and directly addresses the feedback stability concern (feedback voids
the echo-state property without it).

**How to apply:** Per-neuron additive noise in `UpdateState` — sample
ε ~ N(0, σ²) (or uniform) and add to each neuron each step, gated by a
`noise_scaling` knob. Reuse the existing per-neuron bias plumbing for the add.
Active during `Warmup`/training, off (or reduced) at inference.

Open questions:
1. **Injection site** — pre-nonlinearity (state noise) vs post-nonlinearity
   (output noise); Jaeger uses state noise.
2. **Schedule** — constant σ vs annealed; on during warmup or training-only.
3. **Interaction with SR / leak rate** — too much noise washes out memory; needs
   tuning against `EstimateSpectralRadius` and the leak knob.

### Ensemble consensus feedback

**Maturity:** _Novel_ — ESN ensembles (averaging members) are established, but
feeding each member its deviation from the ensemble mean as a coupling signal is
not standard procedure.

Run an ensemble of ESNs (start with three) in parallel on the same task. At each
step compute the ensemble mean of their outputs, then feed back to each member the
deviation of its own output from that mean — i.e. ESN _i_ receives
`y_i − ȳ`, where `ȳ = (1/N) Σ y_i`. A member running above consensus gets a
positive deviation signal, one below gets a negative one; a member exactly at the
mean gets zero.

For multi-dimensional output, average each output channel across the ensemble
independently and feed each member its per-channel deviation. So _D_ output
signals produce a _D_-dimensional deviation vector per ESN, and each component
becomes its own feedback channel — this is how multiple feedback channels are
supported.

**Why:** The deviation couples the reservoirs through their disagreement. The
consensus mean ȳ is a lower-variance estimate of the underlying trajectory than
any single member, so routing each member its own departure from ȳ gives a
self-correcting error signal with no external teacher — exactly what free-running
lacks. Depending on the feedback sign this either drives **synchronization**
(members converge, ensemble acts as a denoiser whose mean is the output) or
**diversity** (members repel to cover more of the dynamics).

**How to apply:** After all _N_ members Step at time _t_, compute the per-channel
mean, form each member's deviation `Δ_i(t) = y_i(t) − ȳ(t)`, and inject it as that
member's feedback at _t+1_ (one-step delay, same causality as output feedback). A
`coupling_scaling` knob sets strength; its sign selects consensus vs diversity.
The ensemble's final output is the mean ȳ.

Open questions:
1. **Coupling sign & strength** — convergence (sync) vs divergence (diversity),
   and the stability bound on `coupling_scaling`.
2. **Member diversity** — identical reservoirs collapse to ȳ trivially;
   differently-seeded `W` / input scaling makes the deviations informative.
3. **Ensemble size _N_** — three is the minimum for a meaningful mean; diminishing
   returns vs cost beyond that.
4. **Readout topology** — shared single readout vs per-member readouts; final
   output taken as the ensemble mean.
