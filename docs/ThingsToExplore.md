# Closed-loop - Areas of Exploration

A running list of exploratory ideas — not approved work, just captured directions
worth investigating.

## 1. Free Running

_Closed-loop / autonomous generative operation — the readout drives the reservoir
forward with no external input (e.g. Lorenz free-run)._

### Cap output dv/dt

**Maturity:** _Heuristic_ — an engineering stabilization trick, not part of the
canonical ESN recipe.

Clamp the step-over-step change of each generated output channel to the largest
dv/dt that channel should physically experience, before feeding it back into the
reservoir. The cap is **per output channel**, not a single global bound — each
channel _c_ carries its own limit `Δmax_c`, since different signals (e.g. the
x/y/z components of Lorenz) have different natural rates of change and a shared
bound would either over-constrain the fast channels or under-constrain the slow
ones.

**Why:** In closed-loop free-running the readout feeds its own output back, so a
single spurious spike can compound and blow up the trajectory. Bounding each
channel's per-step rate of change keeps every generated signal on its own
plausible manifold and buys stability without retraining.

**How to apply:** For each output channel _c_, compute the largest |dv/dt| seen in
that channel's training target → `Δmax_c`. Before feeding `y_c(t)` back, clamp
`y_c(t) - y_c(t-1)` to ±`Δmax_c`. Maintain the per-channel `Δmax` as a small vector
sized to the number of outputs, and apply it in the closed-loop step where the
output is routed back as the next driver.

**Timing — correct the current step, not the next.** We have full control over
time here and can reference backward (the previous sample is retained), so the cap
need not be framed as a feed-forward limiter on what gets injected at _t+1_.
Instead, treat it as a correction of the **current** output: keep `y_c(t-1)`,
compute the realized `dv/dt` at _t_, and if it is out of range massage `y_c(t)`
itself back inside ±`Δmax_c`. The corrected value then becomes the authoritative
`y_c(t)` — what we report, what we feed back, and what `y_c(t+1)` is measured
against — so there is no divergence between "what we emitted" and "what we routed
back."

```
  retain y(t-1) ──┐
                  ▼
  readout ─► y_raw(t) ─► Δ = y_raw(t) − y(t-1)
                          │
                  |Δ| > Δmax ?  ── no ──► y(t) = y_raw(t)
                          │ yes
                          ▼
                  y(t) = y(t-1) ± Δmax   (massage current sample)
                          │
                          ▼
              authoritative y(t): reported AND fed back
```

**Caveat — guardrail, not an accuracy mechanism.** This is a slew-rate limiter: it
stops blow-up but does nothing for correctness. The cause of out-of-range `dv/dt`
in free-run is almost always missing regularization (→ state noise), a too-hot
spectral radius, or a readout overfit — fix those, not the symptom. For Lorenz
(scored on λ / Lyapunov horizon) a slew-limited runaway is still off the
attractor: bounding `dv/dt` does **not** keep the trajectory on the manifold, so
do not expect this to move λ. There is also a decoupling risk — when `y(t)` is
overwritten, the reservoir's internal state still evolved from the un-clamped
dynamics, so the loop is fed a value its own state does not agree with, which can
push things off-manifold in its own way. Keep it as a NaN/explosion safety clamp
while the real dynamics are fixed elsewhere.

Open questions:
1. **Bound estimation** — per channel: largest training |dv/dt| vs a percentile
   (robust to outliers) vs a hand-set physical limit.
2. **Hard vs soft clamp** — hard clip introduces kinks; a soft (tanh/saturating)
   limiter is smoother but adds a nonlinearity to tune.
3. **Coupling across channels** — independent per-channel caps ignore cross-channel
   structure; whether a joint constraint (e.g. on the output vector's velocity
   norm) is ever needed is open.

## 2. General

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
