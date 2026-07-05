# Lorenz free-run — known structural limiters and options for resolution

This module trains an `EnsembleESN` on the Lorenz-63 attractor with the Janus-cursor
pipeline (`Lorenz::Train`) and then evaluates it as a generative model
(`Lorenz::FreeRun`): the future input channels switch from real data to the ensemble's
own consensus, and the rollout is scored against the held-out orbit tail. The
Janus-cursor method itself is system-agnostic — dual cursors over any precomputed
stream (see `JanusCursor.md`); Lorenz-63 is simply the system it is instantiated on
here. `TrainingLog.md` is the run-by-run experimental record.

Campaign status when this document was written: best VPT **4.56 λt** (one seed, one
orbit), seed-sweep median **~2.3 λt**, one-step (teacher-forced) train RMSE in the
**5×10⁻⁴ – 9×10⁻⁴** range. A line-by-line audit of the core stack
(`Reservoir`, `ESN`, `Readout`/HCNN, `EnsembleESN`) found **no implementation bugs** —
the spectral-radius solve runs on the correct augmented delay-line operator, the
backward math and Adam are right, train/inference paths are symmetric, and the
horizon-1 prequential contract holds end to end. What the audit did find are three
*structural* properties that plausibly cap the free-run numbers. They are documented
here so the next experiments target the actual constraints.

One framing fact before the issues: the seed sweep showed that one-step error and
free-run quality **dissociate** (the ablation arm trained to 0.000521 and got *half*
the control's VPT). Better teacher-forced fit is not the lever. Each issue below is
about something else: what the loop *amplifies*, and what the ensemble mechanism can
and cannot correct.

---

## Issue 1 — the consensus coupling is structurally blind to the error that kills free-run

### The mechanism, exactly

Every `EnsembleESN::Step`, each member i receives on its feedback channels

```
phi_i = kappa * (y_i - c),      c = mean_i(y_i)     (Combine::Mean)
```

Summing over members:

```
sum_i phi_i = kappa * (sum_i y_i - M*c) = 0        -- identically, at any kappa
```

Decompose each member's error against the truth `t` into a **shared** part and a
**disagreement** part:

```
y_i - t   =   (c - t)      +      (y_i - c)
              ^^^^^^^^^           ^^^^^^^^^
              shared error        disagreement
              (same for all)      (sums to zero)

phi sees:     NOTHING             everything
```

The coupling is a pure disagreement-shaping device. Any error component all members
have in common — including the entire "we drifted off the true orbit" component —
passes through it untouched, at any coupling strength.

### Why this bites free-run specifically

In free-run all M members are driven by the **same** consensus on the future input
channels, so they leave the true orbit *together*. The printed diagnostics show the
scale separation directly: end-of-training member deviations (dev[]) sit near
**9×10⁻⁴** RMS while the free-run error grows to **~0.5** — the signal φ can act on
is three orders of magnitude smaller than the error that matters. Consequently:

- Sweeping rollout κ (open question #2 in the log) is expected to be near-flat.
  Running that sweep is still worthwhile *as the confirmation*.
- The ensemble, as currently wired, earns its keep as an output-averaging /
  variance-reduction device — which is real, but orthogonal to VPT.

### Options for resolution

**1a. Per-member closed loop (the interesting one).** During the rollout, feed each
member its **own** prediction on the future channels instead of the consensus, and
use the consensus only as the reported output:

```
current:                              proposed:
            +--> member 1 --+                    +--> member 1 --+   y_1 feeds
consensus --+--> member 2 --+--> c    y_i (own) -+--> member 2 --+-> back to
            +--> member 3 --+   |                +--> member 3 --+   member i only
      ^                         |                        |
      +-------------------------+          consensus c reported; phi_i = k(y_i - c)
```

Members then follow genuinely divergent trajectories, the disagreement term becomes
*large*, and κ-coupling turns into a meaningful synchronization force pulling M
independent generative rollouts toward each other. This converts the ensemble from an
averaging device into a trajectory-consensus device — exactly the regime the coupling
was designed for. Caller-level change: an `ExtractInputs_FreeRun` variant that takes
`MemberOutput(i)` instead of the consensus, plus a per-member input in the step loop
(the `EnsembleESN` API would need a per-member-input `Step` overload, or the loop can
drive members individually). Risks: members can blur the consensus if they de-phase
(median combine mitigates), and it changes what "the ensemble's prediction" means
mid-rollout.

**1b. Diversify the members.** Members currently differ only by reservoir seed; their
free-run errors are therefore highly correlated (same architecture, same drive, same
learned map class). Giving members different `spectral_radius` / `input_scaling` /
`history_depth` decorrelates their error processes, which shrinks the shared component
that averaging cannot fix and enlarges the part it can. Requires relaxing the
"identical base config" constraint in `EnsembleConfig` (core change, currently §5 of
the design), or building M single ESNs at the caller level and averaging outside.

**1c. Accept it, and stop paying for it.** If the ensemble is kept as a variance
device, run the κ=0 rollout control to confirm flatness and stop spending sweep time
on rollout-κ. M=3 averaging still buys a small, honest output-noise reduction.

---

## Issue 2 — exposure bias: the teacher-forced → generative handoff shock

### What the traces show

The trained map only ever saw reservoir states produced by *perfect* future-channel
input. At the moment the loop closes, its input distribution shifts — the future
channels now carry the model's own error — and the map's response to that
perturbation class is completely unconstrained by training. The rollout traces
localize the damage (all numbers program-printed, from the seed-sweep runs):

- Control arm (seed 7673895): one-step error 0.000932 → **0.0257 by rollout step 25**
  (~28× in 25 steps, ≈7× the Lyapunov rate).
- Ablation arm: 0.000521 → **0.0292 by step 25** (~56×).
- After the shock, growth drops to roughly Lyapunov-order or slower, punctuated by
  orbit events and re-locks.

```
err
0.3  ------------------------------------------ VPT threshold
                        .--- post-shock: ~Lyapunov-rate growth,
                       /     orbit events, occasional re-locks
       shock:         /
       ~25-50 steps, /
       ~7x lambda   /
5e-4 --+-----------'
       handoff                                          steps -->
```

For scale (arithmetic on printed numbers, not a program metric): a rollout whose
error grew at exactly the Lyapunov rate from a 9.3×10⁻⁴ floor would cross 0.3 at
ln(0.3/0.000932) ≈ **5.8 λt**. The best observed run reached 4.56 λt; the median
runs lose most of the gap in the first ~50 steps. The shock, not the asymptotic
growth rate, is the binding constraint.

### Options for resolution

All three are caller-level; the core needs no changes.

**2a. Noise injection during training (cheapest, do first).** Add small zero-mean
noise to the future channels in `ExtractInputs_Training`. This is the classic ESN
regularizer for closed-loop use: it trains the map to be insensitive to exactly the
perturbation class that feedback introduces. Implementation is a few lines (one RNG,
three channels — remember the derived product channel must be computed *from the
noised values*, or the perturbation is inconsistent). The one hyperparameter is the
noise scale; the natural starting bracket is the observed early-rollout error
(≈10⁻³ to a few 10⁻²). Expect a modest one-step RMSE increase — that is the point,
and the run-5 dissociation says one-step RMSE was not the target anyway.

**2b. Scheduled sampling.** With probability p(epoch), replace the real future sample
with the ensemble's own fresh prediction (`EnsembleESN::Predict` before the training
`Step` — the same closed-loop ordering `FreeRun` already uses). Ramp p from 0 toward
some ceiling over the epochs. This trains on the *actual* generative input
distribution rather than a noise model of it. Costs one extra forward per exposed
step; the schedule (ramp shape, ceiling) is new tuning surface.

**2c. Closed-loop fine-tuning epochs.** After normal training, run a few epochs fully
closed-loop (own predictions on the future channels, still teacher targets). Most
faithful to the deployment distribution, and the most delicate: early in a closed-loop
epoch the inputs are off-manifold, so gradients can chase transients. Keep the lr at
the anneal floor and treat divergence of the epoch RMSE as the abort signal.

**Evaluation caution that applies to all of the above:** the seed sweep showed VPT is
currently gated by fixed decoherence events in the *one* shared eval orbit (five of
six runs died at steps 92–130). Any improvement claimed for 2a/2b/2c must be measured
across varied eval segments (different `INITIAL_LORENZ_STATE` or window placement),
or it may just be re-measuring the same orbit gate.

---

## Issue 3 — the identity burden: the readout reconstructs its own input

### The architectural fact

The readout consumes only the reservoir state x(t). There is no direct input tap —
classical ESN readouts see `[u(t); x(t); 1]`; this one sees `x(t)` alone. For
Lorenz-63 at dt = 0.02 the one-step map is dominated by the identity:

```
S[f] ≈ S[f-1] + dt * F(S[f-1])
       ^^^^^^   ^^^^^^^^^^^^^^
       identity  small increment (the actual dynamics)
```

So the network's main job each step is *reproducing the previous sample* — which it
can only recover from the reservoir state, where that sample arrived scaled by
`input_scaling/sqrt(DIM)` ≈ 0.018 per weight and mixed through a tanh with everything
else. The training numbers prove the reconstruction works (5×10⁻⁴ RMSE), but every
bit of reconstruction error lands directly in the prediction, and the closed loop
compounds precisely that error. Readout capacity and precision are being spent on
copying, not on dynamics.

### Options for resolution

**3a. Train on increments (caller-level, recommended).** Change the target from the
absolute state to the one-step increment, and reconstruct outside the network:

```
training:   targets[k] = S[f].k - S[f-1].k          (ExtractTargets variant)

free-run:   esn_.Predict(delta);                    // network predicts the increment
            y[k] = prev[k] + delta[k];              // reconstruct absolute state
            feed y on the future channels; score y against S[f]; prev = y;
```

The identity component is now exact by construction; the network models only
dt·F(·). Notes: (i) normalized increments are small (the per-step change of a
normalized channel at dt=0.02), so the loss scale drops — Adam's per-parameter
normalization absorbs most of this, but the printed train RMSE will live on a new
scale and is not comparable to previous runs; (ii) during teacher-forced training
`prev` is the real S[f-1], during free-run it is the previous reconstruction — the
handoff shock of Issue 2 still exists, the two fixes are independent and compose;
(iii) VPT scoring stays on the absolute reconstruction, so free-run metrics remain
comparable.

**3b. Direct input tap in the readout (core-level alternative).** Append the raw
input vector to the flattened feature vector the linear readout sees (HCNN's readout
is already a flatten — `input_channels = c_final * N_final` — so this is a width
change plus plumbing the input through `Readout`). Gets the classical `[u; x]` skip
connection. More invasive than 3a for the same benefit; only worth it if increments
prove awkward for some future task shape.

---

## Priority and interaction

The three issues are independent and their fixes compose. Suggested order of attack,
cheapest-per-expected-λt first:

1. **2a (noise injection)** — few lines, directly targets the binding constraint
   (the handoff shock).
2. **3a (increment targets)** — small caller change, removes the identity burden the
   loop amplifies.
3. **1a (per-member closed loop)** — the substantive ensemble experiment; it is the
   only option that gives the consensus coupling a job free-run actually needs done.

And in parallel, fix the measurement: vary the eval orbit (x0 / window placement) so
VPT reflects the model rather than one trajectory's event structure — without that,
none of the above can be honestly scored.
