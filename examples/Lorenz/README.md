# Lorenz free-run — the module, and its known structural limiters

This module trains a single `ESN` on the Lorenz-63 attractor with the Janus-cursor
pipeline (`Lorenz::Train`) and then evaluates it as a generative model
(`Lorenz::FreeRun`): past the training-window edge the future input channels switch
from real data to the model's **own prediction** (single-ESN self-feedback closed
loop), and the rollout is scored against the held-out orbit tail. The Janus-cursor
method itself is system-agnostic — dual cursors over any precomputed stream (see
`JanusCursor.md`); Lorenz-63 is simply the system it is instantiated on here.

Free-run quality on Lorenz is capped less by the one-step fit than by two *structural*
properties of the closed loop: one-step error and free-run horizon **dissociate**, so
better teacher-forced fit is not the lever. Each limiter below is about what the loop
*amplifies*.

---

## Limiter 1 — exposure bias: the teacher-forced → generative handoff shock

The trained map only ever saw reservoir states produced by *perfect* future-channel
input. At the moment the loop closes, its input distribution shifts — the future
channels now carry the model's own error — and the map's response to that perturbation
class is unconstrained by training. The result is a burst of error growth in the first
tens of steps (well above the Lyapunov rate), after which growth settles to roughly
Lyapunov-order, punctuated by orbit events and occasional re-locks.

```
err
0.2  ------------------------------------------ VPT threshold
                        .--- post-shock: ~Lyapunov-rate growth,
                       /     orbit events, occasional re-locks
       shock:         /
       tens of steps,/
       >> lambda    /
 floor -+----------'
       handoff                                          steps -->
```

### Mitigations (caller-level; the core needs no changes)

**1a. Noise injection during training** — knob: `config::TRAIN_FUTURE_NOISE` (Gaussian
stddev; `0` disables, **off by default**). Adds small zero-mean noise to the future
channels (3–5) in `Train` *after* `ExtractInputs_Training` (the FreeRun washout stays
clean). The classic ESN closed-loop regularizer: it trains the map to be insensitive to
exactly the perturbation class feedback introduces. Only channels 3–5 are noised — the
only channels that differ between training and free-run; the product channels 6–7 are
past-derived and stay consistent. Expect a modest one-step RMSE increase — that is the
point.

**1b. Scheduled sampling** — knob: `config::SCHEDULED_SAMPLING_CEILING` (probability
ceiling; `0` disables, **off by default**). With probability `p(epoch)`, substitute the
real future sample with the model's own fresh prediction (`esn_.Predict` before the
training update — the same closed-loop ordering `FreeRun` uses). `p` is a linear ramp
`0 → ceiling` over the epochs (`ScheduledSamplingProfile`). Trains on the *actual*
generative input distribution rather than a noise model of it.

**1c. Closed-loop fine-tuning epochs (not implemented).** After normal training, run a
few epochs fully closed-loop (own predictions on the future channels, still teacher
targets). Most faithful to the deployment distribution, and the most delicate: keep the
lr at the anneal floor and treat epoch-RMSE divergence as the abort signal.

Enable ONE at a time — the campaign runs single-delta experiments, and the config banner
prints the active setting (`[Lorenz config] exposure: 2a future_noise=… 2b ss_ceiling=…`).

---

## Limiter 2 — the identity burden: the readout reconstructs its own input

The readout consumes only the reservoir state x(t). There is no direct input tap —
classical ESN readouts see `[u(t); x(t); 1]`; this one sees `x(t)` alone. For Lorenz-63
at dt = 0.02 the one-step map is dominated by the identity:

```
S[f] ≈ S[f-1] + dt * F(S[f-1])
       ^^^^^^   ^^^^^^^^^^^^^^
       identity  small increment (the actual dynamics)
```

So the network's main job each step is *reproducing the previous sample* — which it can
only recover from the reservoir state, where that sample arrived scaled by
`input_scaling/sqrt(DIM)` and mixed through a tanh with everything else. Reconstruction
works, but every bit of reconstruction error lands directly in the prediction, and the
closed loop compounds precisely that error. Readout capacity is spent on copying, not on
dynamics.

### Mitigations

**2a. Train on increments (caller-level, recommended).** Change the target from the
absolute state to the one-step increment, and reconstruct outside the network:

```
training:   targets[k] = S[f].k - S[f-1].k          (ExtractTargets variant)

free-run:   esn_.Predict(delta);                    // network predicts the increment
            y[k] = prev[k] + delta[k];              // reconstruct absolute state
            feed y on the future channels; score y against S[f]; prev = y;
```

The identity component is now exact by construction; the network models only dt·F(·).
Notes: normalized increments are small, so the printed train RMSE lives on a new scale
and is not comparable to absolute-target runs; the exposure-bias handoff (Limiter 1)
still exists and the two fixes compose; VPT scoring stays on the absolute reconstruction.

**2b. Direct input tap in the readout (core-level alternative).** Append the raw input
vector to the flattened feature vector the linear readout sees (HCNN's readout is already
a flatten, so this is a width change plus plumbing the input through `Readout`). Gets the
classical `[u; x]` skip connection. More invasive than 2a for the same benefit.

---

## Evaluation caution

VPT can be gated by fixed decoherence events in the *one* eval orbit rather than by model
quality. Any improvement claimed for a mitigation must be measured across varied eval
segments (different `INITIAL_LORENZ_STATE` or window placement), or it may just be
re-measuring the same orbit's event structure.
