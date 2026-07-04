# Lorenz Training Campaign Log

Experiment record for `Lorenz::Train()` — online, teacher-forced training of an
`EnsembleESN` on the Lorenz-63 attractor via the Janus-cursor pipeline. One section
per run, each recording the config delta, the headline numbers, and the finding the
run established. Free-run (generative) evaluation does not exist yet; every metric
here is train-side.

The first campaign (6 runs on the old 10000-sample / span-5000 geometry) is retired;
see git history for its record. Its durable conclusions: online single-sample Adam
training is stable and its error floor tracks lr roughly linearly; annealing from a
high lr buys nothing over starting at the floor; and in a controlled kappa-0 A/B at
low lr, the uncoupled ensemble had the lower train RMSE.

## Findings at a glance

| # | run | config | final RMSE | finding |
|---|-----|--------|-----------|---------|
| 1 | new baseline | see shared setup | **0.001412** | best one-step error to date; noise-free late curve |

## Shared setup

`DIM=8` (N=256), `SEED=7673895`, `SPECTRAL_RADIUS=0.90`, `INPUT_SCALING=0.05`,
`LEAK_RATE=1.0`, `HISTORY_DEPTH=8`, tanh arm (`LORENTZ_GAMMA=0`), M=3 members,
`Combine::Mean`, Adam readout updates, kappa 0 (coupling off).

Lr schedule: 1e-4 → 2e-5, hold 1 (cosine from epoch 2), 200 epochs.

Stream: 20000+1 samples at dt=0.02, normalized per channel to [-1, 1].
Cursor window [4000, 19000] (center 11500, span 15000), 100 warmup steps per
epoch, 14901 training steps per epoch (targets S[4100..19000]). Evaluation
runway E = 1000 steps past the window — 20 time units ≈ 18 Lyapunov times of
Lorenz-63, the hard cap on any future free-run VPT measurement.

### Metrics

- **train RMSE** — *prequential* (test-then-train): `outputs` from `EnsembleESN::Step`
  is the consensus read at x(t) *before* that call's readout update, i.e. the honest
  pre-update prediction of that same call's target. RMS over 3 channels x 14901 steps,
  normalized units.
- **dev[i]** — member i's epoch-RMS of its raw consensus deviation `y_i - c`
  (pre-kappa; exactly the signal the coupling would scale into feedback).
- **sd** — population std of the three dev values: the *asymmetry between members*.

---

## Run 1 — new baseline (fell out of parameter-space exploration)

Config: the shared setup above — arrived at by exploration, not a controlled
delta from the retired campaign (seed, input scaling, history depth, window
geometry, lr floor, and epoch count all changed at once).

Result: final RMSE **0.001412 at epoch 199 — best one-step error to date**
(the retired campaign's best was 0.00284, on 1/3 the window). Trajectory:
0.116 @0 → 0.0136 @1 → 0.0047 @20 → 0.0034 @50 → 0.0022 @100 → 0.001412 @199.
Still descending ~0.2%/epoch at the 2e-5 floor — the lr-noise floor remains
unreached even here.

Findings:
- **The late curve is essentially noise-free.** From ~epoch 90 the descent is
  monotone to the 6th decimal and the dev spread is frozen (sd ≈ 3.9e-5 for a
  hundred straight epochs). Between the tiny lr and 3x more steps averaged per
  epoch, the single-sample gradient noise that defined the first campaign's
  curves is gone; what remains looks like clean deterministic convergence.
- No instability of any kind across 200 epochs (kappa 0, so none expected).

---

## Open questions (all blocked on the free-run/eval stage)

1. **Free-run VPT** — the whole point. This baseline is what the generative
   stage gets built against. Runway caps measurement at 18 Lyapunov times.
2. **Does kappa pay for itself in free-run?** Train-side the kappa-0 arm won the
   retired campaign's A/B; the coupled arm must beat this kappa=0 baseline's
   VPT to justify itself.
3. **Lr floor still unreached** at 2e-5; lower floor / longer decay has visible
   headroom if free-run demands a lower one-step error.
