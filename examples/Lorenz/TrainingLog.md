# Lorenz Training Campaign Log

Experiment record for `Lorenz::Train()` — online, teacher-forced training of an
`EnsembleESN` on the Lorenz-63 attractor via the Janus-cursor pipeline. One section
per run, each recording the config delta, the headline numbers, and the finding the
run established. Free-run (generative) evaluation does not exist yet; every metric
here is train-side.

## Findings at a glance

| # | run | config delta | final RMSE | finding |
|---|-----|--------------|-----------|---------|
| 1 | constant lr | lr 0.0015 flat | ~0.045 (plateau) | stable (no divergence); lr-noise floor; barely beats persistence |
| 2 | lr anneal | cosine 0.0015 → 1e-4, hold 100 | 0.00388 | RMSE tracks lr ~linearly; 13.8x below persistence |
| 3 | early anneal + dev metrics | hold 100 → 25 | 0.00415 | hold length immaterial; dev/RMSE ≈ 0.80 constant |
| 4 | strong coupling | kappa ceiling 0.2 → 3.0 (k 50 → 25) | **0.00354** | dev/RMSE invariant under 15x kappa; kappa *symmetrizes* members; κ·lr stability boundary |

## Shared setup

All runs: `DIM=8` (N=256), `SEED=673895`, `SPECTRAL_RADIUS=0.90`, `INPUT_SCALING=0.10`,
`LEAK_RATE=1.0`, `HISTORY_DEPTH=16`, tanh arm (`LORENTZ_GAMMA=0`), M=3 members,
`Combine::Mean`, Adam readout updates, `WEIGHT_DECAY=0`.

Stream: 10000+1 samples at dt=0.02, normalized per channel to [-1, 1]
(window channel std ≈ 0.40 / 0.33 / 0.36). Cursor window [1500, 6500]
(center 4000, span 5000), 100 warmup steps per epoch, 600 epochs,
4901 training steps per epoch (targets S[1600..6500]).

Kappa schedule (caller-managed, set per epoch before warmup — coupled warmup is
deliberate): `kappa(x) = kappa_max * k*x^2 / (1 + k*x^2)`, x = epoch/epochs.
Saturating ramp, strictly below `kappa_max`.

### Metrics

- **train RMSE** — *prequential* (test-then-train): `outputs` from `EnsembleESN::Step`
  is the consensus read at x(t) *before* that call's readout update, i.e. the honest
  pre-update prediction of that same call's target. RMS over 3 channels x 4901 steps,
  normalized units.
- **dev[i]** — member i's epoch-RMS of its raw consensus deviation `y_i - c`
  (pre-kappa; exactly the signal the coupling scales into feedback).
- **sd** — population std of the three dev values: the *asymmetry between members*.

These partition each member's error exactly (mean consensus ⇒ deviations sum to 0):

```
avg member MSE   =   consensus MSE   +   avg deviation MSE
  (vs truth)          (RMSE^2)             (dev^2)

 shared error: what all members get wrong together — averaging can't remove it
 deviation:    idiosyncratic disagreement — cancels in the consensus mean
```

### Baseline

**Persistence** (predict S[f] by copying S[f-1], which sits on input channels 3-5):
RMSE **0.053406** over the same 4900 targets. Any model result must be judged
against this — it is what "just echo the input" scores.

---

## Run 1 — constant lr 0.0015

Config: lr flat 0.0015, kappa ceiling 0.2 (k=50).

Result: RMSE 0.103 → ~0.045 by epoch ~90, then **flat for 500 epochs**
(jitter ±0.006, best 0.032). No kappa inflection anywhere on the 0 → 0.196 ramp.

Findings:
- **No divergence.** The prior harness diverged at this exact lr under single-sample
  online updates; the rebuilt pipeline (Adam + per-epoch reset/warmup + horizon-1
  target alignment) is stable. Retired the "single-sample online is structurally
  inadequate" hypothesis.
- **The plateau is barely above copying**: 0.045 vs persistence 0.0534 — only ~16%
  better. Relative error ~12% of channel std. ~10x short of the ~0.004 one-step
  level that closed-loop generation is expected to need.
- Diagnosis: **lr-noise floor** — weights orbit the optimum with variance set by the
  (constant) learning rate. Budget was not the limit; 500 flat epochs prove it.

## Run 2 — cosine lr anneal (hold 100)

Config delta: `LrProfile` added — hold lr 0.0015 through epoch 100, then
`CosineLR` down to 1e-4 at the final epoch.

Result: RMSE *tracked the lr curve* nearly monotonically:
0.045 (plateau) → 0.027 @300 → 0.0124 @400 → 0.0065 @500 → **0.00388 @599**.
Epoch-to-epoch jitter collapsed alongside lr.

Findings:
- **Noise-floor diagnosis confirmed.** 15x lr reduction bought 11.6x error
  reduction — the floor scales roughly linearly with lr.
- Final error is 13.8x below persistence and matches the old batch pipeline's
  one-step level (~0.0037): online single-sample training now equals batch.
- Curve still descending at epoch 599 → headroom exists below 1e-4 if needed.

## Run 3 — earlier decay + per-member deviation metrics

Config delta: hold 100 → 25 (shallower cosine over epochs 26-599).
Instrumentation delta: `dev[]`/`sd` columns added.

Result: final RMSE 0.00415 (vs 0.00388) — hold length is immaterial; both runs
bottom at the lr-floor.

Findings:
- **dev/RMSE ≈ 0.80, constant** across the entire two-orders-of-magnitude descent
  (0.92 @0 → 0.81 @100 → 0.79 @300 → 0.85 @599). Diversity falls in lockstep with
  accuracy: members never homogenize, never scatter. Each member individually errs
  ~1.28x the consensus throughout — a sustained ~24% ensemble averaging gain.
- **Persistent member hierarchy** at low kappa (ceiling 0.2): from ~epoch 450,
  `dev[0] > dev[1] > dev[2]` every epoch (0.00383/0.00363/0.00316 at 599;
  sd/mean ≈ 8%). Member 0 is the standing outlier — seed-driven, structural.

## Run 4 — strong coupling (kappa ceiling 3.0, k=25)

Config delta: kappa ceiling 0.2 → 3.0, ramp steepness k 50 → 25.
Final kappa reached 2.884 — feedback magnitude ~15x run 3.

Result: final RMSE **0.003535 — best of the campaign** (margin vs run 2 is modest;
kappa is not credited with the improvement, but it demonstrably cost nothing).

Findings:
- **dev/RMSE ≈ 0.81, invariant again** — under 15x the homogenizing pressure.
  Two-run conclusion: the diversity *magnitude* is set by seed differences and
  gradient noise, not by kappa, at any strength tested.
- **Kappa symmetrizes, it does not compress.** The across-member spread collapsed:
  sd/mean(dev) 7.9% (run 3) → **2.6%** (run 4) at epoch 599 (0.000279 → 0.000074
  absolute); run 3's member hierarchy is essentially erased. Mechanism: the
  feedback `phi_i = kappa*(y_i - c)` is a restoring force proportional to each
  member's *own* deviation — the biggest deviator gets the biggest corrective
  kick, equalizing deviation magnitudes while their common level persists.
  `sd/mean(dev)` is therefore the one train-side dial that visibly responds to
  kappa — useful for future kappa sweeps without waiting on free-run results.
- **A kappa x lr stability boundary, mapped for free.** Epochs ~119-262
  (kappa 1.5 → 2.5 while lr 0.0014 → 0.001) show recurrent self-healing
  instability events: RMSE spikes to 0.04-0.075 with one member's deviation
  blowing out (e.g. ep131: member dev 0.097, sd 0.017), recovery within 1-2
  epochs, and — curiously — each spike followed by an anomalously *good* epoch
  (0.013-0.017). Quiet above and below the band. Rough boundary:
  **kappa * lr ≲ 2.5e-3** during the high-lr phase. If the ceiling ever rises
  further, co-schedule the ramp and the anneal to keep the product under the
  boundary. The self-healing itself is evidence the consensus pull works as
  designed. (Free-run involves no lr, so the band should not constrain
  generation-time kappa.)

---

## Open questions (all blocked on the free-run/eval stage)

1. **Does kappa buy free-run stability?** Train-side, kappa 0.2 and 2.9 are
   indistinguishable in accuracy and diversity magnitude; only the member symmetry
   differs. Decisive experiment: free-run VPT at kappa 2.9 vs a kappa=0 control.
2. **Is 0.004 one-step error enough?** The old batch heuristic says roughly yes;
   only closed-loop compounding can confirm.
3. **Lower lr floor?** Run 2/4 curves still descending at 1e-4; ~3e-5 with a longer
   decay has visible headroom if free-run demands a lower one-step floor.
4. **Warmup length** (100) is adequate at SR 0.90 / current error floor; revisit
   (~200) together with any SR increase or sub-1e-3 error targets — washout need
   scales like 1/(1-rho).
