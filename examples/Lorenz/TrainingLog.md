# Lorenz Training Campaign Log

Experiment record for `Lorenz::Train()` + `Lorenz::FreeRun()` — online, teacher-forced
training of an `EnsembleESN` on the Lorenz-63 attractor via the Janus-cursor pipeline,
followed by a generative rollout scored against the held-out tail. One section per run,
each recording the config delta, the headline numbers, and the finding the run
established. Run 1 predates the free-run stage; runs 2+ include a rollout.

## Findings at a glance

| # | run | config delta | train RMSE @199 | VPT | finding |
|---|-----|--------------|-----------------|-----|---------|
| 1 | new baseline | see shared setup (old geometry) | 0.001412 | — | best one-step to date at the time; noise-free late curve |
| 2 | first free-run | + FreeRun(); kappa ceiling 0.01 | 0.001416 | 0.67 λt | rollout stays bounded + on-attractor (no runaway, no collapse); phase lost fast |
| 3 | wider window, hot kappa rollout | window 16000 / E=2000; rollout kappa 0.1 | 0.001405 | ~1.4–1.8 λt* | *0.3-crossing between steps 75–100; summary lost to anchor-floor crash (guard added) |
| 4 | distance channel dropped | products replace distance; kappa 0.05 | 0.000932 | **4.56 λt** | goal-level VPT; the distance ramp was structurally OOD in free-run |
| 5 | past-anchor ablation | past channels = future copy | 0.000521 | 2.35 λt | anchor buys ~2x VPT while *costing* 1.8x train RMSE — one-step error is not a free-run proxy |
| 6 | seed sweep, 3 seeds x 2 arms | SEED ∈ {7673895, 43434334, 84368334} | see table | medians 2.34 / 2.30 λt | anchor's 2x VPT does NOT replicate (run 4 was an outlier); ablation VPT is seed-invariant — the eval orbit's own event structure gates VPT |

## Shared setup (current — runs 3+)

`DIM=8` (N=256), `SEED=7673895`, `SPECTRAL_RADIUS=0.90`, `INPUT_SCALING=0.05`,
`LEAK_RATE=1.0`, `HISTORY_DEPTH=8`, tanh arm (`LORENTZ_GAMMA=0`), M=3 members,
`Combine::Mean`, Adam readout updates. Kappa: saturating ramp (KappaProfile, k=25)
to the config ceiling; the rollout runs at a fixed kappa set in FreeRun.

Lr schedule: pure cosine 1e-4 → 2e-5, 200 epochs.

Stream: 20000+1 samples at dt=0.02, normalized per channel to [-1, 1].
Cursor window [2000, 18000] (center 10000, span 16000), 100 warmup steps per
epoch, 15901 training steps per epoch. Evaluation runway E = 2000 steps past the
window = 36.2 Lyapunov times; the past-anchor runway [0, 2000) is exactly E long
(zero slack — `FREE_RUN_WINDOW_SIZE <= lb` is a hard geometry constraint).

Input channels (control arm): `[past xyz, future xyz, past x*z, future x*z]`.
The distance channel was removed in run 4 (see finding). Runs 1–2 used the older
geometry (span 15000, window [4000, 19000], E = 1000) with distance live on
channel 6 and past x*z on channel 7.

### Metrics

- **train RMSE** — *prequential* (test-then-train): `outputs` from `EnsembleESN::Step`
  is the consensus read at x(t) *before* that call's readout update, i.e. the honest
  pre-update prediction of that same call's target. RMS over 3 channels x all steps,
  normalized units.
- **dev[i]** — member i's epoch-RMS of its raw consensus deviation `y_i - c`
  (pre-kappa; exactly the signal the coupling would scale into feedback).
- **sd** — population std of the three dev values: the *asymmetry between members*.

### Free-run protocol & metrics

Rollout (`Lorenz::FreeRun`): reset cursors, hold kappa fixed, re-sweep the whole
training window teacher-forced but inference-only (anchored washout), then go
generative for E steps — each step reads a fresh consensus (`EnsembleESN::Predict`)
*before* `Step` consumes it as the future-channel input, and scores it against the
true held-out orbit.

- **err** — per-step channel-RMS of (consensus − truth), normalized units.
- **VPT** — first step with err > `VPT_THRESHOLD`, reported in steps and Lyapunov
  times (λ_max = 0.9056, dt = 0.02 → 55.2 steps/λt). Threshold 0.2 as of after the
  seed sweep (≈ the conventional 0.4 x climate-normalized error, climate ≈ 0.5
  here); every run up to and including the sweep used 0.3, so their VPTs read
  ~0.5 λt long vs the new convention.
- **free-run RMSE** — RMS over the full runway. Post-VPT this settles ≈ 0.5 with
  errors oscillating ~0.05–1.0 and repeated near-passes: bounded on-attractor
  wandering (regime-3 "climate" behavior), not divergence or collapse.

---

## Run 1 — new baseline (fell out of parameter-space exploration)

Config: old geometry (span 15000, window [4000, 19000], E = 1000), kappa 0,
lr floor 2e-5, distance channel live. Arrived at by exploration, not a controlled
delta from the retired campaign.

Result: final train RMSE **0.001412 at epoch 199** (the retired campaign's best
was 0.00284, on 1/3 the window). Trajectory: 0.116 @0 → 0.0136 @1 → 0.0047 @20 →
0.0034 @50 → 0.0022 @100 → 0.001412 @199, still descending at the floor.

Findings:
- **The late curve is essentially noise-free.** From ~epoch 90 the descent is
  monotone to the 6th decimal and the dev spread is frozen (sd ≈ 3.9e-5 for a
  hundred straight epochs). The single-sample gradient noise that defined the
  first campaign's curves is gone.
- No instability of any kind across 200 epochs (kappa 0, so none expected).

## Run 2 — first generative rollout

Config delta: FreeRun() built (EnsembleESN::Predict added for closed-loop input
ordering); kappa ceiling 0.01, rollout kappa 0.01. Run-1 geometry (E = 1000).

Result: train 0.001416 @199. **VPT 37 steps (0.67 λt)**; free-run RMSE 0.5047
over 1000 steps.

Findings:
- **Healthy asymptotics on the first attempt**: bounded errors (0.05–1.0), no
  tanh saturation, no attractor collapse; extended low-error near-passes late in
  the rollout (err 0.047–0.081 around steps 825–900).
- **Error compounds ~8x faster than the Lyapunov rate** early on (0.046 @25 from
  a 0.0014 one-step level) — the teacher-forced/generative distribution shift,
  not chaos, sets the horizon.

## Run 3 — wider window, hot-kappa rollout (summary lost to crash)

Config delta: current geometry (span 16000, E = 2000); kappa ceiling 0.02,
rollout kappa 0.1. Distance channel still live.

Result: train 0.001405 @199. Trace: err 0.0117 @25 / 0.045 @75 / 1.12 @100 —
the 0.3-crossing fell between steps 75 and 100 (~1.4–1.8 λt). The run crashed
after step 2000 (`free-run outran the anchor history`): the anchor runway is
exactly E long, and the trailing cursor step underflowed past the seed. Guard
added (stop stepping after the last scored index); VPT/RMSE summary lines were
lost, so this run enters the record as a trace-only bracket.

## Run 4 — distance channel dropped (control arm)

Config delta: distance channel removed from the inputs; layout becomes
`[past xyz, future xyz, past x*z, future x*z]`. Kappa ceiling 0.05, rollout
kappa 0.05. (Entangled delta vs run 3: kappa also changed.)

Result: train **0.000932** @199 (first sub-1e-3). **VPT 252 steps = 4.56 λt** —
at the old batch-pipeline goal level (~4.3 λt). Free-run RMSE 0.5115 over 2000
steps (36.2 λt). Trace held err ≤ 0.05 to step 50, recovered from excursions
(0.126 @100 → 0.038 @125), still 0.021 @225.

Finding: **the distance channel was a built-in train/free-run distribution
mismatch.** In training it ramps −1 → +1 (in envelope); in free-run it keeps
growing past +1, monotone in rollout depth — a guaranteed-OOD input dragging the
reservoir off the trained manifold on top of feedback-error compounding.
Removing it also improved train RMSE ~34%, so it was a nuisance input in-window
too (confounded with the kappa change).

## Run 5 — past-anchor ablation (does the anchor buy anything?)

Config delta vs run 4: past channels disabled surgically — inputs 0–2 duplicate
the future channels, both product channels carry future x*z. Everything else
identical (same seed, geometry, schedule).

Result: train **0.000521** @199 (best one-step ever) but **VPT 130 steps =
2.35 λt** — half the control. Free-run RMSE 0.4858.

Findings:
- **The past anchor buys ~2x VPT.** *(Retracted by the seed sweep below — the
  2x did not replicate across seeds.)* First direct measurement of the Janus
  half-anchoring thesis, in its favor: real anchor signal on half the input
  space extends phase tracking (err @125: 0.038 anchored vs 0.138 unanchored).
  Both arms stay bounded/on-attractor post-VPT — the anchor extends *phase*, it
  is not what prevents runaway.
- **One-step error and free-run quality dissociate.** Concentrating 6 of 8
  channels on the target-predictive future signal halves teacher-forced RMSE and
  halves VPT: in generation those channels all carry the model's own error, so
  the feedback loop gain doubles. Train RMSE must not be used to select
  free-run configs; only VPT arbitrates.
- Caveat: single rollout per arm; the ablation arm's doubled future drive is
  entangled with the past-removal (but a confound that helps one-step and still
  loses free-run strengthens the anchor's case).

---

## Seed comparison — control vs past-ablation

Protocol: run-4 control arm and run-5 ablation arm, identical config except
`SEED`. Single rollout per arm per seed (E = 2000, VPT threshold 0.3, rollout
kappa 0.05).

| seed | arm | train RMSE @199 | VPT steps | VPT λt | free-run RMSE |
|----------|--------------------|----------|-----|------|--------|
| 7673895  | past live          | 0.000932 | 252 | 4.56 | 0.512 |
| 7673895  | past = future copy | 0.000521 | 130 | 2.35 | 0.486 |
| 43434334 | past live          | 0.000826 |  92 | 1.67 | 0.499 |
| 43434334 | past = future copy | 0.000517 | 126 | 2.28 | 0.509 |
| 84368334 | past live          | 0.000937 | 129 | 2.34 | 0.494 |
| 84368334 | past = future copy | 0.000512 | 127 | 2.30 | 0.481 |
| *median* | past live          | 0.000932 | 129 | 2.34 | 0.499 |
| *median* | past = future copy | 0.000517 | 127 | 2.30 | 0.486 |

### Sweep findings

- **The anchor's 2x VPT did NOT survive the median.** Arm medians: 129 vs 127
  steps (2.34 vs 2.30 λt) — parity. Run 4's 4.56 λt was a lucky draw, and the
  arm-pair difference reversed sign at seed 43434334 (1.67 anchored vs 2.28
  unanchored). Run 5's headline finding is retracted.
- **The ablation arm is eerily seed-invariant**: VPT 130 / 126 / 127 (4-step
  spread across reservoir realizations) and train RMSE 0.000521 / 0.000517 /
  0.000512 (< 2% spread). The anchored arm carries all the variance: VPT
  92–252, train 0.000826–0.000937.
- **VPT is gated by fixed events in the shared eval orbit, not by the model.**
  All six runs score against the SAME held-out trajectory (same x0, same
  window), and their traces spike in lockstep near steps 175 and 250 (err
  ~0.8–1.2 in five of six runs at 175) — the eval segment has hard decoherence
  events (lobe transitions) at fixed depths. Five of six VPTs cluster at
  92–130, at the difficulty ramp into the first event; run 4's 252 means it
  survived event #1 and died entering #2. Reservoir-seed replication never
  varies the data: these six VPTs are three model-noise samples against ONE
  orbit. Measuring a real VPT distribution requires varying the eval segment
  (different x0 or window placement), not the seed.
- **Post-VPT climate is universal**: free-run RMSE 0.481–0.512 across all six
  runs regardless of arm or seed; every run stays bounded and on-attractor with
  repeated near-passes (regime 3). Re-lock depth is the discriminator that
  varies: the campaign's deepest episode is seed 84368334 ablation, err
  ≈ 0.02–0.28 for ~300 steps (~5.5 λt) around steps 625–950.
- What DOES survive seeds: the **train-RMSE stratification** (ablation
  ≈ 0.00052, anchored ≈ 0.00083–0.00094, reproducible to a few percent) — and
  with VPT at parity, the run-5 dissociation lesson stands in sharpened form:
  a 1.8x one-step gap moved free-run performance not at all.

---

## Open questions

1. ~~VPT variance across seeds~~ — RESOLVED (sweep above): the anchor's 2x did
   not survive; arms are at VPT parity. Successor question: **vary the eval
   orbit**, not the seed — VPT is currently gated by one orbit's fixed event
   structure, so every arm comparison rides the same few hard transitions.
2. **Kappa in free-run, isolated.** Rollout kappa has never been swept alone
   (0.01 / 0.1 / 0.05 runs all changed other things too). Cheap: sweep
   `SetKappa` in FreeRun on a fixed trained ensemble.
3. **Teacher-forced → generative mismatch** is the dominant known deficit
   (run 2's 8x-Lyapunov error growth). Untried lever: closed-loop training
   exposure (late epochs fed the model's own predictions on the future channels).
4. ~~VPT_THRESHOLD = 0.3 is provisional~~ — RESOLVED: set to 0.2 to match the
   conventional 0.4 x climate-normalized definition (see Metrics).
