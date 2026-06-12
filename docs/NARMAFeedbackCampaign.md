# NARMA-30 Feedback A/B Campaign — Run Log and Analysis

Campaign date: 2026-06-12. Harness: `examples/NARMAFeedback` (see
`docs/FeedbackTrainingMethodology.md` §9.4 for the experiment design this
executes). This document preserves the raw summary tables and the analysis of
every run, including the narratives that were later retracted — the
retractions are part of the record. Raw per-validation-point logs live
untracked in the repo root (`narma_*.log`); everything decision-relevant from
them is captured here.

**Status at end of campaign day 1: ongoing.** No valid feedback advantage
demonstrated on NARMA-30 yet; two structural defects of the training
methodology were found and fixed along the way (the saturation ratchet and
the wrong-space attribution gauge). The standing rule (user-set): a flat
bounded campaign triggers a rethink of the entire methodology — branch on the
Σy diagnostic — **not** a move to the Lorenz target.

---

## 0. Common setup

- Task: tanh-wrapped NARMA-30, canonical coefficients (0.3/0.05/1.5/0.1,
  u ∈ [0, 0.5]); train stream data seed 1939, held-out validation data seed
  7331 (never trained on, fixed across all arms/seeds/runs).
- Full preset: W = 500 (warmup = §6.17 washout), 10 000 pretrain +
  40 000 alternation examples, validation every 1 000 examples (1 000 scored
  steps), 5 paired seeds (73895–73899; P seed 42+k, F seed 43+k).
- ESN: DIM = 8 (N = 256), M = 32, sr 0.92, input_scaling 0.5,
  output_fraction 1.0, 1 feedback channel, feedback_scaling 0.5 (LIVE) /
  0.0 (CTRL — §6.13 compute-matched kill-switch control: same RNG draws,
  same ~4× probe overhead, F provably frozen).
- Feedback defaults per §6.14 unless a run says otherwise: ε = 0.05,
  F lr = 2e-4, p_lr = 5e-4, F = 1 layer × 8 channels.
- Metrics: validation NRMSE; from run 2 on, `tail10` = mean of last 10
  validation points (single-point "final" is a lottery draw from an
  oscillating series — both arms' validation NRMSE swings 0.5 → 2.5 between
  consecutive points, a P-side streaming-noise issue, still unexplained).
- LESION = LIVE's trained system re-validated once with `force_zero`
  (measures how much of the trained system's performance the closed loop
  carries at evaluation; proves dependence, not benefit).
- Wall clock: a full 5-seed A/B (10 arms × 50 500 examples) ≈ 2.7 min.

---

## 1. Run 1 — first full A/B (harness `ada3e3d`)

Config: §6.14 defaults exactly (margin 0, no F weight decay, no box).
Metric: single-point final NRMSE. Attribution gauge: **raw pre-clamp**
`var_f` (later shown to be the wrong space).

```
seed     LIVE      CTRL      delta     LIVEbest  CTRLbest  lesioned
73895   0.60738   2.20868   +1.60129   0.55124   0.60454   80.78169
73896   1.08853   1.41738   +0.32885   0.44598   0.70784   30.65039
73897   1.29795   0.51947   −0.77848   0.46158   0.46977    1.71088
73898   0.47984   0.71006   +0.23023   0.41855   0.40642    0.56390
73899   0.59358   1.40011   +0.80653   0.49148   0.47109    0.56808

mean paired delta: +0.43768   live wins: 4/5
mean var_f: 2.464e-02 (std 0.157 vs eps 0.050)
PRINTED VERDICT: "state-dependent signal — proceed to Lorenz"   ← RETRACTED
```

Seed 73895 LIVE telemetry trajectory (the §6.11/§7.8 saturation runaway,
live):

```
cycles    mean_f    sat%    lever   accept%
11k–21k   ≈ −0.04     0%    0.997    ~80%     healthy regime
22k       −0.32       0%    0.87      71%     drift begins
29k       −1.41       0%    0.21      88%     lever collapsing
33k       −2.21      94%    0.05      94%     saturation watch fires
50k       −3.31     100%    0.006     99%     F pinned at tanh ≈ −1
```

**Analysis (and why the verdict was retracted):**

- The validation series oscillates so violently (0.47 → 2.7 between
  consecutive points, both arms) that single-point finals are noise draws.
  On best-NRMSE the arms were a wash.
- The printed verdict used **pre-clamp** variance. At `mean_f ≈ −3.3` the
  clamp lever is 0.006: raw std 0.157 but the **injected** signal's std was
  ~0.001 — a near-constant −1, i.e. §7.4's "glorified bias" misclassified as
  state-dependent because the gauge measured the wrong space.
- At saturation, accepts still fired at ~99%: probes resolve
  microscopic-but-deterministic loss differences, and every accept commits a
  full-ε creep — "accepts committing full-ε creeps on ever-thinner evidence,"
  almost verbatim the §6.11 warning.
- LESION exploding (80.8) for the runaway seed: P deeply adapted to a
  constant −1 drive; muting it pushes the reservoir far off P's training
  distribution. Dependence, not benefit.

**Actions taken** (commit `3cd21a6`): post-clamp gauge `var_tanh_f` added to
telemetry (§7.4 amended — attribution must use the space the reservoir
sees); F weight decay 1e-4 per §6.11's prescription; tail10 metric.

---

## 2. Run 2 — weight decay 1e-4, tail10, post-clamp gauge

```
seed     LIVEtail  CTRLtail   delta     LIVEbest  CTRLbest  lesioned
73895    1.14802   1.45824   +0.31022   0.63556   0.60454   83.66360
73896    1.20734   1.23072   +0.02338   0.54422   0.70784   73.68704
73897    0.76746   0.67560   −0.09185   0.42471   0.46977    1.13222
73898    0.81766   0.74369   −0.07397   0.43101   0.40642    0.63686
73899    0.97009   0.85487   −0.11522   0.49364   0.47109    0.57319

mean paired tail10 delta: +0.01051   live wins: 2/5
mean var_tanh_f: 4.100e-04 (std 0.0202 vs eps 0.050)
PRINTED VERDICT: no consistent LIVE win
```

Final LIVE telemetry per seed:

```
seed     acc%     std_thf   mean_f    sat%    lever   realiz
73895    85.7%    0.0002    −6.00    100%    0.000   0.114    ← runaway (worse than run 1)
73896   100.0%    0.0001    +4.98    100%    0.000   0.116    ← runaway, POSITIVE direction
73897    82.8%    0.0205    −0.006     0%    1.000   0.013    healthy
73898    78.1%    0.0276    −0.045     0%    0.997   0.014    healthy
73899    80.5%    0.0294    +0.033     0%    0.998   0.017    healthy
```

**Analysis:**

- Weight decay 1e-4 did NOT stop the runaway — 2/5 seeds saturated, in
  **opposite directions**. That settled the §7.8 question: the drift is a
  seed-random **random-walk ratchet**, not task-driven asymmetry. (Why 1e-4
  is powerless: effective per-step shrink ≈ lr · wd = 2e-4 · 1e-4 = 2e-8.)
- The apparent LIVE advantage lived entirely in the runaway (bias) seeds;
  all three healthy seeds lost slightly to CTRL.
- Mechanism, fully stated: at H = 1 the chance-floor accepts (~2/3 at zero
  margin, §6.6) are a random walk in F's output with a full-ε step per
  accept and **no restoring force anywhere**; clamp saturation is the walk's
  absorbing boundary. Margins thin the walk's steps; they cannot bound it.

---

## 3. Absolute-margin sweep (`--margin-sweep`, commit `6d29ed7`)

LIVE arms only, run-2 runaway seeds, full budgets, margins absolute.

```
margin    seed     tail10    acc%      sat%    mean_f   std_thf    mean_e0
1e-06    73895    0.97573   71.8%    100.0%   +2.7903   0.0025    4.317e-05
1e-06    73896    1.22191   86.1%      3.4%   +1.8326   0.0105    7.679e-05
1e-05    73895    1.10915   66.3%     19.7%   −1.9095   0.0081    5.551e-05
1e-05    73896    1.40941   70.2%     23.0%   +1.9539   0.0047    8.464e-05
1e-04    73895    0.82976   12.9%      0.0%   −1.2369   0.0168    4.772e-05
1e-04    73896    0.80718    0.0%      0.0%   +0.0914   0.0074    2.088e-05
```

**Analysis: no absolute margin works.** The mean probe loss is only ~5e-5,
so the ratchet-killing threshold (1e-4) **exceeds the typical loss itself**
— it freezes healthy learning too (accepts 0–12.9%, the survivors firing
only on rare high-error cycles). Margins small enough to keep accepts alive
only slow the drift.

**Action** (commit `dee8065`): margin made **relative** —
`accept iff min(E+, E−) < E0 · (1 − r)` (§6.6 amended; §6.6 had in fact
anticipated "promotion to a small relative value"). Bit-identical to the
original comparison at r = 0, so all §6.13 kill-switch arguments survive.

---

## 4. Relative-margin sweep (commit `dee8065`)

```
margin   seed     tail10    acc%      sat%    mean_f   std_thf    mean_e0
0.5%    73895    1.11920   99.5%    100.0%   −3.5593   0.0011    6.113e-05
0.5%    73896    1.21731   99.5%    100.0%   +3.3401   0.0009    5.668e-05
2%      73895    0.95808   97.0%    100.0%   −3.0551   0.0016    7.077e-05
2%      73896    0.84948   75.5%      0.0%   +0.8110   0.0629    4.127e-05
5%      73895    0.94917   92.2%     81.5%   +2.2034   0.0100    3.729e-05
5%      73896    0.88055   96.1%     98.0%   +2.1895   0.0044    4.732e-05
```

**Analysis: the relative margin failed too, and falsified the model behind
it.** Even demanding a 5% improvement of E0, accepts fire at 92–99%. The
probe's loss delta is `δE ≈ 2(ŷ−y)·δŷ` while `E0 = (ŷ−y)²`, so the
*relative* delta is `≈ 2·δŷ/(ŷ−y)` — numerator and denominator shrink
together as P improves. Even at lever 0.001 the perturbation moves the loss
by percent-level fractions of E0. **No margin of either kind can bound the
walk** — the root cause is that F's magnitude is unanchored.

One suggestive cell: 2%/73896 stayed healthy with std_thf 0.063 > ε and the
best tail10 of the table — first hint of what a non-saturated F might be
worth.

**Action** (commit `18cc51d`): pre-clamp creep-target **box** —
`f* = clamp(Sf ± ε, ±f_box)`, default `f_box = 1.5` (lever at the wall
≈ 0.18, probes stay alive forever). The walk reflects off the wall instead
of absorbing at saturation. §6.11 amended.

---

## 5. Box verification sweep (f_box = 1.5 + relative margins)

Same six cells as §4, box active:

```
margin   seed     tail10    acc%    sat%    mean_f    std_thf   mean_e0
0.5%    73895    1.17073   62.7%   0.0%    −1.4001   0.0115    6.013e-05
0.5%    73896    0.92970   79.3%   0.0%    +1.3930   0.0118    5.638e-05
2%      73895    1.01419   68.3%   0.0%    −1.4301   0.0069    6.935e-05
2%      73896    0.85647   82.7%   0.0%    +1.0771   0.0374    3.607e-05
5%      73895    0.83511   69.9%   0.0%    +1.3885   0.0113    3.191e-05
5%      73896    1.02695   84.9%   0.0%    +1.3871   0.0137    4.599e-05
```

**Analysis: the box holds — ratchet dead in all six cells.** Saturation
0.0% everywhere; `mean_f` hugs ±1.4 (drift pressure persists, the box
contains it); margin barely matters (confirming the box, not the margin, was
the missing anchor); std_thf alive in every cell. Carried-forward config
fixed here: **f_box = 1.5, margin = 0.02** (commit `83eeb2d`).

---

## 6. Run 3 — first VALID A/B (healthy F: box + 2% margin)

Note: the 73895/73896 LIVE arms are config-identical to the §5 2%-margin
cells (same numbers).

```
seed     LIVEtail  CTRLtail    delta     LIVEbest  CTRLbest  lesioned
73895    1.01419   1.45824   +0.44405   0.51504   0.60454   77.40901
73896    0.85647   1.23072   +0.37425   0.56490   0.70784   15.05931
73897    0.78157   0.67560   −0.10596   0.45581   0.46977    1.82976
73898    0.80177   0.74369   −0.05808   0.45171   0.40642   43.56615
73899    0.91968   0.85487   −0.06480   0.43051   0.47109    0.72337

mean paired tail10 delta: +0.11789    live wins: 2/5
mean var_tanh_f: 5.420e-04 (std 0.0233 vs eps 0.050)
PRINTED VERDICT: no consistent LIVE win
```

Final LIVE telemetry per seed:

```
seed     acc%    std_thf   mean_f    sat%   lever   realiz   sgnbal
73895    68.3%   0.0069    −1.430    0.0%   0.205   0.049    −0.054
73896    82.7%   0.0374    +1.077    0.0%   0.376   0.045    +0.004
73897    74.3%   0.0220    −0.022    0.0%   0.999   0.013    −0.023
73898    72.9%   0.0147    −1.363    0.0%   0.232   0.042    −0.012
73899    83.1%   0.0237    +0.037    0.0%   0.998   0.015    −0.004
```

**Analysis:**

- Box generalizes: sat 0% on five fresh-seed arms.
- Mean paired delta **+0.118** — positive, 10× run 2 — with strong
  asymmetry: wins big (+0.44, +0.37), losses small (−0.06…−0.11). On
  best-NRMSE LIVE won 4/5. The sign test still fails (2/5 on tail10).
- The two wins were "wall-sitter" seeds (mean_f ≈ ±1.1–1.4, injected signal
  ≈ a strong bounded bias of tanh(1.4) ≈ ±0.89 plus modest variation);
  near-zero-mean seeds all lost slightly. **Initial narrative: "the useful
  thing F finds first is a large operating-point shift, reached only by some
  walks in 40k cycles." This narrative was RETRACTED after run 4 — see §8.**
- Realization 0.013–0.049, still 3–10× below the §6.14 band 0.1–0.3 → F
  learns at a crawl → run 4 per the spec's own tuning rule.

---

## 7. Run 4 — F-lr sweep + Σy mechanistic diagnostic (commit `badc020`)

New diagnostic, run at the end of every arm from here on:
`corrSigma = corr(tanh F(x), Σ y(t..t−29))` on a validation drive. NARMA-30's
recurrence contains `β·y(t−1)·Σy` — a global scalar of exactly the kind a
broadcast scalar feedback channel is architecturally matched to carry. This
is the sharpest mechanistic question available: *is F learning the useful
thing?*

LIVE arms only; lr ∈ {5e-4, 1e-3, 2e-3}; seeds 73895 (wall-prone in run 3)
and 73897 (center-hugging). Baseline at lr = 2e-4 from run 3: realiz
0.013–0.049.

```
f_lr     seed     tail10    acc%    sat%    mean_f    std_thf   realiz   corrSigma
5e-4    73895    1.07193   80.2%   0.0%    +0.0871   0.0294    0.023     −0.089
5e-4    73897    0.75086   80.8%   0.0%    −0.0434   0.0128    0.009     −0.093
1e-3    73895    1.32157   73.6%   0.0%    −0.0477   0.0242    0.050     −0.067
1e-3    73897    0.87554   68.8%   0.0%    +0.0297   0.0230    0.034     −0.040
2e-3    73895    1.02208   75.5%   0.0%    −0.0544   0.0410    0.049     +0.054
2e-3    73897    1.16384   77.3%   0.0%    −0.0765   0.0664    0.115     −0.036

per-arm finals:    1.864 / 0.745 / 1.039 / 0.600 / 1.642 / 0.733
per-arm bests:     0.588 / 0.529 / 0.610 / 0.500 / 0.598 / 0.573
per-arm lesioned:  2.028 / 1.107 / 1.061 / 0.612 / 1.901 / 0.891
```

**Analysis:**

1. **lr is not the lever.** 10× lr moved realization non-monotonically from
   ~0.02 to at best 0.115 (one cell), with tail10 *worse* on average. Adam's
   per-parameter normalization on single-sample updates apparently decouples
   realized step size from lr — the §6.14 tuning rule hits diminishing
   returns immediately.
2. **Box holds in all six cells**; std_thf grows with lr (up to 0.066 —
   above ε for the first time in a centered seed).
3. **corrSigma ≈ 0 in every cell** (−0.09…+0.05, bracketing zero). F is NOT
   learning the recurrence's global scalar at any tested lr.

---

## 8. Cross-run check: the wall-sitter narrative retracted

Prompted by the user ("did this still hold on round 4?"). Two findings:

**No wall-sitters formed in run 4 at all** — all six cells ended
|mean_f| ≤ 0.09, including 73895 which sat at −1.43 in run 3. Run 3 (lr
2e-4): 3 wall-sitters in 5 walks. Run 4 (higher lr): 0 in 6. Lopsided enough
to suggest higher lr suppresses wall-sitting (the weight-decay pull scales
with lr while the ±ε creep does not) — and either way, wall-sitting is a
stochastic walk outcome, not a destination the dynamics reliably find.

**Seed 73895 kept winning without the wall.** Run-3 CTRL results are valid
comparators for run-4 LIVE cells (CTRL's F is frozen dead, so feedback.lr
cannot affect it; same seed, same stream, same P config):

```
seed 73895 (CTRL tail = 1.458):            seed 73897 (CTRL tail = 0.676):
  run 3, wall (−1.43):    1.014  (+0.44)     run 3, centered:  0.782  (−0.11)
  run 4 @5e-4, centered:  1.072  (+0.39)     run 4 @5e-4:      0.751  (−0.08)
  run 4 @1e-3, centered:  1.322  (+0.14)     run 4 @1e-3:      0.876  (−0.20)
  run 4 @2e-3, centered:  1.022  (+0.44)     run 4 @2e-3:      1.164  (−0.49)
```

73895 beats its control at **every** operating point, wall or centered, any
lr; 73897 loses at every operating point. The predictor of win/loss is not
where F's walk landed — it is **which seed you are on**. The stable
cross-run pattern is per-seed sign consistency, suggesting
reservoir/P-realization-dependent benefit, not operating-point-dependent.

Caveats: tail10 carries the validation-oscillation noise; single
unreplicated cells; 73896 (run 3's other winner) was never in the sweep.

---

## 9. Where the campaign stands; next steps

**Evidence so far** points at rethink branch 1 — the zeroth-order
probe/creep learning rule (§7.1 sample inefficiency + §7.2 myopia): F does
not learn Σy at any tested lr, and the wins that exist do not come from
state-dependent signal (std_thf ≤ ~0.07, corrSigma ≈ 0; per-seed sign
consistency unexplained).

The rethink branch map (agreed with the user — a flat campaign reopens the
methodology, not a move to Lorenz):

```
flat result + diagnostic readout       →  what's broken                →  rethink target
──────────────────────────────────────────────────────────────────────────────────────────
corrSigma ≈ 0 even at supervised        →  F cannot represent Σy from  →  F architecture /
ceiling                                    the (subsampled) state         injection path

corrSigma ≈ 0 probe-trained but high    →  learning rule can't extract →  probe/creep rule:
supervised                                 the signal                     H>1 probes, §6.12
                                                                          batch-F outer loop,
                                                                          different F optimizer

corrSigma high, P still flat            →  P/reservoir can't exploit a →  injection architecture:
                                           single broadcast scalar        multi-channel, vertex
                                           (§2.3 single-input collapse)   subsets, vtx_feedback

deltas exist but drown in P's           →  co-adaptation / streaming   →  schedule (§7.3):
validation oscillation                     noise floor                    freeze-P phases,
                                                                          §6.12 outer loop
```

**Queued next experiments:**

1. **`--fcap-check` supervised ceiling** (agreed, not yet built): train F
   *directly* on Σy targets (series is generated; the supervisor is free)
   while driving closed-loop; measure the same corrSigma. Discriminates
   "learning rule can't extract it" from "F can't represent it" — the first
   branch decision in the map above. (~30 lines, pattern matches
   `--flr-sweep`; uses `TrainOnlineStepRegression` on F through the ESN
   seam.)
2. **Per-seed benefit test**: rerun 73895/73897 LIVE with different *F*
   seeds only — does the win/loss sign track the reservoir realization?
3. Any positive result must replicate on a fresh validation data seed
   before it is believed.

**Methodology changes the campaign produced** (all in
`docs/FeedbackTrainingMethodology.md` as amendments with evidence):

| Change | Commit | Spec section |
|---|---|---|
| Post-clamp attribution gauge `var_tanh_f` / `mean_tanh_f` | `3cd21a6` | §7.4 amended |
| Relative accept margin `min < E0·(1−r)` | `dee8065` | §4 step f, §6.6 amended, §6.14 |
| Pre-clamp creep-target box `f* = clamp(Sf ± ε, ±f_box)`, default 1.5 | `18cc51d` | §6.11 amended, §6.14 |
| `ESN::LastFeedbackRaw()` diagnostic accessor + Σy correlation | `badc020` | harness-side |

**Open P-side issue, unexplained:** validation NRMSE oscillates 0.5 → 2.5
between consecutive validation points in BOTH arms (so not feedback-related)
— streaming P training at constant p_lr = 5e-4 is noisy. It inflates every
tail10 comparison. Candidate §7.3 watch item; lowering p_lr below F's lr
would invert the §6.9 timescale ordering, so the fix is not obvious.
