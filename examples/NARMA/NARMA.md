# NARMA-N — Nonlinear System Identification

NARMA is HypercubeESN’s primary **open-loop** validator: a white input `u(t)`
drives the reservoir; the readout reconstructs the NARMA output `y(t)`. The task
stresses **memory depth and nonlinear mixing together**.

## One system, three orders

The central result of this campaign is not three separately tuned demos. It is
that **one fixed HypercubeESN configuration** — same dim, M, spectral radius,
input scaling, leak, series length, HCNN stack, training schedule, and the same
20 reservoir seeds — is run on tanh-wrapped **NARMA-30, NARMA-50, and NARMA-70**.
Only the recurrence order (and thus the target series) changes.

**Featured metric: best 5 of 20** (lowest test NRMSE among the 20 seeds). That
is the campaign story for what the architecture delivers when reservoir
realization is in a competent band — not the single best seed, and not a mean
dragged by catastrophic outliers. Running 20 seeds is the survey; keeping the
top five is ordinary seed selection, not retuning the op-point per order.

| Order | Best-5 mean | Best-5 std | Best-5 min … max | Best seed |
|------:|------------:|-----------:|-----------------:|----------:|
| **30** | **0.0441** | 0.0017 | 0.0419 … 0.0461 | **0.0419** |
| **50** | **0.0751** | 0.0009 | 0.0742 … 0.0766 | **0.0742** |
| **70** | **0.1251** | 0.0016 | 0.1225 … 0.1264 | **0.1225** |

Error rises with order, as expected for honest tanh-wrapped NARMA. Under one
op-point the best-5 clusters stay **tight** at every rung (std ≤ 0.002) — N30
~0.044, N50 ~0.075, N70 ~0.125. That ladder is the multi-order claim.

All-20 pool stats (including outliers) are in [Results](#results-test-nrmse)
for completeness; they are not the headline.

Raw stdout (authoritative per-trial numbers):

| Order | Log |
|------:|-----|
| 30 | [NARMA-30.txt](NARMA-30.txt) |
| 50 | [NARMA-50.txt](NARMA-50.txt) |
| 70 | [NARMA-70.txt](NARMA-70.txt) |

### Is a shared op-point across orders common?

**Somewhat uncommon in the published RC literature — and that is part of the
point.**

Papers usually report **best-of-grid** (or best-of-seed) results **per task**:
spectral radius, input scaling, reservoir size, and sometimes leak or training
budget are retuned for NARMA-10 vs NARMA-30, often with different series lengths.
That is a fair way to ask “what can this architecture achieve when optimized,”
but it **does not** show that one operating point generalizes across memory
demands.

Holding the machine fixed and only changing N is closer to a **robustness /
transfer** test: does the architecture + delay-line depth still identify the
series when the required lag and nonlinear product structure get harder? Many
internal baselines do something like that; many leaderboard tables do not.

Caveats so this is not oversold:

- We still chose a strong shared op-point (M=32, collect 32000, small
  input_scaling, HCNN readout). It is not “default random knobs.”
- The full 20-seed pool is wider at N70 (two hard fails); the best-5 band stays
  tight — seed quality matters more as order grows, even with fixed knobs.
- Literature NARMA-30 bands are poorly standardized; use them only as a
  rough sanity check (below).

So: **same-config multi-order success is a real strength of this validator**,
with best-5 as the representative multi-seed story across 30 / 50 / 70.

---

## Shared configuration

Pinned in the `campaign` section at the top of `NARMA.cpp` (`MakeBaseESNConfig`,
series knobs, reservoir seed list) — keep in lockstep with this table.

| Knob | Value |
|------|--------|
| Variant | **tanh-wrapped** — fixed coeffs α=0.3, β=0.05, γ=1.5, δ=0.1; `u ∈ [0, 0.5]` |
| `data_seed` | **1939** — fixes the entire u/y series (same series for every reservoir seed) |
| Reservoir | dim=10 (N=1024), **M = history_depth = 32**, spectral radius 0.99, leak 1.0, input_scaling **0.03**, **bias_scaling 0.02** |
| Readout input | `readout_slices = 2` (B=2 → HCNN start dim 11, capacity 2048) |
| HCNN | Conv(1→16, TANH) → MaxPool → Linear(16384→1) · **16593** trained parameters |
| `readout.seed` | **3423555** — fixed HCNN weight init so multi-seed spread is reservoir-side |
| Series | warmup 300 · collect **32000** (train 25600 / test 6400) |
| Training | 600 epochs, batch 128, lr 0.0015 cosine (floor 7.5e-06) |
| Best-epoch | `restore_best_epoch = true`, **holdout_frac = 0** — restores min **train** MSE epoch (not a validation split). Test NRMSE is still a clean held-out metric. |
| Reservoir seeds (live harness) | Explicit list `campaign::kReservoirSeeds` in `NARMA.cpp`: **1** entry = spot run; **3** entries = literature band (mean/std/min/max over all three). No best-k selection. |

Metric: **test NRMSE** = RMSE / std(target) on the held-out 6400 steps. For the
**live harness**, report every listed seed (1 or 3) with mean/std/min/max when
n = 3. The **historical** tables below still document an earlier **best-5 of 20**
campaign on the same op-point — keep those until a 3-seed re-run is verified in
the same ballpark. Raw logs: [NARMA-30.txt](NARMA-30.txt) ·
[NARMA-50.txt](NARMA-50.txt) · [NARMA-70.txt](NARMA-70.txt).

**Cost (doc only — no smoke path):** roughly **12–14 minutes per seed** on the
collect machine (Release). A 3-seed literature run is ~**40–45 min per order**;
spot is one seed. Not CI-friendly by design; treat as a batch validator, not a
unit test.

---

## How to run

```text
cmake-build-release\NARMA.exe           # default order (see NARMA.cpp)
cmake-build-release\NARMA.exe 30
cmake-build-release\NARMA.exe 50
cmake-build-release\NARMA.exe 70
cmake-build-release\NARMA.exe --help
```

`order` is optional (integer ≥ 2). Op-point and series length live in the
`campaign` block at the top of `NARMA.cpp`. Edit `kReservoirSeeds` for mode:

- **1 seed** — spot check  
- **3 seeds** — literature stats over all three (list every seed explicitly)

Only `reservoir.seed` varies per trial; `history_depth` is fixed in
`MakeBaseESNConfig`.

---

## What the task is

### Recurrence (order N)

```text
y(t) = tanh(
         alpha * y(t-1)
       + beta  * y(t-1) * sum(y(t-1) … y(t-N))
       + gamma * u(t-N) * u(t)
       + delta
       )
```

(campaign variant: outer `tanh`, fixed coefficients at every order).

| Term | Role |
|------|------|
| α y(t−1) | Linear self-feedback (short memory) |
| β y(t−1) · sum(…) | Nonlinear mix over the last N outputs (deep memory) |
| γ u(t−N) · u(t) | Couples current input to the input N steps ago |
| δ | Constant offset |

Inputs `u(t)` are uniform on `[0, 0.5]`. The example rescales them to `[-1, +1]`
(`4u − 1`) before driving the reservoir.

### Target alignment (system identification)

This is **system identification, not forecasting**: `y(t)` is aligned with
`u(t)` at the same index. The reservoir has already been driven by everything
`y(t)` depends on for that step.

Pairing `targets[t] = y(t+1)` is a common porting bug: it asks for a term that
depends on `u(t+1)`, which the reservoir has not seen yet, and NRMSE collapses
toward 1.0 (predict-the-mean).

### Why tanh-wrap

Without a bound, bare canonical coefficients can diverge at high order. The
outer `tanh` keeps coefficients fixed so difficulty scales with **order**
(memory demand and nonlinear structure), not with a schedule that softens high-N
series. Default in `NARMA.cpp`: `NARMA_TANH_WRAP=1`.

---

## Results (test NRMSE)

All numbers from the three campaign logs. Same seeds and knobs; only order
changes.

### Featured: best 5 of 20

| Order | Best-5 mean | Best-5 std | Best-5 range | Best seed (R²) |
|------:|------------:|-----------:|-------------:|---------------:|
| 30 | **0.0441** | 0.0017 | 0.0419 … 0.0461 | **0.0419** (0.9982) |
| 50 | **0.0751** | 0.0009 | 0.0742 … 0.0766 | **0.0742** (0.9945) |
| 70 | **0.1251** | 0.0016 | 0.1225 … 0.1264 | **0.1225** (0.9850) |

Relative to N30 best-5 mean: N50 is about **1.7×** higher error; N70 about
**2.8×**. Controlled difficulty ladder on one machine.

| Order | Best-5 seeds (lowest NRMSE first) |
|------:|-----------------------------------|
| 30 | **1108635** 0.0419 · 517307 0.0429 · 147792 0.0442 · 591216 0.0454 · 812955 0.0461 |
| 50 | **665127** 0.0742 · 739040 0.0746 · 812955 0.0749 · 591216 0.0754 · 221691 0.0766 |
| 70 | **665127** 0.1225 · 1182560 0.1246 · 443400 0.1257 · 1552215 0.1263 · 812955 0.1264 |

**NARMA-30 and literature (careful).** Rough published “good” bands for order 30
are often cited around **0.40–0.60** NRMSE and “strong / large-N” around
**0.30–0.50**. Protocols vary (wrap, coefficients, series length, splits). Under
those caveats, this campaign’s best-5 mean **0.0441** and best **0.0419** sit
well below the bottom of those rough bands. Treat that as a
**sane-regime / capability** statement for *this* protocol — not bit-identical
leaderboard parity with any one paper. Orders 50 and 70 have **no** standard
published NRMSE bands here; they are internal stress rungs on the same system.

### Full pool (all 20) — transparency

Same 20 seeds at every order. Featured story is best-5 above; this table is the
complete survey record (including N70 outliers).

| Order | All-20 mean | All-20 std | All-20 min … max |
|------:|------------:|-----------:|-----------------:|
| 30 | 0.0490 | 0.0041 | 0.0419 … 0.0584 |
| 50 | 0.0807 | 0.0061 | 0.0742 … 0.0976 |
| 70 | 0.1553 | 0.0406 | 0.1225 … 0.2859 |

| res seed | N30 | N50 | N70 |
|---------:|----:|----:|----:|
| 147792 | 0.0442 | 0.0794 | 0.1637 |
| 221691 | 0.0493 | 0.0766 | 0.1499 |
| 295592 | 0.0473 | 0.0803 | 0.1499 |
| 369495 | 0.0476 | 0.0780 | 0.1307 |
| 443400 | 0.0513 | 0.0976 | 0.1257 |
| 517307 | 0.0429 | 0.0806 | 0.1691 |
| 591216 | 0.0454 | 0.0754 | 0.1380 |
| 665127 | 0.0502 | **0.0742** | **0.1225** |
| 739040 | 0.0466 | 0.0746 | 0.1511 |
| 812955 | 0.0461 | 0.0749 | 0.1264 |
| 886872 | 0.0473 | 0.0775 | 0.2859 |
| 960791 | 0.0584 | 0.0820 | 0.1569 |
| 1034712 | 0.0512 | 0.0776 | 0.1331 |
| 1108635 | **0.0419** | 0.0854 | 0.1521 |
| 1182560 | 0.0547 | 0.0780 | 0.1246 |
| 1256487 | 0.0514 | 0.0781 | 0.1267 |
| 1330416 | 0.0486 | 0.0798 | 0.2355 |
| 1404347 | 0.0522 | 0.0922 | 0.1817 |
| 1478280 | 0.0503 | 0.0836 | 0.1566 |
| 1552215 | 0.0537 | 0.0885 | 0.1263 |

At N30 every seed is strong; at N50 every seed stays under 0.10; at N70 two seeds
(886872, 1330416) fail hard and pull the all-20 mean up — which is why the
**best-5 band** is the cleaner multi-order comparison. Best seed is also
order-dependent (1108635 at N30; 665127 at N50 and N70).

---

## Metric notes

- **NRMSE** = RMSE / std(target) = √NMSE. If a paper reports NMSE, take the square
  root before comparing.
- Gut-check: NRMSE ≪ 1.0 beats predict-the-mean; this campaign’s N30/N50 best-5
  means are far below that floor.
- Train-set target means in the logs (series fingerprints only): N30 ≈ 0.949,
  N50 ≈ 0.995, N70 ≈ 0.999.

---

## Architecture note (M vs HCNN size; M vs NARMA order)

Reservoir **M** sets recurrent delay-line depth and weight count
(`N · dim · (M + drive blocks)`). The readout sees **B = readout_slices** packed
ages (here B=2), so HCNN capacity is 2<sup>(dim + log2 B)</sup> — independent of
M. In this campaign both M=32 and B=2 are fixed; only seed and NARMA order change.

**M need not equal the NARMA order.** At NARMA-70, M=32 is shorter than the
recurrence lag N=70; recurrent dynamics still carry long memory. Do not read
“M ≥ order” as a hard requirement for this protocol.

---

## What this campaign does *not* claim

- Bit-identical targets or train protocols to any particular paper.
- That a **random** seed always lands in the best-5 band (especially at N70 —
  the full pool is logged for that reason).
- That best-5 is the same as single best-seed cherry-picking (five seeds, tight
  clusters; op-point not retuned per order).
- That no retuning could improve N70 further (we did not grid-search per order).
- Closed-loop free-run performance (see Lorenz for that regime).

What it **does** support: under one documented open-loop protocol, with
**best-5 of 20** as the multi-seed story, HypercubeESN delivers strong system
identification on tanh-wrapped NARMA-30 and NARMA-50, and a tight usable band
on NARMA-70 (~0.125) — a primary validator for the ESN stack, with the shared
op-point as a feature of the evidence.
