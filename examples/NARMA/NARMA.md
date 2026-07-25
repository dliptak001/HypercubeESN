# NARMA-N — Nonlinear System Identification

## Spotlight — NARMA-30 at NRMSE 0.0570

Full write-up: **[NARMA-30.md](NARMA-30.md)** — best seed **0.0570** (R² 0.9968),
5-seed mean **0.0590** at M=16 / DIM10 / collect 32000 / tanh-wrap. About
**~5.3×** under the rough literature strong-band floor (0.30); order-30
protocols are poorly standardized (see [Reading the results](#reading-the-results)).
The multi-seed M-sweep table later still uses **collect 8000** and an older
op-point — even there, NARMA-30 reaches **0.09–0.13** mean NRMSE.

## Spotlight — NARMA-50 at NRMSE 0.0797

Full write-up: **[NARMA-50.md](NARMA-50.md)** — best seed **0.0797** (R² 0.9937),
5-seed mean **0.0882** at M=32 / DIM10 / collect 32000 / tanh-wrap (same drive
op-point as N30 spotlight). No literature band for order 50.

## Spotlight — NARMA-100 (TBD)

> **Template** — fill NRMSE / R² / knobs after the collect run; leave literature
> cells as **—** (order 100 is not a standard published RC rung).

| | NRMSE | vs this run |
|--|------:|------------|
| **HypercubeESN (best to date)** | **TBD** | — |
| Literature “strong / large-N” band | — | *no published band* |
| Literature “good” band | — | *no published band* |

**Test NRMSE TBD** (R² = TBD) on **tanh-wrapped NARMA-100**. No comparable
literature NRMSE band — treat this as an internal stress rung against N30
([NARMA-30.md](NARMA-30.md)) / N50, not a leaderboard claim.

| Knob | Value |
|------|--------|
| Variant | tanh-wrapped (α=0.3, β=0.05, γ=1.5, δ=0.1) |
| Reservoir | DIM=TBD (N=TBD), M=`history_depth`=TBD, FSF **TBD** |
| Seed | res TBD (single seed / multi-seed TBD) |
| Series | warmup TBD · collect TBD (train TBD / test TBD) |
| Drive | sr TBD · leak TBD · input_scaling TBD |
| Readout | TBD epochs, batch TBD, lr TBD (cosine) |
| HCNN | TBD |

## What this example demonstrates

NARMA (Nonlinear Auto-Regressive Moving Average) is the classic reservoir
computing benchmark. Unlike the sine-wave demo (which is mostly a memory
test), NARMA stresses **memory depth and nonlinear mixing at the same
time**: the target depends on a long history of itself *and* on a delayed
copy of the input, multiplied together.

The reservoir is driven by a white (uncorrelated) input `u(t)` and the
readout must reconstruct the NARMA output `y(t)`. This is **system
identification, not forecasting** — `y(t)` is aligned to `u(t)`, so the
reservoir has already seen everything it needs (see "Target alignment"
below).

**FSF A/B:** set `base.reservoir.full_state_feedback` / `fsf_scaling` (and the
`sweep_fsf_seeds` list) in `NARMA.cpp`. Log: `FSF: ON|OFF …`. See
[examples/README.md](../README.md).

## The recurrence

For order `N`:

```
y(t) = alpha*y(t-1) + beta*y(t-1)*sum(y(t-1 .. t-N)) + gamma*u(t-N)*u(t) + delta
```

- `alpha*y(t-1)` — linear self-feedback (short memory)
- `beta*y(t-1)*sum(...)` — **nonlinear** mixing over the last `N` outputs (deep memory)
- `gamma*u(t-N)*u(t)` — couples the current input to the input `N` steps ago
- `delta` — constant offset (gives `y` a small positive mean)

Inputs `u(t)` are drawn uniformly from `[0, 0.5]`. The example rescales them
to `[-1, +1]` (`u*4 - 1`) before driving the reservoir.

## Target alignment

NARMA is **system identification, not forecasting**: `y(t)` is produced from
`u(t)` and `u(t−N)`, so a reservoir driven by `u(t)` has already been shown
everything `y(t)` depends on. The example pairs `inputs[t] = u(t)` with
`targets[t] = y(t)` at the *same* index — no one-step-ahead shift.

The shift is not cosmetic. Pairing `targets[t] = y(t+1)` (a common porting
bug) asks the readout to predict `y(t+1)`, which carries the term
`gamma*u(t+1)*u(t+1−N)` and so depends on `u(t+1)` — an input the reservoir has
not been driven with yet. That term is unlearnable and NRMSE collapses toward
1.0 (predict-the-mean).

## Coefficient schedule

`NARMACoefficientsFor(N)` is the single source of truth for the
order-dependent coefficients. Bare canonical coefficients (`beta = 0.05`)
have **no real fixed point past N ≈ 23** and diverge. The schedule keeps
high-order NARMA bounded:

- `delta` drops `0.1 → 0.01` at `N ≥ 20`
- `beta` is scaled to hold `beta*N = 0.5` at `N ≥ 24`

The generator carries a magnitude-based divergence guard that throws a
fully-described exception if the recurrence ever blows up (it survives
`-ffast-math`, which the Release build enables).

## tanh-wrapped variant (`NARMA_TANH_WRAP`) — honest order-scaling

The coefficient schedule above has a side effect: by *weakening* `beta`/`delta`
as the order grows, it makes higher-order NARMA **smoother and easier**, not
harder. So NRMSE across orders is **not** a memory-depth ladder — e.g. order 30
can score lower (better) than order 10 purely because its series is smoother.

The fix is the standard literature form: wrap the recurrence in `tanh`, which
bounds `y(t)` in `(-1, 1)` unconditionally. Coefficients can then stay **fixed**
at every order (the canonical `0.3 / 0.05 / 1.5 / 0.1`), so the nonlinearity is
preserved and difficulty scales honestly with order.

A compile switch at the top of `NARMA.cpp` selects between the two for clean
A/B comparison:

```cpp
#define NARMA_TANH_WRAP 0   // 0 = legacy (scheduled coeffs, no squashing)
                            // 1 = tanh-wrapped (fixed coeffs, honest scaling)
```

Flip it (or build with `-DNARMA_TANH_WRAP=1`) and rerun. The banner echoes the
active variant on the `Variant:` line. Both share the same input stream and
seed, so at a *fixed* order the only difference is the squashing. The tanh form
holds the nonlinearity at full strength at every order, so high orders stay
genuinely hard instead of being *smoothed easier* the way the legacy schedule
makes them (whose NRMSE can flatten or drop with order). As the Results below
show, that difficulty surfaces mainly as a deeper memory requirement, not a
strictly higher NRMSE. With `tanh_wrap` on, `MakeNARMATask` ignores the
order-dependent schedule and uses the fixed canonical coefficients.

## Reading the results

The metric is `NRMSE = RMSE / std(target) = √NMSE` — if a paper reports NMSE,
square-root it first (NMSE 0.16 → NRMSE 0.40). As a quick gut-check on any
single run: `≤ 0.22` is compelling, `≤ 0.30` credible, `< 1.0` beats
predict-the-mean, and `≥ 1.0` means something is broken.

Rough literature bands on this same metric, by order — and where HypercubeESN
lands on the hardest row:

| Order    | "good" NRMSE | strong / large-N | Baseline | **This project** |
|----------|--------------|------------------|----------|------------------|
| NARMA-10 | 0.20–0.40    | —                | clean    | see M-sweep table |
| NARMA-20 | 0.30–0.50    | 0.20–0.35        | rough    | see M-sweep table |
| NARMA-30 | 0.40–0.60    | 0.30–0.50        | rough    | **0.0570** best / 0.0590 mean ([NARMA-30.md](NARMA-30.md)) |
| NARMA-50 | —            | —                | none     | **0.0797** best / 0.0882 mean ([NARMA-50.md](NARMA-50.md)) |
| NARMA-100| —            | —                | none     | TBD ([spotlight](#spotlight--narma-100-tbd)) |

**Read the NARMA-30 row carefully.** The strong / large-N floor is **0.30**.
Our best test NRMSE is **0.0570** (5-seed mean **0.0590**) — about **5× lower
error** than that floor (0.30 / 0.0570 ≈ 5.3). Even the older collect-8000
multi-seed means (~0.09–0.13 at good M) sit well under 0.30. That is the
headline result of this example; full seed table in [NARMA-30.md](NARMA-30.md).

NARMA-10 is the only order with a clean, comparable baseline (Jaeger 2001;
Rodan & Tiňo 2011): your *aligned* task (`y(t)` from `u(t), u(t−10)`) equals the
literature's "predict `y(t+1)` from `u(≤t)`" up to index relabeling, so it
compares directly. Orders 20/30 are **not** well-standardized — many papers wrap
the recurrence in `tanh(...)` (like the variant above) and schedule coefficients
differently — so treat their bands as a *sane-regime* check, not exact targets;
your series won't be bit-identical to any specific paper's. The gap above is still
large enough that protocol mismatch alone is an unlikely full explanation: we
have not located published NARMA-30 NRMSE numbers that sit below the strong band’s
floor, much less near 0.06 (current best **0.0570**).

## Results: memory-depth (M) sweep

The sweep holds the target series fixed and varies only `M`, the reservoir
delay-line depth (`history_depth`) — an isolated test of how much past state the
readout can exploit. Each cell is the **mean test NRMSE over 5 seeds**
(73895–73899); lower is better.

| M  | N10 D10    | N10 D12    | N20 D10    | N20 D12    | N30 D10    | N30 D12    |
|----|------------|------------|------------|------------|------------|------------|
| 1  | 0.2128     | 0.1872     | 0.6894     | 0.5671     | 0.7991     | 0.8261     |
| 2  | 0.1698     | 0.1590     | 0.4080     | 0.3637     | 0.7447     | 0.7643     |
| 4  | 0.1415     | 0.1315     | 0.3040     | 0.2826     | 0.4280     | 0.2711     |
| 8  | 0.1265     | 0.1077     | 0.2531     | 0.2192     | 0.1917     | 0.1326     |
| 16 | **0.1089** | **0.0927** | 0.2106     | 0.1825     | 0.1290     | 0.1207     |
| 24 | 0.1134     | 0.0968     | **0.1724** | **0.1588** | 0.1422     | 0.1385     |
| 32 | 0.1579     | 0.1321     | 0.1818     | 0.1663     | **0.0901** | **0.0944** |
| 48 | 0.2671     | 0.2689     | 0.2488     | 0.2523     | 0.1485     | 0.1582     |

Column key: `N10`/`N20`/`N30` give the NARMA order — NARMA-10 is the **legacy**
variant, NARMA-20 and NARMA-30 are **tanh-wrapped**; `D10`/`D12` give the
reservoir dimension — DIM-10 (1024 neurons) and DIM-12 (4096).

Against the literature bands above, `M=1` (current state only) is the plain-ESN
reference point — NARMA-10 there reads 0.213 (D10) / 0.187 (D12), at the low edge
of the cited 0.20–0.40 band. Every deeper row beats those baselines because `M`
adds an explicit tapped delay line of past reservoir states that a plain ESN
lacks; the sub-0.1 numbers reflect that added memory, not a like-for-like win.

Shared config: `sr 0.92, leak 1, input_scaling 0.5`, 600 epochs, warmup 300 /
collect 8000 (train 6400 / test 1600).

**Note — `collect` is a first-class knob.** Longer post-warmup series cut test
NRMSE on this architecture (Linear ~8k features after pool/flatten): a
single-seed NARMA-30 / DIM-10 / M=16 ladder went
`collect 8000 → 0.0787`, `16000 → 0.0713`, `32000 → ~0.06` (warmup 300,
FSF off) — the current multi-seed collect-32000 best is **0.0570**
([NARMA-30.md](NARMA-30.md)). That is mostly sample count vs readout capacity
(at 8k train is under-parameterized relative to features), not a change in
NARMA dynamics. `NARMA.cpp` comments 8000 as the low-res default and 32000 as
high-res.

Hold `collect` fixed when comparing M / seed / FSF within this repo. Classic
NARMA-10 protocols often use shorter series (train ~2k–5k); NARMA-30 paper
protocols vary widely. For apples-to-apples *internal* tables use **8000**; for
the absolute best score we report, use the **32000** spotlight config — and note
that even the 8000-point means already clear the literature strong band.

> **Untuned baseline — read these numbers accordingly.** Every cell above uses
> the **same** `sr 0.92, leak 1, input_scaling 0.5`, held fixed across *all* DIM ×
> M × order combinations. Nothing here is tuned per cell. These hyperparameters
> interact strongly with both reservoir size and memory depth, so tuning `sr` /
> `leak` / `input_scaling` for a specific DIM / M / order could yield
> **dramatically** better NRMSE than the table shows. Treat this as a single
> fixed-operating-point baseline, not a best-effort result.

**Analysis**

- **Memory depth dominates, and the curve is U-shaped.** Too few taps (`M` below
  the order) starve the lag history; too many dilute the readout with stale
  state. The optimum sits just above the order, and every column turns back up by
  `M=48` — so more taps is not freely better.
- **Every order clears the "compelling" cutoff at its `M*`.**
- **Reservoir size helps modestly.** Picking `M*` right matters more than 4× the
  neurons.
- **Seed spread collapses at depth.** Across-seed std is ≤ 0.01 at each optimum —
  deep memory stabilizes the readout across reservoir realizations as well as
  lowering error.