# Activation Function A(x) — Central-Slope-Boosted tanh

> Status: **shipped path, exploratory use.** `Reservoir::UpdateState` always
> calls `A_lorentz(s, lorentz_gamma, lorentz_inv_sigma2)`. The product default is
> `lorentz_gamma = 0`, which recovers **plain `tanh(s)` exactly**. Nonzero γ is
> the experimental regime documented below — early, from narrow sweeps; treat
> those numbers as evidence, not a settled product default.

Related config (see [Reservoir.md](Reservoir.md) / `ReservoirConfig`):

| Field | Default | Role |
|-------|---------|------|
| `lorentz_gamma` (γ) | `0.0` | Peak central gain boost; **0 = off** (plain tanh) |
| `lorentz_inv_sigma2` (1/σ²) | `250.0` | Lorentzian width (σ ≈ 0.063 at default) |

Bias is applied **after** A: `activation = A_lorentz(s, …) + bias[v]`, then the
leak blend. Changing γ does not change the bias path.

## Definition

`A_lorentz` (inline in `Reservoir.cpp`) is a `tanh` whose local slope is scaled
by a Lorentzian bump centered on the origin:

```
  A(x) = tanh( x · g(x) ),     g(x) = 1 + γ · φ(x),     φ(x) = 1 / (1 + x²/σ²)
```

In code the width is passed as `inv_sigma2 = 1/σ²`:

```cpp
inline float A_lorentz(float x, float gamma, float inv_sigma2) noexcept
{
    const float phi  = 1.0f / (1.0f + x * x * inv_sigma2);
    const float gain = 1.0f + gamma * phi;
    return std::tanh(x * gain);
}
```

`g(x)` is a unit-baseline gain with a Lorentzian peak of height `γ` and
half-width `σ` at the origin. `φ` is the Lorentzian (Cauchy) kernel — chosen
over a Gaussian deliberately: **no `exp`, heavier tails**, so the gain decays
back to 1 gradually rather than abruptly.

**Sign of γ (from code comments):**

- `γ = 0` → `g ≡ 1` → plain `tanh(x)` (shipped default).
- `γ > 0` → steeper central slope, tanh tails (sharpening) — the regime in the
  experiments below.
- `γ < 0` with `|γ| > 1` → central gain can cross zero → non-monotone “fold”.
  Not used in the tables below; available as a sweep axis.

## Shape and limits (γ > 0)

```
   A(x)
    1 |- - - - - - - - - -=========   ← saturates exactly like tanh (g→1)
      |                ,-'
      |             ,-'
      |          ,/'   ← slope ≈ (1+γ) near origin: steeper than tanh
   ---+---------/-----------------  x
      |      ,/'
      |   ,-'
      |,-'
   -1 ==========- - - - - - - - - -
        |<-- ~σ -->|
        boosted region    tails = plain tanh
```

- **Near the origin** (`|x| ≪ σ`): `φ→1`, `g→1+γ`, so `A(x) ≈ tanh((1+γ)x)`.
  The slope at the origin is exactly **`A'(0) = 1 + γ`** (the bump's `x·g'(x)`
  term vanishes at 0, leaving `sech²(0)·g(0) = g(0)`).
- **In the tails** (`|x| ≫ σ`): `φ→0`, `g→1`, so `A(x) → tanh(x)` — identical
  saturation, identical self-limiting behavior, bounded in `(-1, 1)`.
- **`γ = 0`**: recovers plain `tanh` exactly. `γ` is a clean "off" knob.
- **Monotonicity**: `A` is strictly increasing for `γ < 8`. (The bump's
  contribution to `d/dx[x·g(x)]` bottoms out at `−γ/8` at `x² = 3σ²`, so the
  derivative `h'(x) ≥ 1 − γ/8` stays positive below `γ = 8`.) The tested
  `γ = 1.5` is comfortably inside the well-behaved regime.

## The two parameters

| Param | Config / code | Meaning | Values used in tables below |
|-------|---------------|---------|-----------------------------|
| `γ` (gamma) | `lorentz_gamma` | Peak gain boost; slope at origin is `1+γ` | e.g. `1.1`, `1.4`, `1.5` (not the default `0`) |
| `σ` (sigma) | `lorentz_inv_sigma2 = 1/σ²` | Half-width of the boosted region | e.g. `100` (σ=0.1), `~250` (σ≈0.063); default inv_sigma2 is `250` |

Hand-found per task in early sweeps, not jointly tuned. `γ` controls *how much*
the center is steepened; `σ` controls *how wide* the steepened region is relative
to the reservoir's operating amplitude.

## Why this is not just a higher spectral radius

The obvious objection: a steeper central slope raises the small-signal loop gain,
so isn't this equivalent to raising `sr`? **Empirically no** — raising `tanh`'s
`sr` produced no improvement on the same task (tested). The reason is structural:

```
  raise sr (uniform):     scales ALL recurrent weights → small-signal gain ↑
                          BUT large excursions ↑ too → more units pinned in the
                          saturated tanh tails → memory destroyed. Net wash.

  A(x) (region-selective): small-signal gain ↑ only where |x| ≲ σ
                           large excursions UNCHANGED (g→1 in the tails)
                           → expands near-origin dynamic range WITHOUT paying
                             in extra saturation.
```

With a small input scaling (e.g. `input_scaling≈0.02`) almost every unit sits in
`|x| ≲ σ`, so `A` amplifies exactly the regime the reservoir actually occupies,
while the rare large swings still saturate and self-limit. A scalar `sr` cannot
express this decoupling: it moves both regimes together. This is why `A` reaches
the same dynamics as `tanh` at a much gentler external drive (documented per-task
below) — it supplies the small-signal gain internally instead of buying it with
input amplitude or `sr`. Note this is an operating-point claim, not a task-error
claim: at matched (best) operating points `A` and `tanh` perform the same on every
task tested (see NARMA-30 below); the decoupling buys a gentler drive, not lower
error.

## Early experimental evidence

Experiments below compare **nonzero γ** (labelled `A`) against **plain tanh**
(γ = 0, or an equivalent pure-tanh build). They are not the current default
config unless you set `lorentz_gamma` yourself.

### NARMA-30 (memory-bound task) — parity, reached at a far gentler drive

This is the hard, memory-binding task and the one that earns the most scrutiny.
NARMA-30, `DIM=8 N=256 leak=1`, 3 seeds `{73896, 73897, 73898}`, 600 epochs
(batch 128, cosine lr), history-depth sweep `M ∈ {28,…,36}` around the optimum.
Each activation is tuned to its **own** best operating point: `A(γ=1.1, σ≈0.063)`
at `sr=0.92, input_scaling=0.019`; `tanh` at `sr=0.95, input_scaling=0.1`. NRMSE
(mean over seeds, lower is better):

```
   M     A mean    tanh mean    Δ=A−tanh        seed std (A / tanh)
  28     0.1606    0.1594       +0.0012  ↑      0.0033 / 0.0015
  30     0.1101    0.1118       −0.0017  ↓      0.0024 / 0.0032
  32     0.1019    0.1022       −0.0003  ≈      0.0040 / 0.0036
  34     0.1017    0.1044       −0.0027  ↓      0.0037 / 0.0034
  36     0.1085    0.1080       +0.0005  ≈      0.0011 / 0.0026
  ----   ------    ------
  mean   0.1166    0.1172       −0.0006         (0.5%, well inside the seed std)
```

**This is parity.** Every per-`M` gap is smaller than the seed-to-seed std; `A` is
a hair ahead in the `M=30–34` band and a hair behind at the edges, netting to 0.5%
overall — noise. Both bottom out at NRMSE ≈ 0.10 (R² ≈ 0.99) at the same optimal
`M ≈ 32–34`; the best single run is `A`'s 0.0975 with `tanh`'s 0.0984 right beside
it.

> **Supersedes an earlier claim.** A first, coarser sweep (`M ∈ {16,32,48}`, 2
> seeds, `tanh` at `input_scaling=0.04`) showed `A` ahead by ~5% at M=32 and ~12%
> at M=48 and was written up here as "`A` wins when memory binds." That gap did
> **not survive proper tuning of `tanh`**: `input_scaling=0.04` under-drove `tanh`
> (it wants ~0.1), and at its real operating point `tanh` improves from ~0.110 to
> ~0.102, erasing the margin. The honest result is parity, not a memory win.

What *does* survive is the through-line of every other task: `A` reaches the
identical floor at **~5× gentler input** (`0.019` vs `0.1`) and lower `sr` (`0.92`
vs `0.95`). The differentiator is the operating point, not the task error.

### Sine-wave prediction (easy task) — parity, at a lower nominal `sr`

Predict `sin(0.1t)` next-step from raw reservoir state. `DIM=8 N=256 M=16 leak=1
input_scaling=0.1`, 1500 epochs, two seeds. `A(γ=1.4, σ≈0.071)` tuned to
`sr=0.9`; `tanh` tuned to `sr=0.98`. NRMSE (lower is better):

```
   seed            tanh @ sr=0.98     A(γ=1.4) @ sr=0.90
   73895           0.000046           0.000070
   84745874578     0.000081           0.000085
```

All four runs hit R²=1.000000. The NRMSE spread is noise — both activations are
pinned at the readout/numerical floor, because sine-from-state is an easy,
low-memory task where the nonlinearity is not the bottleneck. The result is a
**tie, and the informative part is the `sr` offset**: `A` reaches the same floor
at `sr=0.90` where `tanh` needs `sr=0.98`. That ~0.08 gap is `A`'s `1+γ = 2.4`
central slope supplying the extra small-signal loop gain internally, so nominal
`sr` is dialed down to land on the same effective dynamics:

```
  A(γ=1.4) at sr=0.90   ≈   tanh at sr=0.98     (matched effective small-signal gain)
```

Two takeaways: (1) `A` is a **safe drop-in — no regression** on an easy task; and
(2) this is not in tension with "raising tanh's `sr` did nothing" on NARMA-30. On
the trivial task `sr` barely matters, so the offset is bookkeeping; on the
memory-bound task uniform `sr` couldn't buy *non-saturated* gain, which is exactly
the regime `A`'s region-selective boost is built for.

### Streaming anomaly detection (easy task) — parity at far gentler drive

Learn a multi-harmonic "normal" signal, then flag windows whose next-step RMSE
exceeds 10× baseline. `DIM=8 N=256 M=16 leak=1`, 1000 epochs, seed 73895,
30 windows (9 true-anomaly + 1 washout expected). `A(γ=1.4, σ≈0.071)` at
`sr=0.90, input_scaling=0.1`; `tanh` at `sr=0.99, input_scaling=1.9`.

```
                 baseline RMSE   flagged windows   anomaly RMSE ratios
   tanh            0.006124          10/10          11.8×, 50.1×, 40.3× (noise/DC/freq)
   A(γ=1.4)        0.006149          10/10          11.9×, 45.7×, 38.1×
```

Identical detection — same 10 windows, ratios within ~5%, same baseline and
threshold. Anomaly detection here is easy for both (ratios sit 12×–50× against a
10× threshold — large margin either way), so a tie is the expected outcome and the
nonlinearity is again not the bottleneck.

The informative part is the **operating-point offset**: `tanh` must be driven hard
(`input_scaling=1.9`, well into saturation) to separate normal from anomalous in
state space; `A` reaches the same separation at a **19× gentler input** (`0.1`)
while sitting in its boosted small-signal regime — same `~0.09` `sr` offset as the
sine task on top. `A` trades external drive for internal gain.

Note the 19× input-scaling ratio is **not** the `1+γ = 2.4` pointwise slope:
`input_scaling` and central slope compound nonlinearly through the recurrent loop
and the saturation level over a window, so the operating-point offset is much
larger than the pointwise gain and is not a clean function of `γ`.

### Signal classification (easy task) — parity at ceiling

Classify input waveforms from reservoir state. `A(γ=1.4, σ≈0.071)` at
`sr=0.90, input_scaling=0.1`; `tanh` at `sr=0.99, input_scaling=1.5`. **Both reach
100% accuracy.** Accuracy is a ceiling metric, so once both saturate it cannot
show a margin — parity is the expected outcome and the only distinguishing signal
is, once more, the operating point: `A` hits the ceiling at a ~15× gentler input
drive (`0.1` vs `1.5`) and lower `sr` (`0.90` vs `0.99`).

### Memory capacity (linear) — `tanh` wins, and that *supports* the thesis

Linear memory-capacity profile, `DIM=11 N=2048 F=2048`, `leak=1`, warmup 2000 /
collect 15000, `Kmax=2000`, ridge `1e-4`. Swept `sr ∈ {0.90, 0.95, 1.00, 1.10}`
and history depth `M ∈ {1,2,4,8,16,32,64}`. TotalMC (higher = more linear memory;
bounded above by `N=2048`). This is the one metric where `tanh` cleanly beats `A`
— and it is the *expected* price of the mechanism, not a counterexample to it.

**`A` retains only ~⅔–⅘ of `tanh`'s linear MC** (matched `input_scaling=0.2`,
`sr=1.00`):

```
   M      tanh        A(γ=1.4,σ≈0.071)    A/tanh
   1      41.14        27.37              0.67
   4     101.78        77.44              0.76
  16     214.37       178.89              0.83
  64     751.78       520.91              0.69
```

Read this through the **memory–nonlinearity tradeoff** (Dambre et al. 2012: total
computational capacity is conserved; linear memory and nonlinear processing draw on
one budget). MC scores *linear* reconstruction of past inputs. `A`'s central-slope
boost makes the reservoir more expansively nonlinear, which spends budget on
nonlinear computation and drains the linear-MC bucket. So the MC deficit is not a
defect — it is the direct, measurable signature that `A` *reallocates* capacity
from linear memory toward nonlinear processing, exactly as the mechanism predicts.
What it does **not** buy is a task-error win: on NARMA-30 (a nonlinear memory task,
where the reallocation should help if anywhere) `A` and `tanh` land at parity once
both are tuned. So the honest reading is that `A` shifts *where* the capacity sits
without changing *how much* useful work the readout extracts on these tasks — a
genuine change in the reservoir's character, not a free lunch.

Two secondary trends both reinforce the picture:

- **The whole memory story lives at `sr=1.00`, the edge of chaos.** MC scales
  superlinearly with `M` only there; at `sr≤0.95` it plateaus (~80) because the
  reservoir holds no long correlations to extend. `sr=1.10` is past the echo-state
  boundary, but the instability is M-gated — it even leads at `M=1`, then craters
  as the delay integral runs long enough to see the divergence:

  ```
   M      sr=1.00      sr=1.10
   1        55.22       53.92    ← tied; 1.10 not yet diverging
  16       281.11        1.13    ← gone
  64      1003.60        0.00    ← dead          (tanh, input_scaling=0.06)
  ```

- **`A` is nearly immune to `input_scaling`; `tanh` is not.** Shrinking the drive
  is *how you keep `tanh` linear* → more MC (`tanh` M=64 sr=1.00: `is=0.06`→1003.6
  vs `is=0.2`→751.8, +34%). `A` barely moves and not even monotonically
  (`is=0.1`→458 vs `is=0.2`→521 at M=64). `A` has already raised the small-signal
  gain *internally*, so the knob that most helps `tanh`'s linear memory does almost
  nothing for `A` — the same decoupling seen on the `sr` axis, now on the input
  axis.

### Summary across tasks

| Task | Regime | Result | Notable |
|------|--------|--------|---------|
| NARMA-30 | memory-bound | parity (≈0.10 NRMSE, Δ 0.5% ⊂ noise) | `A` matches at `input_scaling` 0.019 vs 0.1 |
| Sine prediction | easy | parity (both R²=1.0) | `A` matches at `sr` 0.90 vs 0.98 |
| Streaming anomaly | easy | parity (same 10 flags) | `A` matches at `input_scaling` 0.1 vs 1.9 |
| Signal classification | easy | parity (both 100%) | `A` matches at `input_scaling` 0.1 vs 1.5 |
| Linear memory capacity | capacity probe | **`tanh` wins** (`A` ≈ 0.67–0.83×) | `A` trades linear MC for nonlinear computation |

The honest bottom line: across every task tested — easy and memory-bound alike —
nonzero-γ `A` is a **no-regression drop-in that matches `tanh` on task error at a
markedly gentler operating point** (lower `sr`, ~5–19× smaller input drive). It
does not beat `tanh` on error anywhere once `tanh` is properly tuned. Its
distinguishing, measurable property is the reallocation of capacity (lower linear
MC, gentler drive to reach the same dynamics), not a performance win. Whether that
gentler operating point is *useful* — e.g. for dynamic range, quantization, or
hardware drive limits — is the open question that would justify nonzero γ over
plain `tanh` (γ = 0).

## Call site (current code)

`Reservoir::UpdateState` always applies:

```cpp
const float activation =
    A_lorentz(s, lorentz_gamma_, lorentz_inv_sigma2_) + vtx_bias_[v];
```

There is no separate commented `std::tanh` branch. Set knobs on `ReservoirConfig`
(or ESN’s reservoir config) before construction:

```cpp
cfg.reservoir.lorentz_gamma = 1.1f;           // e.g. NARMA-tuned exploratory
cfg.reservoir.lorentz_inv_sigma2 = 250.0f;    // 1/σ²  (σ ≈ 0.063)
// lorentz_gamma = 0.0f  → plain tanh (default)
```

No other code path is affected — `A` is a drop-in scalar nonlinearity with the same
codomain as `tanh` when γ ≥ 0 and γ is below the monotonicity bound.

## Open questions / next steps

1. **Is the gentler operating point actually worth anything?** This is now the
   central question. Nonzero-γ `A` matches `tanh` on error but reaches it at lower
   `sr` and far smaller input drive. That only matters if some downstream
   constraint cares — fixed-point/quantized state, analog or hardware drive
   limits, dynamic-range headroom. Absent such a constraint, plain `tanh` (γ = 0)
   is the simpler choice.
2. **Tune `(γ, σ)` jointly with the operating point.** Values tried so far
   (`(1.5,σ=0.1)`, `(1.4,σ≈0.071)`, `(1.1,σ≈0.063)`) were hand-found per task. A
   real 2D grid — co-swept with `input_scaling`/`sr`, since the right `σ` tracks
   operating amplitude — would say whether any `(γ,σ)` breaks parity into an actual
   error win, or confirm that none does.
3. **Transfer to Lorenz free-run** — the primary target (NARMA-30 is the proxy
   here). A small-signal-gain boost could sharpen a chaotic attractor’s fine
   structure, or could perturb the free-run Lyapunov exponent; needs a direct test.
4. **Interaction with `leak_rate`.** `leak < 1` is the other stability knob; worth
   a sweep alongside `(γ, σ)`.
5. **Cost.** `A` adds a divide + a couple of FMAs per unit per step over a lone
   `tanh`; the recurrent block (`dim × history_depth` FMAs) dominates, so overhead
   is expected to be small — worth a timed run if γ becomes default-nonzero.
6. **Negative γ / fold regime.** Supported in code; not characterized in the tables
   above.
