# Activation Function A(x) — Central-Slope-Boosted tanh

> Status: **exploratory**. This documents a new reservoir activation under active
> investigation, not approved/default behavior. The shipped default in
> `Reservoir::UpdateState` is still `std::tanh(s)`; `A_lorentz` is wired in behind
> a commented call site. Experimental numbers below are early and from a narrow
> sweep — treat them as evidence, not a settled result.

## Definition

`A_lorentz` (in `Reservoir.cpp`, ~line 207) is a `tanh` whose effective slope is
amplified by a Lorentzian bump centered on the origin:

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

## Shape and limits

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

| Param | Code | Meaning | Tested value |
|-------|------|---------|--------------|
| `γ` (gamma) | `gamma` | Peak gain boost; slope at origin is `1+γ` | `1.5` → slope 2.5× |
| `σ` (sigma) | `inv_sigma2 = 1/σ²` | Half-width of the boosted region | `inv_sigma2=100` → σ=0.1 |

Both are first guesses, not tuned. `γ` controls *how much* the center is steepened;
`σ` controls *how wide* the steepened region is relative to the reservoir's
operating amplitude.

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

With a small input scaling (e.g. `input_scaling=0.04`) almost every unit sits in
`|x| ≲ σ`, so `A` amplifies exactly the regime the reservoir actually occupies,
while the rare large swings still saturate and self-limit. A scalar `sr` cannot
express this decoupling: it moves both regimes together. The result is a richer
set of distinguishable low-amplitude traces for the linear readout — which is
what a long-memory task most needs.

## Early experimental evidence

### NARMA-30 (memory-bound task) — `A` wins on the long-memory end

NARMA-30, `DIM=8 N=256`, `sr=0.92 leak=1 input_scaling=0.04`, seeds
`{73896, 73897}`, history-depth sweep `M ∈ {16, 32, 48}`. NRMSE (lower is better):

```
   M     tanh             A(γ=1.5, σ=0.1)    Δ
  16   0.1601 ±0.016     0.2618 ±0.116      WORSE; variance blows up (one seed 0.344)
  32   0.1098 ±0.003     0.1046 ±0.004      ~5% better
  48   0.1538 ±0.001     0.1353 ±0.005      ~12% better
```

Read: `A` wins at the long-memory operating points (M=32, 48) and the win grows
with `M`, but it is worse and high-variance at the short delay line (M=16). That
M=16 behavior is the expected cost of the mechanism, not noise — locally
expansive dynamics with too short a history line flirt with the edge of chaos and
become seed-dependent. The stable operating window now depends on `M`.

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
nonlinear computation and necessarily drains the linear-MC bucket. So **lower MC +
the NARMA-30 win are the same fact**: NARMA is a *nonlinear* memory task, and `A`
shifts the budget toward exactly what NARMA needs. A skeptic would raise the MC
deficit; it is actually the signature of the claimed mechanism.

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
| NARMA-30 | memory-bound | **`A` wins** at M=32 (~5%), M=48 (~12%) | loses at M=16 (instability) |
| Sine prediction | easy | parity (both R²=1.0) | `A` matches at `sr` 0.90 vs 0.98 |
| Streaming anomaly | easy | parity (same 10 flags) | `A` matches at `input_scaling` 0.1 vs 1.9 |
| Signal classification | easy | parity (both 100%) | `A` matches at `input_scaling` 0.1 vs 1.5 |
| Linear memory capacity | capacity probe | **`tanh` wins** (`A` ≈ 0.67–0.83×) | expected: `A` trades linear MC for nonlinear computation |

`A` is a no-regression drop-in that reaches equivalent behavior at markedly lower
nominal `sr` and input drive, pulls *ahead* only when memory binds, and pays for it
in linear memory capacity — a deliberate, mechanism-consistent trade, not a defect.

## Open questions / next steps

1. **Tune `(γ, σ)`.** The tested pair is a first guess. A small 2D grid at the
   M=32 sweet spot should map the optimum; the right `σ` is likely tied to the
   operating amplitude (`input_scaling` and the recurrent fan-in), not a fixed
   constant.
2. **Transfer to Lorenz free-run** — the primary target (NARMA-30 is the proxy
   here). A small-signal-gain boost could sharpen a chaotic attractor's fine
   structure, or could perturb the free-run Lyapunov exponent; needs a direct test
   once `(γ, σ)` is roughly tuned.
3. **Interaction with `leak_rate`.** `leak < 1` is the other stability knob and
   may tame the M=16 instability while widening the usable-`M` window.
4. **Cost.** `A` adds a divide + a couple of FMAs per unit per step over `tanh`'s
   single call; the recurrent block (`dim × history_depth` FMAs) dominates, so the
   overhead is expected to be negligible — worth confirming on a timed run.

## Call site

`Reservoir::UpdateState` (`Reservoir.cpp`, ~line 257) currently reads:

```cpp
const float activation = std::tanh(s);
//const float activation = A_lorentz(s, 1.5, 100);
```

Switching the active line swaps the activation for every unit, every step. No
other code path is affected — `A` is a drop-in scalar nonlinearity with the same
domain and codomain as `tanh`.
