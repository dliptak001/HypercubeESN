# Activation Function A(x) — archived note

> **Status: archive only.** Implementation removed from the product.
>
> The live reservoir uses plain `std::tanh` in `Reservoir::UpdateState`.
> This note keeps the definition, the math, and the measured operating-point
> finding. It is not a current API guide and not a claim that A beats tanh on
> task error.

## Finding

A region-selective central-slope boost on top of `tanh` — call it **A** — was
compared to plain `tanh` on several HypercubeESN tasks. Once each activation was
tuned to its own operating point, **task error was at parity** (NARMA-30, sine,
anomaly, classification). **A did not win on error.** What did differ was the
operating point: A reached that parity at **lower nominal spectral radius** and
**much lower input scaling** (about 5× on the retuned NARMA-30 pair; larger
ratios on easy tasks where tanh was driven hard). Linear memory capacity (Jaeger
MC) moved the other way: **tanh held more linear memory** (A roughly ⅔–⅘ of
tanh at matched `is`/`sr` in the tables below). So the honest summary is
*reparameterization of drive / small-signal gain*, not a free performance win.
That is why the knobs were dropped from the product and this note was kept.

## Definition (historical)

When A was in-tree, pre-activation `s` went through:

```
A(x) = tanh( x * g(x) )
g(x) = 1 + gamma * phi(x)
phi(x) = 1 / (1 + x^2 / sigma^2)     # Lorentzian / Cauchy kernel
```

Code stored width as `inv_sigma2 = 1/sigma^2`:

```cpp
// historical — no longer in Reservoir.cpp
inline float A_lorentz(float x, float gamma, float inv_sigma2) noexcept
{
    const float phi  = 1.0f / (1.0f + x * x * inv_sigma2);
    const float gain = 1.0f + gamma * phi;
    return std::tanh(x * gain);
}
```

| Symbol | Historical config field | Role |
|--------|-------------------------|------|
| `gamma` | `lorentz_gamma` (default **0**) | Peak central gain; `gamma = 0` ⇒ plain `tanh` |
| `sigma` | via `lorentz_inv_sigma2 = 1/sigma^2` (default **250** ⇒ sigma ≈ 0.063) | Width of the boosted band |

Bias was always **after** the nonlinearity: `activation = A(s) + bias[v]`, then
the leak blend. Bias path was independent of `gamma`.

**Sign of gamma**

| gamma | Behavior |
|-------|----------|
| `0` | `g ≡ 1` → plain `tanh(x)` |
| `> 0` | Steeper near origin; tails still `tanh` (regime in the tables) |
| `< 0`, `|gamma| > 1` | Central gain can cross zero → non-monotone fold; not used in tables |

Lorentzian (vs Gaussian) was chosen for no `exp` and heavier tails so gain
returns to 1 gradually. That is an implementation preference, not a task claim.

## Math checks

### Origin slope

Let `h(x) = x * g(x)`. Then `A(x) = tanh(h(x))` and

```
A'(0) = sech^2(0) * h'(0) = h'(0)
```

At `x = 0`, `phi = 1`, `g = 1 + gamma`, and the `x * g'(x)` contribution to
`h'` vanishes, so **`A'(0) = 1 + gamma`**. Near the origin,
`A(x) ≈ tanh((1+gamma) x)`.

### Half-width of phi

`phi(x) = 1/2` when `x^2 = sigma^2`, i.e. `|x| = sigma`. So `sigma` is the
half-width at half-maximum of the gain bump on the pre-activation axis — not a
guarantee about post-activation `|state|`.

### Monotonicity for gamma > 0

With `u = x^2 / sigma^2`:

```
h'(x) = 1 + gamma * (1 - u) / (1 + u)^2
```

For `u ≥ 0` the second term is minimized at `u = 3` with value `-gamma/8`, so

```
h'(x) ≥ 1 - gamma/8
```

Thus **`h'` (and therefore `A'`) stays positive for `0 ≤ gamma < 8`**. Tested
values (`gamma` up to 1.5) sit well inside that regime. (At `gamma = 8` the bound
touches zero at one radius; above 8, `h'` can go negative.)

### Tails

As `|x| → ∞`, `phi → 0`, `g → 1`, so `A(x) → tanh(x)`. Codomain stays `(-1, 1)`.

## Structural idea (not a theorem about task error)

Raising spectral radius scales **all** recurrent weights: small-signal gain and
large excursions move together, so more saturation can erase memory. A with
`gamma > 0` raises gain mainly where `|x| ≲ sigma` and leaves deep tails like
`tanh`. That is a **region-selective** reweighting of the nonlinearity, not the
same one-parameter move as `sr`.

**What this does *not* prove by itself**

- That A must beat tanh on NARMA or any other score.
- That “effective loop gain” equals `sr * (1+gamma)` in closed form — spectral
  radius is measured on the **linear recurrent** block *before* the nonlinearity;
  A is nonlinear. Matching operating points is empirical, not algebraic.
- That every unit lives in `|x| ≲ sigma` at low `input_scaling` without state
  histograms (plausible at weak drive; not measured in this note).

## Evidence (historical campaigns)

Protocols below predate removal of the knobs. Each activation was tuned to its
**own** best operating point unless noted. Numbers are as recorded in the original
campaign notes; they are not re-run against the current tree.

### NARMA-30 — parity at lower drive

- Setup: dim 8, N=256, leak=1, 3 seeds `{73896, 73897, 73898}`, 600 epochs
  (batch 128, cosine LR), M in `{28,…,36}`.
- A: `gamma=1.1`, `sigma≈0.063`, `sr=0.92`, `input_scaling=0.019`.
- tanh: `sr=0.95`, `input_scaling=0.1`.
- NRMSE mean over seeds (lower better):

```
   M     A mean    tanh mean    Δ = A - tanh     seed std (A / tanh)
  28     0.1606    0.1594       +0.0012          0.0033 / 0.0015
  30     0.1101    0.1118       -0.0017          0.0024 / 0.0032
  32     0.1019    0.1022       -0.0003          0.0040 / 0.0036
  34     0.1017    0.1044       -0.0027          0.0037 / 0.0034
  36     0.1085    0.1080       +0.0005          0.0011 / 0.0026
  mean   0.1166    0.1172       -0.0006          (~0.5%; inside seed noise)
```

**Read as parity.** Per-M gaps sit inside seed std. Both bottom near NRMSE ≈ 0.10
around M ≈ 32–34.

> **Superseded claim.** An earlier coarse sweep (M in `{16,32,48}`, 2 seeds, tanh
> at `input_scaling=0.04`) looked like an A win on memory. That gap **did not
> survive** retuning tanh (`input_scaling≈0.1`). The retained result is parity, not
> “A wins when memory binds.”

What **does** remain: same floor at about **5× lower input scaling** (0.019 vs 0.1)
and slightly lower `sr` (0.92 vs 0.95). Differentiator = operating point, not error.

### Sine next-step — parity (ceiling / floor), lower nominal `sr`

- dim 8, M=16, leak=1, `input_scaling=0.1`, 1500 epochs, two seeds.
- A: `gamma=1.4`, `sigma≈0.071`, `sr=0.90`.
- tanh: `sr=0.98`.

Both runs hit R² = 1.0; NRMSE is at numerical floor. Nonlinearity is not the
bottleneck. Informative detail: A used lower nominal `sr`. Interpreting that gap
as “matched small-signal loop gain” is a **heuristic** consistent with
`A'(0)=1+gamma`, not a derived identity.

### Streaming anomaly — parity detection, lower drive

- Same architecture family; A at `sr=0.90`, `input_scaling=0.1`; tanh at
  `sr=0.99`, `input_scaling=1.9`.
- Same 10 flagged windows; baseline RMSE ~0.006 either way; anomaly ratios within
  ~5%.

Task is easy for both (large margin over 10× threshold). The large input-scaling
ratio (~19×) is **not** equal to `1+gamma`; drive and central slope compound through
the loop. Record it as an empirical OP offset only.

### Signal classification — both at 100%

A at `input_scaling=0.1`, `sr=0.90`; tanh at `input_scaling=1.5`, `sr=0.99`.
Accuracy is a ceiling metric → no margin to show. Again, OP differs; error does not.

### Linear memory capacity — tanh wins

- dim 11, N=F=2048, leak=1, warmup 2000 / collect 15000, Kmax 2000, ridge 1e-4.
- Matched `input_scaling=0.2`, `sr=1.00`; A with `gamma=1.4`, `sigma≈0.071`.
- TotalMC (higher = more linear memory; ceiling N=2048):

```
   M      tanh        A            A/tanh
   1      41.14        27.37        0.67
   4     101.78        77.44        0.76
  16     214.37       178.89        0.83
  64     751.78       520.91        0.69
```

**Interpretation (careful).** Lower linear MC under a more expansive central
nonlinearity is **consistent with** the usual memory–nonlinearity tradeoff
(linear reconstruction of past inputs vs nonlinear computation). That is a useful
reading, not a formal proof that “total capacity is conserved” on this stack (MC
meter vs HCNN readout are different probes). What is solid: A did **not** turn the
MC deficit into a NARMA-30 error win once tanh was tuned.

Secondary notes from the same campaign family (tanh tables unless noted):

- Long-M MC story is sharpest near `sr ≈ 1.0`; `sr = 1.10` collapses at large M
  (instability / loss of useful memory).
- Shrinking `input_scaling` helps tanh linear MC more than A’s (A already raises
  small-signal gain inside the unit). Numbers: tanh M=64 `sr=1.00`:
  `is=0.06` → TotalMC 1003.6 vs `is=0.2` → 751.8; A at M=64 moved little and not
  monotonically (`is=0.1` → ~458 vs `is=0.2` → ~521 in the notes).

## Summary table

| Task | Result | What differed |
|------|--------|----------------|
| NARMA-30 | parity (~0.10 NRMSE) | A: `is` 0.019 vs tanh 0.1; `sr` 0.92 vs 0.95 |
| Sine | parity (R² = 1) | A lower nominal `sr` |
| Streaming anomaly | parity (same flags) | A much lower `input_scaling` |
| Classification | both 100% | A lower `input_scaling` / `sr` |
| Linear MC | **tanh higher TotalMC** | A more nonlinear at center |

## Product disposition

| Keep | Drop |
|------|------|
| This note and the operating-point finding | Live `lorentz_*` knobs and `A_lorentz` |
| Plain `tanh` as the only reservoir activation | Any marketing of A as a NARMA “win” |

**Revisit in code only if** a concrete constraint needs lower external drive
(quantization, hardware / analog limits, dynamic-range headroom) and a controlled
A/B is run against tuned tanh — not because of error tables above.

## Historical call site

```cpp
// was:
activation = A_lorentz(s, lorentz_gamma_, lorentz_inv_sigma2_) + bias;

// is:
activation = std::tanh(s) + bias;
```
