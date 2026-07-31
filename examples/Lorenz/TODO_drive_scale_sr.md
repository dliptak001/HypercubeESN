# TODO — drive scale / spectral radius / per-channel gains

**Context:** DIM12 + M10/M12 already look respectable on Unseen free-run
(top-10% VPT×duty). Architecture-side bets (external feedback, richer drive
layouts) did not move us the right way. Remaining cheap levers are **dynamics
knobs** on the existing 4-in `[x, y, z, x*z]` stack — not new ports or layers.

**Date parked:** 2026-07-30  
**Status:** (4) gains + `Campaign_DriveGainAB` ready. (3) `Campaign_SpectralRadiusAB`
ready. (1) still config-only A/B.

**Out of scope (already known bad or not believed useful here):**

| Idea | Why parked |
|------|------------|
| `LEAK_RATE` < 1 | Always detrimental in this storefront |
| Train-time drive noise | Always detrimental |
| Scheduled sampling / “anti-overfit free-run mix” | Random multi-orbit ICs already mitigate open-loop overfit; not the failure mode we see |

**Protocol for every arm:** one change at a time; fixed DIM/M/epochs/seed set;
Unseen FreeRunSurvey (or short SeedSweep); report top-10% freerun means
(VPT, duty, VPT×duty, RMSE). Prefer matched seeds/orbits across arms.

---

## Baseline (current)

```text
DriveLayout::XyzXz   →  [x, y, z, x*z]
INPUT_SCALING        = 0.04   (global, all channels)
SPECTRAL_RADIUS      = 0.99
LEAK_RATE            = 1.0    (leave alone)
```

No per-channel gains. Free-run rebuilds products from predicted `(x,y,z)`.

---

## 1. Global `INPUT_SCALING` grid  (config-only — do first)

Free-run is usually more sensitive to drive scale than to almost anything else
that does not change architecture.

| Try | Notes |
|----:|-------|
| 0.02, 0.03 | quieter drive → less closed-loop error amp |
| **0.04** | current baseline |
| 0.05, 0.06, 0.08 | stronger drive (watch saturation / worse duty) |

**How:** flip `config::INPUT_SCALING`, or pass `input_scaling` into `SeedSweep`
(>0 overrides for the sweep; 0 keeps config). Keep SR and layout fixed for a
pure scale A/B.

**Done when:** best scale is locked (or baseline confirmed) on survey-scale
Unseen metrics, documented next to the seed/M that won.

---

## 3. Spectral radius a notch off the edge  (campaign ready)

```text
SPECTRAL_RADIUS = 0.99   // house default; reassignable for A/B
```

Near-edge SR can amplify prediction error in closed loop. Mild contraction is
cheap to test and often flat — still worth one matched pass **after** (1).

| Try |
|----:|
| 0.95, 0.97, 0.98, **0.99** |

**How:** `Campaign_SpectralRadiusAB(dim, M, sr_a, sr_b, threads, runs)` —
matched SeedSurvey per arm, roll-up + `SrAB_*.csv/txt` under `RESULTS_DIR`.
Example (DIM12 M12, baseline vs contractive):

```cpp
return Campaign_SpectralRadiusAB(/*dim=*/12, /*history_depth=*/12,
                                 /*sr_a=*/0.95f, /*sr_b=*/0.99f,
                                 /*num_threads=*/0, /*num_runs=*/50);
```

Or flip `config::SPECTRAL_RADIUS` for a single-arm run.

**Done when:** clear win, or “flat / baseline is fine” written down so we stop
retuning SR.

---

## 4. Per-channel input gains  (campaign ready — tuning open)

**Status:** `INPUT_SCALE_CH` + `Campaign_DriveGainAB` landed. Default all `1.0`.

**Problem:** one global `INPUT_SCALING` treats `x`, `y`, `z`, and `x*z` the same.
Normalized stream is ~[-1,1] per channel, but product scale and channel roles
still differ; a soft per-channel multiplier is the usual next step without
adding ports.

**Wiring:**

```text
FillDrive → features → drive[i] *= INPUT_SCALE_CH[i]
ReservoirStep then multiplies by global INPUT_SCALING (and fan-in)
```

- `config::INPUT_SCALE_CH[kMaxDriveChannels]` — layout feature order
- Train + free-run both go through `FillDrive`
- Banners / campaign metadata print `drive_ch=[...]`
- Weights stem still `in4` / `in8` only — **document gains in results**; train and
  load-time gains must match or freerun is meaningless

Suggested first grid (multipliers on top of locked global scale from (1)):

| Channel | Index (XyzXz) | First try |
|---------|--------------:|-----------|
| x, y | 0, 1 | 1.0 (reference) |
| z | 2 | 0.7 … 1.2 |
| x*z | 3 | 0.5 … 1.0 |

**How:** matched A/B via campaign (lists size = `n_in` for current drive layout):

```cpp
return Campaign_DriveGainAB(/*dim=*/12, /*history_depth=*/12,
                            /*gains_a=*/{1.f, 1.f, 1.f, 1.f},
                            /*gains_b=*/{1.f, 1.f, 0.9f, 0.7f},
                            /*num_threads=*/0, /*num_runs=*/50);
```

Or assign `config::INPUT_SCALE_CH[i]` for a single-arm run. Coarse 2D
(z-gain × xz-gain) is enough; do not start a 4D grid. Pair further candidates
as new B arms against the locked winner.

**Tuning done when:** clear Unseen top-10% lift vs global-only baseline, or a
documented negative so we stop chasing per-channel gains.

---

## Suggested order

1. **INPUT_SCALING** grid @ DIM12, best M (10 or 12), 1–3 seeds  
2. Best scale → **SR** {0.95, 0.97, 0.98, 0.99}  
3. Best of those → **per-channel gains** (z, xz only first)

One change per survey. Stop when overnight SeedSweep mean VPT×duty stops moving.

---

## Success criterion

Mean top-10% **VPT×duty** (and/or VPT, duty) improves vs current baseline on a
matched Unseen survey / short SeedSweep; **or** a clear negative for each arm
so the lever is closed. Do not over-index on single-orbit freeruns.
