# TODO — drive scale / spectral radius / per-channel gains

**Context:** DIM12 + M10/M12 already look respectable on Unseen free-run
(top-10% VPT×duty). Architecture-side bets (external feedback, richer drive
layouts) did not move us the right way. Remaining cheap levers are **dynamics
knobs** on the existing 4-in `[x, y, z, x*z]` stack — not new ports or layers.

**Date parked:** 2026-07-30  
**Status:** (4) gains **locked** as `constexpr INPUT_SCALE_CH = {1,1,0.9,0.7}`;
`Campaign_DriveGainAB` / `drive_gains` / `HistoryDepthSweep` / `SpectralRadiusAB`
**removed**. (3) SR: flip `config::SPECTRAL_RADIUS` or campaign SR override (no A/B helper).
(1) still config-only A/B for global scale.

**Out of scope (already known bad or not believed useful here):**

| Idea | Why parked |
|------|------------|
| `LEAK_RATE` < 1 | Always detrimental in this storefront |
| Train-time drive noise | Always detrimental |
| Scheduled sampling / “anti-overfit free-run mix” | Random multi-orbit ICs already mitigate open-loop overfit; not the failure mode we see |
| Multi-layout drive enum (XyzXy / Quadratic8) | **Removed** — fixed 4-in `[x,y,z,xz]` only (`kNumDriveChannels`) |

**Protocol for every arm:** one change at a time; fixed DIM/M/epochs/seed set;
Unseen FreeRunSurvey (or short SeedSweep); report top-10% freerun means
(VPT, duty, VPT×duty, RMSE). Prefer matched seeds/orbits across arms.

---

## Baseline (current)

```text
drive (fixed)        →  [x, y, z, x*z]
INPUT_SCALING        = 0.015  (global; check Lorenz.h)
SPECTRAL_RADIUS      = 0.999
LEAK_RATE            = 1.0    (leave alone)
INPUT_SCALE_CH       = {1, 1, 0.9, 0.7}  (constexpr soft z/xz)
```

Free-run rebuilds products from predicted `(x,y,z)`.

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

## 3. Spectral radius a notch off the edge  (config / override only)

```text
SPECTRAL_RADIUS = 0.999   // house default; reassignable per campaign arg
```

Near-edge SR can amplify prediction error in closed loop. Mild contraction is
cheap to test and often flat — still worth one matched pass **after** (1).

| Try |
|----:|
| 0.95, 0.97, 0.98, **0.999** |

**How:** flip `config::SPECTRAL_RADIUS` in `Lorenz.h`, or pass `spectral_radius`
on SeedSweep / Parallel* / FreeRun* (`>0` overrides for that run). No dedicated
A/B campaign (`Campaign_SpectralRadiusAB` removed).

**Done when:** clear win, or “flat / baseline is fine” written down so we stop
retuning SR.

---

## 4. Per-channel input gains  (LOCKED)

**Status:** **Locked.** `constexpr config::INPUT_SCALE_CH = {1.f, 1.f, 0.9f, 0.7f}`
for `[x, y, z, x*z]`. Soft z/xz from the tuning grid below. `Campaign_DriveGainAB`
and per-campaign `drive_gains` parameters **removed** — edit `Lorenz.h` and rebuild
if you ever revisit.

**Wiring:**

```text
FillDrive → features → drive[i] *= INPUT_SCALE_CH[i]
ReservoirStep then multiplies by global INPUT_SCALING (and fan-in)
```

- Train + free-run both go through `FillDrive`
- Banners / campaign metadata print `drive_ch=[...]`
- Weights stem does not encode gains — train and load-time gains must match

Historical first grid (already folded into the lock):

| Channel | Index | Locked |
|---------|------:|--------|
| x, y | 0, 1 | 1.0 |
| z | 2 | 0.9 |
| x*z | 3 | 0.7 |

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
