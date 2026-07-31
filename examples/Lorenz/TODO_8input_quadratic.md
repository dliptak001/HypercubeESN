# TODO — 8-input quadratic drive (vs 4-in baseline)

**Context:** `num_inputs` must divide \(N = 2^{\mathrm{dim}}\) → legal counts are powers of 2. Next step after 4 is 8.  
**Date parked:** 2026-07-29  
**Status:** switchable support + campaign exist. **Decision path:** collapse
to baseline only (see next section) — multi-layout did not earn its keep.

---

## TODO — collapse DriveLayout to XyzXz only (remove the enum)

**Date added:** 2026-07-31  
**Goal:** hard-code 4-in `[x, y, z, x*z]` and **delete** the layout switch
machinery. The entire `DriveLayout` enum goes away:

```text
enum class DriveLayout
{
    XyzXz = 0,      // 4-in [x, y, z, x*z]  — keep as the only drive
    Quadratic8 = 1, // 8-in  — remove
    XyzXy = 2,      // 4-in [x, y, z, x*y] — remove
};
```

**Blast radius (when executing):**

- `Lorenz.h` / `Lorenz.cpp`: `config::DRIVE_LAYOUT`, `FillDrive`,
  `NumDriveChannels`, `DriveLayoutName`, CSV drive columns
- Campaigns: `ConfigDriveRestore`, `drive_layout` overrides on SeedSweep /
  FreeRun / FreeRunSurvey / ParallelSeedSweep; `Campaign_DriveLayoutAB`
- Weight stems / arch that encode `n_in` (already `_in{Nin}` — stays 4)
- README, this TODO, `TODO_drive_scale_sr.md`

**Do not do while multi-layout A/B is still in active use.** Park until
storefront is XyzXz-only in practice.

---

## Baseline (current)

```text
num_inputs = 4
drive = [ x, y, z, x*z ]
```

- `x*z` matches the bilinear term in \(\dot y\) (ODE-aligned).
- Free-run: rebuild product from predicted channels (`ExtractDrivePredicted`).
- Each input channel owns block size \(N/4\) (e.g. 512 at \(N=2048\)).

---

## Proposed arm: 8-in quadratic lift

```text
num_inputs = 8
drive = [ x, y, z, x*y, x*z, x*x, y*y, z*z ]
```

| slots | features | role |
|------:|----------|------|
| 1–3 | `x, y, z` | state |
| 4–5 | `xy, xz` | both Lorenz bilinears (\(\dot z\) has \(xy\), \(\dot y\) has \(xz\)) |
| 6–8 | `x², y², z²` | pure-quadratic pad to power-of-2 |

**Dropped intentionally:** `yz` (not in the ODE; full degree-2 monomials would be 9 channels).

All features free-run-safe (functions of the predicted 3-vector only).

---

## Implementation sketch (when picked up)

1. `MakeESNConfig`: `num_inputs = 8`.
2. `ExtractDriveReal` / `ExtractDrivePredicted`: fill 8 floats from state / prediction.
3. Trace CSV drive columns if still dumped — extend or document new layout.
4. Keep single global `INPUT_SCALING` for the first A/B (no per-channel gains yet).
5. Score with **mean-of-trial-means** survey (not one freerun): VPT, RMSE, duty vs 4-in `xz` baseline, matched seeds/orbits/M/protocol.

### Caveat

8-in shrinks each channel’s hypercube block (\(N/8\) vs \(N/4\)). More features, less spatial real estate per channel — not free. If flat/worse, try `INPUT_SCALING` / `M` before jumping to 16-in.

---

## Optional follow-ups (only if 8-in wins or is interesting)

- Ablation: `[x,y,z,xy,xz,yz,?,?]` vs pure squares (complete bilinear vs squares).
- 16-in only if 8-in clearly helps and block size still healthy.
- Do **not** over-index on single-run per-axis VPT when deciding.

---

## Success criterion

Mean-of-trial-means **vector** VPT (and/or duty) improves vs matched 4-in baseline on a SeedSurvey-scale run; or a clear negative result documented so we stop chasing products.
