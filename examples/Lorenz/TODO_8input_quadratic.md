# TODO — multi-layout / 8-input quadratic drive — **DONE / COLLAPSED**

**Date closed:** 2026-07-31  

Storefront is fixed **4-in `[x, y, z, x*z]`** only (`kNumDriveChannels = 4`).
`DriveLayout` enum, `XyzXy`, `Quadratic8`, `Campaign_DriveLayoutAB`, and
`drive_layout` campaign overrides have been removed.

Historical notes below kept for archaeology only.

---

## Original context (parked 2026-07-29)

`num_inputs` must divide `N = 2^dim` → legal counts are powers of 2. Next step
after 4 was 8. Switchable support + A/B campaign existed; multi-layout did not
earn its keep → collapse to baseline.
