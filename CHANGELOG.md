# Changelog

All notable changes to HypercubeESN are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)-style sections.
Versioning: public releases use [SemVer](https://semver.org/). **2.0.0** is the
first public major aligned across CMake, the C++ package, and the PyPI wheel.

---

## [2.0.0] — TBD (public release)

**Status:** version numbers bumped to 2.0.0 in-tree; **tag / PyPI / GitHub Release
pending** after Lorenz half-anchored free-run storefront is filled.

**Upgrading from 1.x?** Jump to [Migration notes](#migration-notes-1x-to-20).

### Breaking

- **Version alignment.** Project, package, and tags are **2.0.0** (was CMake
  0.2.x / Python 1.x). Pin `GIT_TAG v2.0.0` (or later) for FetchContent.
- **`verbose` default is `false`** (`ReservoirConfig` and Python `ESN`). Demos
  that want the construction banner must set `verbose=true`.
- **Full-state linear feedback (FSF) removed** from the reservoir (if you used
  internal FSF knobs, that path no longer exists). Prefer caller-owned
  **external feedback** (`num_external_feedback_channels` + `ReservoirStep(u, fb)`).
- **Experimental knobs removed** from the public path (history_floor depth taper,
  Lorentzian / A(x) activation, aux side channels). A(x) findings are archived in
  `docs/ActivationFunctionA.md`.
- **Python pickle persistence v8.** New fields:
  `num_external_feedback_channels`, `external_feedback_scaling`, `bias_scaling`,
  `readout_slices`. Older pickles still load; missing keys use defaults.
  Pickles newer than v8 refuse to load on older installs.

### Highlights

- **HypercubeCNN readout: self tap (K = dim + 1).** Vendored HCNN conv kernels
  moved from **nearest neighbors only** (kernel width **dim** — one weight per
  Hamming-1 edge) to **nearest neighbors plus the current vertex** (kernel width
  **K = dim + 1**). Each site now has an explicit **self/center** contribution
  alongside its dim bit-flip neighbors. That is a material upgrade to the only
  trained stage in HypercubeESN: readout quality improved **across the board**
  (tasks and dims), not a one-benchmark tweak. Pin: `third_party/HypercubeCNN`
  (see `VENDORED.md`). Details: [docs/Readout.md](docs/Readout.md).

### Added

- **Python closed-loop surface:** `num_external_feedback_channels`,
  `external_feedback_scaling`, `reservoir_step(inputs, external_feedback=None)`.
- **Python multi-slice readout:** `readout_slices` (B, power of two ≤ M).
- **`dim=` alias** for `reservoir_hypercube_dimension` (Python); `ESN::Dim()` (C++).
- **C++ `std::span` overloads** for drive / train / copy / predict / score, with
  length checks.
- **`R2FromWindow` / `NRMSEFromWindow` / `AccuracyFromWindow`** when targets are a
  scored slice only (not a full `[0, start+count)` buffer).
- **`TargetSpectralRadius()` / `RealizedSpectralRadius()`** on `ESN`.
- **`PredictFromReadoutInput`** (clearer name for B·N HCNN input; historical
  `PredictFromState` kept).
- **Lorenz harness:** Janus half-anchored free-run, `FORWARD_ONLY` ablation,
  GS operational metrics (`duty`, `n_relock`, `n_unlock`, `mean_locked_sojourn`).
- **House style in docs:** hypercube dimension **dim**; powers as
  `2<sup>dim</sup>` in prose.

### Changed

- **HypercubeCNN kernel geometry** — see **Highlights** (dim → dim+1 with self
  tap). Retrain readouts that were frozen against older neighbor-only HCNN
  weights; blobs are not layout-compatible in spirit even when loaders succeed.
- **`num_layers` default remains 1** (C++ and Python). `0` = auto
  `min(dim-2, 2)`.
- SDK / README install one-liners: `pip install hypercube-esn` and C++ clone +
  `BasicPrediction` / FetchContent.
- Documentation pass: Reservoir, Readout, CPP_SDK, Python_SDK, feedback mechanism,
  NARMA / MC storefronts.

### Fixed

- Doc bugs: `history_depth` is recurrent **M**, not readout **B**; scoring
  buffers must cover `[0, start+count)` unless using `*FromWindow`.
- Power-of carets in markdown; Related Work (Katori) wording.

### Validators (storefront)

| Validator | Status |
|-----------|--------|
| NARMA N30/50/70 best-5 NRMSE | **0.0441 / 0.0751 / 0.1251** (frozen) |
| Memory capacity peaks (dim 5…12) | **~30 → ~1400+** (frozen; see MemoryCapacity.md) |
| Lorenz half-anchored free-run | **TBD** — fill before public tag |

### Migration notes (1.x → 2.0) <a id="migration-notes-1x-to-20"></a>

1. Rebuild / reinstall wheels; do not mix 1.x extension modules with 2.0 headers.
2. **Retrain HCNN readouts** after the dim → dim+1 self-tap kernel change (or
   re-export from a 2.0 train); do not expect old neighbor-only weights to
   match new kernels.
3. Set `verbose=true` if you relied on construction banners.
4. Replace any FSF usage with external-feedback ports + host policy.
5. Re-export or re-train pickles if you need new closed-loop / B fields explicitly.
6. Prefer `dim=` / document `readout_input_width` when B > 1.

---

## [1.4.0] and earlier

Pre-2.0 public Python wheels and intermediate private branch work. See git history
on `main` / release tags `v1.*` for detail. Not exhaustively backfilled here.
