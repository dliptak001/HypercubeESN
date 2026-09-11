Vendored snapshot of dliptak001/HypercubeCNN @ **v1.0.4**
(commit `ef1b710e06c13d65a6b6316f6f48a8f61b6ed63d`);
canonical upstream — do not edit here, re-vendor instead.

Upstream: https://github.com/dliptak001/HypercubeCNN  (Apache-2.0)

Vendored contents (HypercubeCNNCore surface):
- Core: HCNN.{h,cpp}, HCNNTypes.h, HCNNInput.h, HCNNArch.h, HypercubeCNN.h
- Private impl (build-only): HCNNConv.*, HCNNPool.*, HCNNNetwork.*, HCNNReadout.*,
  ThreadPool.h
- Optional public products: HCNNSpatialAug.*, HCNNSpatialEmbed.*, HCNNTrainHelpers.*
- LICENSE (Apache-2.0, retained from upstream)

CMakeLists.txt in this directory is HypercubeESN build glue (ours), NOT part of the
upstream snapshot. Updates flow one way: re-vendor by recopying the core files +
LICENSE from upstream at a chosen tag/commit; never hand-edit the snapshot here.

Readout host usage (facade-native train/infer): see docs/Readout.md and docs/CPP_SDK.md.

**Notable geometry (this pin):** conv kernels use **K = dim + 1** — dim Hamming-1
neighbors **plus a self/center tap** at each vertex. Earlier neighbor-only (K = dim)
kernels are obsolete for this host; retrain after re-vendoring. Called out in
[CHANGELOG.md](../../CHANGELOG.md) for HypercubeESN 2.0.0.

**Weight-init seed (this pin):** `HCNNConfig::weight_seed` and
`HCNN::RandomizeWeights` take full **`uint64_t`**. Seeds with high half zero keep
the historical `mt19937(seed32)` path; wider seeds expand both halves via
`seed_seq`. Shipped upstream in HypercubeCNN **v1.0.1**; this tree is a clean
re-vendor of **v1.0.4** (no local HCNN fork).

### Re-vendor checklist

1. Check out (or archive) the chosen HypercubeCNN tag on the upstream repo.
2. Copy the vendored content files listed above into this directory (overwrite).
3. Leave this directory’s `CMakeLists.txt` alone (ESN host glue).
4. Update the pin line at the top of this file to the new tag/commit.
5. Build `HypercubeESNCore` (and Python wheels if needed) before committing.
