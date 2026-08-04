Vendored snapshot of dliptak001/HypercubeCNN @ v1.0.0-2-g8859550
(commit 885955076fb46ab274fe1941e2ad7bfd698748b9);
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
[CHANGELOG.md](../../CHANGELOG.md) for HypercubeESN 2.0.

---

## In-tree drift (must land upstream before HypercubeESN v2.0.0 deploy)

**Policy reminder:** this tree is supposed to be a read-only re-vendor of
[dliptak001/HypercubeCNN](https://github.com/dliptak001/HypercubeCNN). HypercubeESN
development has temporarily edited files **in this directory** (seed-width and any
other HCNN fixes). Those edits do **not** automatically flow to the standalone
GitHub project.

### Release gate (HypercubeESN public **v2.0.0**)

When the HCNN changes have been **vetted** in HypercubeESN and we are **ready to
tag / deploy HypercubeESN v2.0.0**:

1. **Migrate** the vetted HCNN diffs from this vendored tree into the
   **GitHub HypercubeCNN** repo (PR + review on upstream).
2. **Tag** a new HypercubeCNN release (e.g. after v1.0.0) that includes at least:
   - `uint64_t` weight-init seed end-to-end (`HCNNConfig::weight_seed`,
     `HCNN::RandomizeWeights`, `HCNNNetwork::randomize_all_weights`)
   - low-32 historical path when high half is zero; full `seed_seq` expansion
     for wider seeds
   - any other HCNN fixes that shipped with HypercubeESN 2.0 (self-tap K=dim+1
     if not already on the upstream pin, etc.)
3. **Re-vendor** this directory from that upstream tag/commit; update the pin
   line at the top of this file and drop or rewrite this “in-tree drift” section
   so HypercubeESN again matches canonical HypercubeCNN.
4. Only then **publish** HypercubeESN v2.0.0 (tag / PyPI / GitHub Release) with
   a clean pin — no permanent fork of HCNN living only under
   `third_party/HypercubeCNN`.

Do **not** treat “ESN works on feedback” as done for HCNN; **upstream
HypercubeCNN must receive the same API/behavior** so other consumers and future
re-vendors stay aligned.
