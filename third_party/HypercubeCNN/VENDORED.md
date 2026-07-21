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

Host integration plan: docs/revendor_HypercubeCNN.md
