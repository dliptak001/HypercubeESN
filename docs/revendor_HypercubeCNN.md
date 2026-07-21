# Re-vendor HypercubeCNN into HypercubeESN (best-use plan)

**Status:** Phases A–E implemented (2026-07-21). Keep this file as the integration record; use checkboxes/progress log for any follow-ups.  
**Audience:** maintainers integrating HypercubeCNN as the ESN readout.  
**Upstream:** [HypercubeCNN](https://github.com/dliptak001/HypercubeCNN) (Apache-2.0).  
**Related:** [adapt_HypercubeCNN.md](adapt_HypercubeCNN.md) (host checklist / API deltas),  
[Readout.md](Readout.md) (product story),  
[../third_party/HypercubeCNN/VENDORED.md](../third_party/HypercubeCNN/VENDORED.md) (pin).

**Goal:** HypercubeESN is the **canonical RC host** for HypercubeCNN v1.0.0 — full
facade use, not a minimal “compile-green” rename. Reservoir on the cube → HCNN
reads the same cube.

**Pin target:** HypercubeCNN **v1.0.0** (or later SHA on that release line; record
exact hash in `VENDORED.md`).

---

## North star

| Principle | Meaning for ESN |
|-----------|-----------------|
| **One product story** | Reservoir on the cube → HCNN reads the same cube. |
| **Facade-native** | Build / train / infer / checkpoint through public HCNN products (`HCNN`, arch, train helpers, inputs). |
| **PIMPL at the boundary** | Public ESN headers stay free of `HCNN.h`; mapping lives in `Readout.cpp`. |
| **RC-correct, not vision-correct** | Full-N state, no spatial pack/aug on the readout path. Spatial stays vendored for library parity only. |
| **Interop-ready** | Arch + HCNW beside (not instead of) ESN `vector<double>` blobs. |

---

## Snapshot delta (why re-vendor)

| | ESN pin (pre-work) | Upstream v1.0.0 |
|--|--------------------|-----------------|
| Identity | `v0.1.0-21-g1b807c1` | **v1.0.0** (+ minor post-tag commits) |
| Public front door | `HCNN.h` re-exports Network + `LossType` | `HCNN.h` + types/input; Network **private** |
| Ctor | `…, TaskType, LossType, num_threads` | `…, TaskType, num_threads` (loss fixed by task) |
| Train API | `Train*` + `Train*Regression` | Unified `Train*` overload by target type |
| Arch product | Hand `AddConv`/`AddPool` only | `LayerSpec` / `HCNNConfig::Build` |
| Train session | Positional knobs | `TrainParams`, `HCNNTrainer`, `cosine_lr` |
| Infer | Embed + Forward | + `Predict` / `PredictClass` |
| Core TUs | 5 `.cpp` | 8 `.cpp` (+ spatial + train helpers) |

**Note:** `Train*Regression` is **removed** in v1.0.0 (not kept as aliases). Use
`float*` overloads of `TrainStep` / `TrainBatch` / `TrainEpoch`.

---

## Coupling map (blast radius)

```text
third_party/HypercubeCNN  ──link──▶  HypercubeESNCore
                                         │
                                         ├── Readout.cpp   ◄── primary HCNN host
                                         ├── ESN.cpp       (via Readout API)
                                         ├── examples/*    (via ESN / ReadoutConfig)
                                         └── python/bindings.cpp (via ESN)
```

| Layer | Expected impact |
|-------|-----------------|
| `third_party/HypercubeCNN/*` | Full replace |
| Vendor `CMakeLists.txt` | ESN glue; list all core sources |
| `Readout.cpp` / `Readout.h` | Major impl rewrite; config expansion |
| `ESN.*` | Light (load modes, optional model paths) |
| Examples | Rebuild + optional new knobs |
| Python | Rebuild; bind new config fields |
| Docs | This plan, adapt, Readout, CPP_SDK |

---

## Capability → ESN adoption

| CNN v1.0.0 capability | Role in ESN |
|----------------------|-------------|
| Ctor without `LossType`; unified `Train*` | Required |
| `LayerSpec` / `apply_arch` / `HCNNConfig::Build` | **Primary** architecture path |
| `TrainParams` + `HCNNTrainer` | Batch `Train()` epoch hygiene |
| `Predict` / `PredictClass` | Default infer path |
| `HCNNInputView` / full-capacity contract | All train/infer sites (states are length N) |
| `evaluate_regression` / `evaluate_classification` | MSE/CE/accuracy helpers; keep multi-output R² semantics |
| `HCNNBestMetricCheckpoint` | Optional best-metric restore inside batch Train |
| `GetWeights`/`SetWeights` + `reset_optimizer_moments` | Eval vs resume-train load modes |
| `save_weights` / `load_weights` (HCNW) | Optional portable model files |
| Arch sidecar (`LayerSpec`) | Persist topology with weights |
| BN / pool type / optimizer knobs | First-class `ReadoutConfig` where useful |
| Spatial embed/aug | **Do not** wire into reservoir readout |

---

## Target architecture (Readout as host)

```text
ReadoutConfig  ──map──▶  hcnn::HCNNConfig + LayerSpec[]
                              │
                              ▼
                        HCNNConfig::Build()   →  unique_ptr<HCNN>
                              │
                    PrepareBuffers / TrainDefaults
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   HCNNTrainer          Predict /             GetWeights /
   (batch epochs)       PredictClass          SetWeights / HCNW
         │                    │                    │
         ▼                    ▼                    ▼
   optional best-metric   R2 / Accuracy      ESN State + optional
   checkpoint             (multi-out R2)     .hcnw + arch sidecar
```

Public RC workflow (`Train`, `TrainStep*`, `Predict*`, `Weights`/`SetState`) stays
familiar; the **engine** is a proper HCNN host.

---

## Phases

### Phase A — Vendor full core (foundation)

- [x] Pin upstream tag/SHA; update `VENDORED.md`
- [x] Copy full `HypercubeCNNCore` sources + public headers into `third_party/HypercubeCNN/`
- [x] Update ESN glue `CMakeLists.txt` (all core `.cpp`; keep portable flags; `HCNN_FAST_TANH`)
- [x] Do **not** replace glue with upstream’s full project CMake
- [x] Do **not** hand-edit vendored sources

**Files to vendor (core):**  
`HCNN.*`, `HCNNTypes.h`, `HCNNInput.h`, `HCNNArch.h`, `HypercubeCNN.h`,  
`HCNNConv.*`, `HCNNPool.*`, `HCNNNetwork.*`, `HCNNReadout.*`, `ThreadPool.h`,  
`HCNNSpatialAug.*`, `HCNNSpatialEmbed.*`, `HCNNTrainHelpers.*`, `LICENSE`

**Do not vendor:** `dataloader/`, `examples/`, `python/`, `tests/`, MNIST data.

### Phase B — Facade-native build & train (core quality leap)

- [x] Map `ReadoutConfig` → `LayerSpec[]` + `HCNNConfig`; use `Build()`
- [x] Expand config knobs: `use_batchnorm`, pool type, optimizer, channel growth (defaults preserve old behavior)
- [x] Batch `Train` via `HCNNTrainer` + `TrainParams` + HCNN `cosine_lr`
- [x] Online `TrainStep` / `TrainStepBatch` via unified train + `TrainParams`
- [x] Infer via `Predict` / `PredictClass` (drop Embed+Forward scratch when unused)
- [x] Full-capacity `HCNNInputView` at call sites
- [x] Metrics: keep multi-output `R2` semantics; batch `ForwardBatch` for R2/Accuracy
- [x] Drop `LossType`; drop all `*Regression` train names
- [x] `ReadoutLoadMode` on `SetState` (Eval vs ResumeTrain)

**Defaults after B:** arch via `HCNNConfig`, `Predict`, Trainer path, capacity discipline.  
**Default off (API present later in C/D):** BN, best-epoch restore, HCNW.

**Cosine schedule note:** HCNN `cosine_lr` uses `epoch / max(num_epochs-1, 1)` so the
last epoch hits `lr_min`. Old ESN `CosineLR(e/horizon)` differs slightly — retune if
surveys shift; keep `CosineLR` / `ExponentialDecayLR` in `Readout.h` for callers that
drive online LR themselves.

### Phase C — Training quality

- [x] Optional best-metric epoch restore (`HCNNBestMetricCheckpoint` / dual acc); config-gated, default off
- [x] Load modes: eval vs resume-train on `Readout::SetState` and `ESN::SetReadoutState`
- [x] Batch eval via `ForwardBatch` for R²/Accuracy on large sets (done in Phase B)
- [x] Document multi-ESN `num_threads = 1` policy

### Phase D — Model I/O & interop

- [x] Keep `Weights()` / `SetState` `vector<double>` for ESN checkpoints (unversioned, unchanged)
- [x] HCNW + arch export/import: `Readout::SaveHcnnModel` / `LoadHcnnModel`, ESN wrappers
- [x] Arch sidecar `format=hypercube_esn_readout_arch` version 1 (`.arch.json`); HCNW via `hcnn::save_weights`

### Phase E — Surface, examples, Python, docs

- [x] Example(s) print arch summary / param count (`BasicPrediction`)
- [x] Python bindings: HCNW I/O, arch summary, best-epoch knobs / accessor
- [x] Refresh `Readout.md`, `adapt_HypercubeCNN.md`, CPP_SDK snippets (A–D)
- [x] Smoke: HCNW round-trip; Release link of core + key examples

---

## Verification matrix

| Path | Check |
|------|--------|
| Regression batch `Train` | BasicPrediction / NARMA-scale: finite preds, R² moves sensibly |
| Classification batch | SignalClassification accuracy path |
| Online `TrainStep` / `TrainStepBatch` | Streaming path no crash; lr/wd forwarded |
| Infer | Predict / free-run short smoke |
| Checkpoint | `Weights` → `SetState` size match + eval; HCNW round-trip (`SaveHcnnModel` / `LoadHcnnModel`) |
| Threads | Multi-ESN with HCNN `num_threads=1` |
| Python | Optional pybuild + `test_basic` |

Expect **non-bit-identical** metrics vs the old pin (new core, cosine definition,
optional best-epoch later).

---

## Risk register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Compile break (ctor, `*Regression`) | High until patched | Same PR as vendor + Readout |
| Cosine schedule drift | Medium | Document; retune `lr_*` if needed |
| Best-epoch default flip | Medium–High | Gate off until validated |
| BN blob size | High if enabled casually | Default off; document |
| Multi-output R² vs global helper | Medium | Keep multi-output R² API semantics |
| Hand-edit vendor tree | High long-term | Re-vendor only |
| Scope creep | Process | Ship A+B first, then C/D |

---

## Explicitly out of scope (readout path)

- Including private `HCNNNetwork` / layers in public ESN headers  
- Spatial embed/aug for reservoir state  
- Freezing HCNN around ESN’s pre-v1 train API  
- Replacing ESN’s public RC workflow with raw `HCNN` in every example  

---

## Delivery order

1. **A + B (first PR / this work):** full vendor + facade build/train/infer.  
2. **C:** best checkpoint, load modes, batch eval.  
3. **D:** HCNW + arch interop.  
4. **E:** docs/Python continuous; finalize after B green.

---

## Progress log

| Date | Note |
|------|------|
| 2026-07-21 | Plan written; Phase A+B implementation started. |
| 2026-07-21 | Phase A complete: pin `v1.0.0-2-g8859550` (`8859550`), full core vendored. |
| 2026-07-21 | Phase B complete: facade-native Readout; Release `HypercubeESNCore` links clean. |
| 2026-07-21 | Phase C: `restore_best_epoch` + holdout frac; ESN load mode; threading docs. |
| 2026-07-21 | Phase D: HCNW + `.arch.json` Save/Load; ArchSummary; round-trip smoke OK. |
| 2026-07-21 | Phase E: BasicPrediction arch print; Python HCNW/best-epoch bindings. |
