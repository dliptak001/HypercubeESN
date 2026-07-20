# Adapting HypercubeESN to current HypercubeCNN

**Audience:** maintainers re-vendoring HypercubeCNN and updating `Readout`.  
**Status:** guidance only — HypercubeESN may still pin an older HCNN snapshot.  
**Upstream:** [HypercubeCNN](https://github.com/dliptak001/HypercubeCNN) (Apache-2.0).  
**Do not edit** `third_party/HypercubeCNN` by hand; re-vendor from upstream (see [VENDORED.md](../third_party/HypercubeCNN/VENDORED.md)).

This note captures **why** HypercubeCNN’s public surface moved, **what** must change in HypercubeESN, and **how** to integrate cleanly as a long-term readout host. It is written from HypercubeESN’s current `Readout` usage and HypercubeCNN’s post–0.2 facade work (train unify, session defaults, input types, private Network, etc.).

---

## 1. Why re-vendor and adapt

| Goal | Rationale |
|------|-----------|
| **One readout product** | HypercubeCNN is the shared learned head for HypercubeESN (and teaching demos). Keeping a frozen fork accumulates silent drift (ctor shape, train names, weight blob layout). |
| **ESN can adapt** | HypercubeESN is still evolving; it should follow a **clean HCNN facade**, not freeze HCNN around a temporary vendor pin. |
| **Smaller public HCNN** | Apps (including ESN) use only `hcnn::HCNN` (+ optional helpers). Network/layers are private and not installed. |
| **Safer train API** | One vocabulary (`TrainStep` / `TrainBatch` / `TrainEpoch`) with overload by target type; wrong `TaskType` still throws. |

HypercubeRC is deprecated; treat HypercubeESN as the **canonical host** when choosing HCNN interop rules.

---

## 2. Current HypercubeESN ↔ HCNN coupling (as of old vendor)

All HCNN use is confined to **`Readout.cpp`** (PIMPL). Public ESN headers only forward-declare `hcnn::HCNN`.

| Concern | What Readout does today (old snapshot) |
|---------|----------------------------------------|
| Construct | `HCNN(dim, num_outputs, 1, task, LossType::Default, num_threads)` |
| Arch | `AddConv` / optional `AddPool(MAX)` / `RandomizeWeights` / `SetOptimizer(ADAM)` / `PrepareBuffers` |
| Batch train | `TrainEpoch` vs `TrainEpochRegression` by `ReadoutTask` |
| Online train | `TrainStep` / `TrainBatch` vs `*Regression` twins |
| Infer | Manual `Embed` + `Forward` (scratch buffers); local argmax for class |
| Weights | `GetWeights` → `vector<double>`; `SetWeights` from double blob |
| Inputs | Always **full** length `N = 2^dim` (reservoir state) |
| Schedule | **ESN-owned** `CosineLR` / `ExponentialDecayLR` — not HCNN helpers |
| Spatial / MNIST path | **Unused** (correct for ESN) |

Nothing in ESN should include `HCNNNetwork`, `HCNNConv`, etc.

---

## 3. Breaking / behavioral changes to absorb on re-vendor

### 3.1 Constructor (required)

**Old (vendored):**
```cpp
net_ = std::make_unique<hcnn::HCNN>(
    d, config_.num_outputs, /*input_channels=*/1,
    task_type, hcnn::LossType::Default, config_.num_threads);
```

**New:**
```cpp
net_ = std::make_unique<hcnn::HCNN>(
    d, config_.num_outputs, /*input_channels=*/1,
    task_type, config_.num_threads);
// Loss is fixed by TaskType: Classification → CE, Regression → MSE.
// Default optimizer is Adam (SetOptimizer still fine and explicit).
```

Remove any use of `hcnn::LossType`.

### 3.2 Train API (required for “best” facade)

**Old:** dual names  
`TrainEpoch` / `TrainEpochRegression`, same for Step/Batch.

**New (preferred):** one name, overload by target type:

```cpp
// Classification
net_->TrainEpoch(states, n, int_labels, num_samples, batch,
                 lr, momentum, weight_decay, /*class_weights=*/nullptr, shuffle_seed);

// Regression
net_->TrainEpoch(states, n, float_targets, num_samples, batch,
                 lr, momentum, weight_decay, shuffle_seed);
```

Same pattern for `TrainStep` / `TrainBatch`.

**Transition:** `Train*Regression(...)` still exist as **thin aliases** to the float* overloads. ESN may call either after re-vendor; prefer the unified names in new code.

Dispatch in `Readout` can stay:

```text
if (classification)  Train*(..., int labels, ...);
else                 Train*(..., float targets, ...);
```

without a separate `*Regression` suffix.

### 3.3 Weights (verify)

`GetWeights` / `SetWeights` layout now includes BN γ/β + running stats when layers use BN.  
Current Readout stacks typically **do not** enable BN — blob shape still “kernel + bias + readout” for that arch. After re-vendor:

- Keep round-trip tests (checkpoint save/load).  
- If you ever enable BN on the readout, update checkpoint docs and any size assumptions.  
- Optimizer moments are **still not** in the blob; `SetWeights(blob)` default is eval-style (moments not reset unless you pass `reset_optimizer_moments=true`).

### 3.4 Optional conveniences (not required for ESN)

| HCNN API | ESN recommendation |
|----------|-------------------|
| `Predict` / `PredictClass` | Optional replace of Embed+Forward scratch; same math |
| `TrainParams` / `SetTrainDefaults` | Optional; ESN already passes lr each call |
| `HCNNTrainer` + `cosine_lr` | Optional; ESN already owns schedules in `ReadoutConfig` |
| `HCNNInputView` / `HCNNInputBatch` | Optional; ESN states are already full length N |
| Spatial embed / aug | **Do not use** for reservoir readout |
| `HCNNNetwork` / layers | **Never include** in ESN sources |

---

## 4. Re-vendor checklist (mechanical)

1. **Pick an upstream commit/tag** on HypercubeCNN you trust (prefer a tagged release once one exists that includes the facade work below).  
2. **Copy core library sources** into `third_party/HypercubeCNN/` per `VENDORED.md` (headers + `.cpp` for the static core, `LICENSE`).  
   - Public install surface upstream is larger now (`HCNNTypes`, `HCNNInput`, `HCNNArch`, helpers, spatial).  
   - For ESN you may still vendor **only what `HypercubeCNNCore` needs to build**, **or** vendor the full core set matching upstream `CMakeLists.txt` for the library target — keep ESN’s `third_party/HypercubeCNN/CMakeLists.txt` as **ESN glue**, not upstream’s full project.  
3. **Update `VENDORED.md`** with commit hash / tag and file list.  
4. **Rebuild** `HypercubeESNCore` and fix compile errors (ctor, includes).  
5. **Patch `Readout.cpp`** (section 5).  
6. **Run ESN examples / tests** that hit Train, TrainStep, TrainStepBatch, Predict, Weights/SetState (regression + classification).  
7. **Do not** edit vendored upstream sources for ESN-specific fixes — fix ESN or re-vendor.

### Minimum compile fixes after a modern re-vendor

| Location | Change |
|----------|--------|
| `Readout.cpp` ctor args | Drop `LossType::Default` |
| Includes | Still only `#include "HCNN.h"` in `Readout.cpp` (HCNN.h pulls types) |
| Train dispatch | Prefer unified `Train*` + float*/int* overload (section 3.2) |
| Smoke | Any test that assumed old ctor |

---

## 5. Suggested `Readout.cpp` train dispatch (target shape)

Conceptual (illustrative — adapt to your coding style):

```cpp
// Batch Train() loop body, after lr is computed:
if (is_classification) {
    net_->TrainEpoch(
        states, static_cast<int>(n),
        int_targets.data(),
        static_cast<int>(num_samples), config_.batch_size,
        lr, config_.momentum, config_.weight_decay,
        /*class_weights=*/nullptr,
        /*shuffle_seed=*/static_cast<unsigned>(e + 1));
} else {
    net_->TrainEpoch(
        states, static_cast<int>(n),
        targets,   // float*, num_samples * num_outputs
        static_cast<int>(num_samples), config_.batch_size,
        lr, config_.momentum, config_.weight_decay,
        /*shuffle_seed=*/static_cast<unsigned>(e + 1));
}
```

Online:

```cpp
if (classification)
    net_->TrainStep(state, n, static_cast<int>(target[0]), lr, mom, wd);
else
    net_->TrainStep(state, n, target, lr, mom, wd);
```

Keep ESN’s public `Readout::Train` / `TrainStep` / `TrainStepBatch` **float* targets** API unchanged if you want host stability; only the HCNN call sites simplify.

---

## 6. Best practices for HypercubeCNN as ESN readout

### 6.1 Topology and sizing

- Set `ReadoutConfig::dim` so **`2^dim == ReservoirNeuronCount()`** (one float per vertex, `input_channels = 1`).  
- Pass **`input_length = N`** always (full state). Do not pass short lengths and rely on HCNN zero-pad unless you intentionally want zeros in the unused tail.  
- Prefer **`use_pooling`** decisions for *topology*: pooling mixes antipodal pairs and drops DIM; conv-only keeps full vertex resolution into the FLATTEN head (more features, more params).

### 6.2 Tasks and loss

- Map `ReadoutTask` → `hcnn::TaskType` at construction only.  
- Loss is **fixed by task** (CE vs MSE). Do not reintroduce a parallel loss enum on the ESN side.  
- Classification labels for HCNN must be **`int`** (or `const int*`); convert from ESN’s float class-index buffers at the boundary (as today).

### 6.3 Training

- Pass **learning rate every step/epoch** (or adopt `TrainParams` later if you want).  
- **Adam** is HCNN’s default; explicit `SetOptimizer(ADAM)` remains good documentation.  
- `momentum` on train calls is for **SGD**; under Adam it is ignored — keep config fields if you might switch optimizers.  
- Use **`num_threads = 1`** on HCNN when the host parallelizes across many ESN instances (avoid nested pools).  
- Own the **LR schedule** in ESN (`CosineLR` / `ExponentialDecayLR`) unless you deliberately migrate to `hcnn::cosine_lr` / `HCNNTrainer` for shared behavior with HCNN demos.

### 6.4 Inference

- `Embed` + `Forward` is valid; `Predict` is a convenience that embeds into internal scratch.  
- Classification: logits from Forward/Predict, then argmax (or `PredictClass` if you adopt it).  
- Softmax is **not** applied in `Forward` — correct for CE training and for exporting logits.

### 6.5 Weights and checkpoints

- Prefer **`GetWeights` / `SetWeights`** for the HCNN blob (float).  
- ESN may keep double blobs for its own checkpoint format; convert at the edge.  
- After load, decide: eval only (default moments) vs continue training (`SetWeights(blob, true)` or re-`SetOptimizer`).  
- Architecture (dim, layers, channels, task, outputs) must match the blob; HCNN validates size on `SetWeights`.

### 6.6 What not to couple to

| Avoid in ESN core | Why |
|-------------------|-----|
| Spatial aug/embed | Image packing, dual pad contracts — irrelevant to reservoir state |
| Including Network/layers | Private implementation; breaks on install-only consumers |
| Assuming SGD default | Upstream default is Adam |
| Hand-editing `third_party/HypercubeCNN` | Diverges from upstream; re-vendor only |

### 6.7 PIMPL discipline (keep)

- `Readout.h` must not include `HCNN.h` (forward declare only).  
- Map `ReadoutActivation` → `hcnn::Activation` only in `Readout.cpp`.  
- This isolates ESN’s public API from HCNN header churn.

---

## 7. Optional later upgrades (after a successful re-vendor)

1. **`Predict` / `PredictClass`** — drop scratch Embed+Forward if you want less boilerplate.  
2. **`TrainParams`** — single bag for lr/momentum/wd/shuffle if train call sites grow.  
3. **Weight file helpers** (`save_weights` / `load_weights`) — only if you want HCNN-native files alongside ESN checkpoints.  
4. **Pin a tagged HypercubeCNN release** in VENDORED.md once upstream tags the facade series.  
5. **Python bindings** — if they wrap Readout only, HCNN changes stay internal to `Readout.cpp`.

---

## 8. Verification matrix (ESN side)

After re-vendor + Readout edits, smoke at least:

| Path | Checks |
|------|--------|
| Regression batch `Train` | Finite preds, R² moves in the right direction on a toy sine / NARMA-scale run |
| Classification batch `Train` | Accuracy path (e.g. SignalClassification-style) |
| `TrainStep` / `TrainStepBatch` | Online path without crash; lr/wd forwarded |
| `Predict` / `PredictFromState` (ESN) | Match pre-re-vendor behavior on fixed seed if you have one |
| `Weights` / `SetState` | Round-trip blob size and eval after load |
| Multi-ESN / threads | `num_threads=1` still used where surveys parallelize |

---

## 9. Summary

| Do | Don’t |
|----|--------|
| Re-vendor from HypercubeCNN; update VENDORED.md | Hand-patch vendored sources |
| Drop `LossType` from ctor | Keep dual loss enums |
| Prefer unified `Train*` + target-type overload | Invent a third train naming scheme in ESN |
| Keep full-N state buffers | Short `input_length` with non-zero “pad” semantics |
| Keep HCNN behind Readout PIMPL | Leak HCNN types into public ESN headers |
| Keep ESN-owned LR schedules unless you choose to share | Assume HCNN demos’ MNIST recipe applies to reservoirs |

**Bottom line:** HypercubeCNN is the readout engine; HypercubeESN is the host. Adapt ESN’s `Readout.cpp` to the **current HCNN facade** (ctor, unified train names, private internals) on re-vendor so the pair stays one architecture — not two forked APIs.

---

## 10. Related docs

| Doc | Role |
|-----|------|
| [Readout.md](Readout.md) | Product story of HCNN-as-readout (may lag this checklist) |
| [CPP_SDK.md](CPP_SDK.md) | HypercubeESN public C++ API |
| [../third_party/HypercubeCNN/VENDORED.md](../third_party/HypercubeCNN/VENDORED.md) | Vendor pin and update rule |
| Upstream [HypercubeCNN docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeCNN/blob/main/docs/CPP_SDK.md) | Canonical HCNN public API |
