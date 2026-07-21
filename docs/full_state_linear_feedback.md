# Full-state linear feedback — theory and landed API

> Status: **landed (mechanism)**; training V is deferred.  
> Ehlers, Nurdin & Soh (arXiv:2312.15141) motivate a full-state gain **V**.  
> HypercubeESN applies it as a **dedicated internal drive port** (`B_fsf`), not by
> hijacking input or external feedback. Config and API live on
> `ReservoirConfig` / `ESN`; dynamics notes in [Reservoir.md](Reservoir.md).

## Paper idea (compact)

```
φ_k     = V · x_k          (V length N; scalar drive)
x_{k+1} = g(A x_k + B u_k + … + B_fsf φ_k + …)
```

Schematic effective map: **A′ ≈ A + B_fsf Vᵀ** (structure of the paper’s
A′ = A + B Vᵀ, with a dedicated injection matrix `B_fsf` rather than reusing
`B_in`). Hypercube topology and the construction-time spectral-radius rescale of
the **recurrent** block are unchanged. Drive ports (input, external feedback,
FSF) sit **outside** that rescale.

This is **not** Jaeger output feedback (ŷ → reservoir). FSF uses a projection of
the full state. External ŷ feedback remains a separate, caller-owned port
([reservoir_feedback_mechanism.md](reservoir_feedback_mechanism.md)).

## Landed realization (HypercubeESN)

| Item | Behavior |
|------|----------|
| Enable | Construction-only: `ReservoirConfig::full_state_feedback` |
| Off | **Zero** FSF allocation (no buffer, weights, or V) |
| φ | Scalar: φ = V · x each `Step` from pre-update published state |
| V | Length N, **init 0**; settable via `SetFullStateFeedbackGain` — **not** trained by the library |
| Weights `B_fsf` | U(−1,1) × `fsf_scaling/√dim` from standalone **`fsf_seed`** (not mixed from `seed`) |
| Staging | Internal inside `Reservoir::Step`; no public inject-FSF |
| Warmup / Run | Still apply FSF when enabled (batch train sees the FSF-closed map) |
| Coexist | Independent of multi-channel **external** feedback (e.g. Lorenz) |

```cpp
ESNConfig cfg;
cfg.reservoir.full_state_feedback = true;
cfg.reservoir.fsf_seed            = 1;      // config param; default 1
cfg.reservoir.fsf_scaling         = 0.5f;
ESN esn(cfg);

std::vector<float> V(esn.ReservoirNeuronCount(), 0.f);
// set V for your A/B experiment…
esn.SetFullStateFeedbackGain(V.data(), V.size());

esn.ReservoirWarmup(u, T);   // FSF applies automatically
// … Train / Predict as usual
```

Accessors: `FullStateFeedbackEnabled()`, `GetFullStateFeedbackGain(...)`.  
`GetConfig()` returns enable / seed / scaling; **V is not in the config** (rebuild
weights from config + seeds, re-`Set` V if nonzero).

## What is deliberately not in the library yet

- Nested optimization / training of V (paper GD + linear probe, etc.)
- Persistence of V next to readout weights
- Python FSF bindings
- Multi-channel FSF (permanently out of scope — scalar φ only)

Callers who want a trained V run their own outer loop and call
`SetFullStateFeedbackGain`. See arXiv:2312.15141 for theory and nested-loop
practice; do not claim paper NMSE numbers on the HCNN readout without a matching
linear-probe protocol.

## Drive-port map

```
  input              — always (InjectInput / ReservoirStep inputs)
  external feedback  — optional caller values (InjectExternalFeedback /
                       ReservoirStep second arg)
  FSF                — optional internal φ = V·x (this document)
```

## Reference

Ehlers, P. J., Nurdin, H. I., & Soh, D. *Improving the Performance of Echo State
Networks Through Feedback*. arXiv:2312.15141.
