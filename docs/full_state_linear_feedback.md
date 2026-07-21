# Full-state linear feedback — theory and landed API

> Status: **landed (mechanism)**; training V is deferred.  
> Ehlers, Nurdin & Soh (arXiv:2312.15141) motivate a full-state gain **V**.  
> HypercubeESN applies it as a **dedicated internal drive port** (`B_fsf`), not by
> hijacking input or external feedback. Config and API live on
> `ReservoirConfig` / `ESN`; dynamics notes in [Reservoir.md](Reservoir.md).
>
> **Binding:** there is **no** separate staging vector **w**. Staging uses the
> same **V** that builds φ: `pad[v] = φ · V[v]`. That choice is permanent.

## Paper idea (compact)

```
φ_k     = V · x_k                    (V length N; scalar drive)
pad     = φ_k * V                    (elementwise; same V — not a second vector)
x_{k+1} = g(A x_k + B u_k + … + gather(B_fsf, pad) + …)
```

Schematic effective map still has the paper’s **rank-1-in-φ** flavor (one scalar
from the full state, then a fixed spatial inject pattern determined by V and
B_fsf). Hypercube topology and the construction-time spectral-radius rescale of
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
| Staging | `vtx_fsf_[v] = φ * V[v]` (**w ≡ V forever**; no separate w) |
| Gather | XOR neighbor gather of `vtx_fsf_` through **B_fsf** (same form as input) |
| V | Length N, **init 0**; settable via `SetFullStateFeedbackGain` — **not** trained by the library |
| Weights `B_fsf` | U(−1,1) × `fsf_scaling/√dim` from standalone **`fsf_seed`** (not mixed from `seed`) |
| Warmup / Run | Still apply FSF when enabled (batch train sees the FSF-closed map) |
| Coexist | Independent of multi-channel **external** feedback (e.g. Lorenz) |

```cpp
ESNConfig cfg;
cfg.reservoir.full_state_feedback = true;
cfg.reservoir.fsf_seed            = 1;      // seeds B_fsf only
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
B_fsf from config + `fsf_seed`, re-`Set` V if nonzero).

## What is deliberately not in the library / design

- A separate staging vector **w** (rejected forever; **w ≡ V**)
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
  FSF                — internal: φ = V·x, pad = φ⊙V, gather B_fsf
```

## Reference

Ehlers, P. J., Nurdin, H. I., & Soh, D. *Improving the Performance of Echo State
Networks Through Feedback*. arXiv:2312.15141.

---

## Appendix: FSF for Dummies

**Short answer:** **V is not the feedback weight matrix.** You have **one** length-N
vector **V** and **one** inject weight block **B_fsf**. V both **scores** the state
and **paints** the FSF field; B_fsf is how that field enters each neuron.

### What happens each step (FSF on)

```text
  1. Have the full state x  (N numbers, one per neuron)

  2. Make ONE number:
        φ  =  V · x   =  V[0]·x[0] + V[1]·x[1] + … + V[N-1]·x[N-1]

  3. Paint the FSF pads with the SAME V (not a second vector w):
        pad[v]  =  φ * V[v]     for each vertex v
        (so neighbors can differ — gather is load-bearing unless V is flat)

  4. Each neuron mixes its neighbors’ pads with its own B_fsf weights
        (XOR gather — same style as normal input / external feedback)

  5. That sum joins recurrent + input (+ external feedback) → activation → new state
```

| Symbol | What it is | Shape |
|--------|------------|--------|
| **x** | Reservoir state | N numbers |
| **V** | Score state → φ **and** stage pad[v] = φ·V[v] | **vector length N** |
| **φ** (phi) | One drive strength this step | **scalar** |
| **pad** / `vtx_fsf_` | Per-vertex FSF field this step | N numbers |
| **B_fsf** | How pads are gathered into each neuron | **weight block** N×dim |

There is **no** separate staging vector **w**. That design is closed forever: **w ≡ V**.

### Mental model vs the code

A natural first picture: collect all N → sum → threshold → one feedback value →
redistribute through a weight matrix.

| That picture | Actual |
|--------------|--------|
| Collect all N | Yes — full state **x** |
| Sum | **Weighted** sum: φ = V·x |
| Threshold | **No** threshold. φ is a float |
| One feedback strength | Yes — **φ** |
| Paint onto vertices | **pad[v] = φ · V[v]** (same V) |
| Into each neuron | **B_fsf** neighbor gather of **pad** |

So:

- **V** = “how do I score x into φ **and** how do I weight pads by vertex?”
- **B_fsf** = “how does this neuron mix neighbors’ pads?”

### Tiny numeric sketch (N = 3)

```text
x = (0.5, -0.2, 0.1)
V = (1, 0, 0)
  φ = 0.5
  pad = (0.5·1, 0.5·0, 0.5·0) = (0.5, 0, 0)

V = (0, 0, 0)
  φ = 0, pad = (0,0,0)  →  FSF contributes nothing

Then B_fsf gathers neighbors’ pad values into each neuron’s sum.
```

### Where `fsf_seed` and `fsf_scaling` sit

- **`fsf_seed` / `fsf_scaling`** build **B_fsf** only (inject/gather weights).
- **V** is a settable length-N vector (default **0**). With V = 0, φ and pads are
  zero even if the port is allocated.

### One-line summary

**φ** = V · x. **Pads** = φ ⊙ V (same V). **B_fsf** gathers pads into the dynamics.  
No second N-vector. Ever.
