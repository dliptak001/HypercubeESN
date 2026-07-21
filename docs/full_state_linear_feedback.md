# Full-state linear feedback — theory and landed API

> Status: **landed (mechanism)**.  
> Ehlers, Nurdin & Soh (arXiv:2312.15141) motivate a full-state gain **V**.  
> HypercubeESN applies it as a **dedicated internal drive port** (`B_fsf`), not by
> hijacking input or external feedback. Config and API live on
> `ReservoirConfig` / `ESN`; dynamics notes in [Reservoir.md](Reservoir.md).
>
> **Binding:** there is **no** separate staging vector **w**. Staging uses the
> same **V** that builds φ: `pad[v] = φ · V[v]`. That choice is permanent.
>
> **Why w ≡ V:** a free **w ≠ V** would only reshape how φ is painted on the cube.
> Any practical gain over tying the paint to V is small next to the cost of a
> second N-vector. **w ≡ V** keeps one gain vector and an honest gather.
>
> **V is construction-time only:** drawn from `fsf_seed` × `fsf_v_scaling` when
> FSF is enabled. **No** Set/Get of V.

## Paper idea (compact)

```
φ_k     = V · x_k                    (V length N; scalar drive)
pad     = φ_k * V                    (elementwise; same V)
x_{k+1} = g(A x_k + B u_k + … + gather(B_fsf, pad) + …)
```

Drive ports (input, external feedback, FSF) sit **outside** the recurrent
spectral-radius rescale. This is **not** Jaeger output feedback (ŷ → reservoir).

## Landed realization (HypercubeESN)

| Item | Behavior |
|------|----------|
| Enable | Construction-only: `full_state_feedback` |
| Off | **Zero** FSF allocation |
| V | Length N; U(−1,1)×`fsf_v_scaling` from **`fsf_seed`** (first N draws) |
| B_fsf | N×dim; U(−1,1)×`fsf_scaling`/√dim from **same seed** (next N·dim draws) |
| φ / pads | φ = V·x; `pad[v] = φ·V[v]` (**w ≡ V**) |
| Gather | XOR neighbors of `pad` × B_fsf |
| API | No Set/Get V — enable + seed + scales only |
| Warmup / Run | Apply FSF when enabled |
| Coexist | Independent of multi-channel **external** feedback |

```cpp
ESNConfig cfg;
cfg.reservoir.full_state_feedback = true;
cfg.reservoir.fsf_seed            = 1;
cfg.reservoir.fsf_scaling         = 0.5f;  // B_fsf
cfg.reservoir.fsf_v_scaling       = 1.0f;  // V
ESN esn(cfg);
// V and B_fsf already drawn; no SetFullStateFeedbackGain
esn.ReservoirWarmup(u, T);
```

`Create(GetConfig())` rebuilds the same V and B_fsf (deterministic in `fsf_seed`
and the two scales).

## What is deliberately not in the design

- Separate staging vector **w** (rejected forever; **w ≡ V**)
- Runtime Set/Get of V
- Multi-channel FSF
- Library training of V (outer-loop research may still re-implement offline)

## Drive-port map

```
  input              — always
  external feedback  — optional caller values
  FSF                — internal: φ = V·x, pad = φ⊙V, gather B_fsf
```

## Reference

Ehlers, P. J., Nurdin, H. I., & Soh, D. *Improving the Performance of Echo State
Networks Through Feedback*. arXiv:2312.15141.

---

## Appendix: FSF for Dummies

**Short answer:** **V is not the feedback weight matrix.** You have **one** length-N
vector **V** and **one** inject weight block **B_fsf**. V both **scores** the state
and **paints** the FSF field; B_fsf is how that field enters each neuron. Both are
drawn at construction from `fsf_seed` (plus scales). No Set/Get.

### What happens each step (FSF on)

```text
  1. Full state x  (N numbers)

  2. φ  =  V · x

  3. pad[v]  =  φ * V[v]     (same V; w ≡ V forever)

  4. Each neuron: XOR-gather neighbors’ pads × B_fsf

  5. Sum with recurrent + input (+ external feedback) → activation → new state
```

| Symbol | What it is | Shape |
|--------|------------|--------|
| **x** | Reservoir state | N |
| **V** | Score → φ and paint pads (from seed + `fsf_v_scaling`) | N |
| **φ** | One drive strength this step | scalar |
| **pad** | Per-vertex FSF field | N |
| **B_fsf** | Gather pads into each neuron (from seed + `fsf_scaling`) | N×dim |

### One-line summary

**φ** = V · x. **Pads** = φ ⊙ V. **B_fsf** gathers pads. V and B_fsf come from
**seed + scales** at construction — nothing else.
