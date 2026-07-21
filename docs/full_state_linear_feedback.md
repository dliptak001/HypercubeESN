# Full-state linear feedback — theory and landed API

> Status: **landed (mechanism)**.  
> Ehlers, Nurdin & Soh (arXiv:2312.15141) motivate a full-state gain **V**.  
> HypercubeESN applies it as a **dedicated internal drive port** (`B_fsf`), not by
> hijacking input or external feedback. Config and API live on
> `ReservoirConfig` / `ESN`; dynamics notes in [Reservoir.md](Reservoir.md).
>
> **Binding:** there is **no** separate staging vector **w**. Staging uses the
> same **V** that builds φ: `pad[v] ∝ φ · V[v]`. That choice is permanent.
>
> **Why w ≡ V:** a free **w ≠ V** would only reshape how φ is painted; the extra
> N-vector is not worth the management cost.
>
> **V is construction-time only:** drawn U(−1,1) from `fsf_seed` with **no scale
> baked in**. Independent scales apply in `Step` for score (φ) and stage (pads).
> **No** Set/Get of V.

## Design (canonical)

```
Construction (fsf_seed, standalone RNG — not reservoir.seed):
  V[i]     ~ U(-1, 1)                         // first N draws; full range stored
  B_fsf    ~ U(-1, 1) * fsf_scaling / √dim    // next N·dim draws

Each Step:
  φ        = fsf_score_scaling * (V · x)      // score side
  pad[v]   = fsf_stage_scaling * φ * V[v]     // stage side (w ≡ V)
  UpdateState: gather neighbors of pad through B_fsf
```

Score and stage scales are independent: tune “how hard we listen” vs “how hard we
paint” without redrawing V or coupling them by baking a single scale into V.

`Create(GetConfig())` rebuilds the same V and B_fsf (deterministic draw order).

## Landed realization

| Item | Behavior |
|------|----------|
| Enable | Construction-only: `full_state_feedback` |
| Off | Zero FSF allocation |
| V | Length N; **U(−1,1)** from `fsf_seed` (first N draws) |
| B_fsf | N×dim; U(−1,1)×`fsf_scaling`/√dim (same seed, next draws) |
| Score | φ = `fsf_score_scaling` · (V · x) in `Step` |
| Stage | pad[v] = `fsf_stage_scaling` · φ · V[v] in `Step` |
| Gather | XOR neighbors of pad × B_fsf in `UpdateState` |
| API | No Set/Get V — enable + seed + three scales |

```cpp
ESNConfig cfg;
cfg.reservoir.full_state_feedback = true;
cfg.reservoir.fsf_seed            = 1;
cfg.reservoir.fsf_scaling         = 0.5f;  // B_fsf
cfg.reservoir.fsf_score_scaling   = 1.0f;  // φ
cfg.reservoir.fsf_stage_scaling   = 1.0f;  // pads
ESN esn(cfg);
esn.ReservoirWarmup(u, T);
```

## What is not in the design

- Separate staging vector **w** (forever **w ≡ V**)
- Runtime Set/Get of V
- Baking score/stage scale into stored V
- Multi-channel FSF
- Library training of V

## Drive-port map

```
  input              — always
  external feedback  — optional caller values
  FSF                — internal: score V·x, stage φ⊙V, gather B_fsf
```

## Reference

Ehlers, P. J., Nurdin, H. I., & Soh, D. *Improving the Performance of Echo State
Networks Through Feedback*. arXiv:2312.15141.

---

## Appendix: FSF for Dummies

**V is not the feedback weight matrix.** One length-N vector **V** (random ±1
pattern from seed) and one inject block **B_fsf**. Scales are volumes applied when
the signal is used, not folded into V.

### Each step

```text
  1. Full state x
  2. φ  =  score_scale · (V · x)
  3. pad[v]  =  stage_scale · φ · V[v]     (same V; w ≡ V)
  4. Gather neighbors’ pads × B_fsf into each neuron
  5. + recurrent + input (+ external feedback) → activation → new state
```

| Symbol | Role | Shape |
|--------|------|--------|
| **V** | Score and paint pattern (U(−1,1) from seed) | N |
| **φ** | One drive strength this step | scalar |
| **pad** | Per-vertex FSF field | N |
| **B_fsf** | Gather pads into neurons | N×dim |
| **score_scale** | Tunable strength of φ | scalar |
| **stage_scale** | Tunable strength of painting | scalar |

### One-line summary

**V** fixed from seed in [−1,1]; **φ** and **pads** use **separate scales** in
`Step`; **B_fsf** injects. No second N-vector. No Set/Get.
