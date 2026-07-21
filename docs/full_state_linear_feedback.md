# Full-state linear feedback — theory and landed API

> Status: **landed (mechanism)**.  
> Ehlers, Nurdin & Soh (arXiv:2312.15141) motivate a full-state gain **V**.  
> HypercubeESN applies it as a **dedicated internal drive port** (`B_fsf`), not by
> hijacking input or external feedback. Config and API live on
> `ReservoirConfig` / `ESN`; dynamics notes in [Reservoir.md](Reservoir.md).
>
> **Binding:** **w ≡ V** forever (`pad[v] ∝ φ · V[v]`). No separate staging vector.
> **V** is U(−1,1) from `fsf_seed` (no scale baked in). **No** Set/Get of V.

## Design (canonical)

```
Construction (fsf_seed, standalone RNG — not reservoir.seed):
  V[i]     ~ U(-1, 1)                         // first N draws; full range stored
  B_fsf    ~ U(-1, 1) * fsf_scaling / √dim    // next N·dim draws

Each Step:
  φ        = V · x
  pad[v]   = fsf_stage_scaling * φ * V[v]     // w ≡ V
  UpdateState: gather neighbors of pad through B_fsf
```

### Why both `fsf_scaling` and `fsf_stage_scaling`?

They scale **different objects** on the path from state to neuron sum:

| Scale | What it multiplies | Analogy |
|-------|--------------------|---------|
| **`fsf_stage_scaling`** | How strong the **pad field** is: pad ∝ stage · (V·x) · V | Volume of the FSF *message* painted on the cube |
| **`fsf_scaling`** | How strong **B_fsf** is (inject weights, like `input_scaling`) | How hard that message is *wired into* each neuron |

```text
  x ──(V)──► φ ──(V, stage_scale)──► pad ──(B_fsf, fsf_scaling)──► neuron sum
```

- Change **stage** → same inject wiring, louder/softer pads.  
- Change **fsf_scaling** → same pads, stronger/weaker inject weights (and relative
  row geometry stays whatever B_fsf drew; overall magnitude moves with the scale).

They are **not** redundant the way score×stage was (that pair only appeared as a
product into pad). Stage acts **before** the gather; B_fsf scale is **in** the
gather weights — same split as task input: raw drive vs `input_scaling` on weights.

`Create(GetConfig())` rebuilds the same V and B_fsf (deterministic draw order).

## Landed realization

| Item | Behavior |
|------|----------|
| Enable | Construction-only: `full_state_feedback` |
| Off | Zero FSF allocation |
| V | Length N; **U(−1,1)** from `fsf_seed` (first N draws) |
| B_fsf | N×dim; U(−1,1)×`fsf_scaling`/√dim (same seed, next draws) |
| φ / pads | φ = V·x; pad[v] = `fsf_stage_scaling` · φ · V[v] |
| Gather | XOR neighbors of pad × B_fsf |
| API | No Set/Get V |

```cpp
ESNConfig cfg;
cfg.reservoir.full_state_feedback = true;
cfg.reservoir.fsf_seed            = 1;
cfg.reservoir.fsf_scaling         = 0.5f;  // B_fsf
cfg.reservoir.fsf_stage_scaling   = 1.0f;  // pads
ESN esn(cfg);
```

## What is not in the design

- Separate **w** (forever **w ≡ V**)
- Runtime Set/Get of V
- Separate score scale (redundant with stage under w ≡ V)
- Multi-channel FSF
- Library training of V

## Drive-port map

```
  input              — always
  external feedback  — optional caller values
  FSF                — internal: φ = V·x, pad ∝ φ⊙V, gather B_fsf
```

![Drive ports: input, external feedback, and FSF into one reservoir step](drive_ports_flow.jpg)

## Reference

Ehlers, P. J., Nurdin, H. I., & Soh, D. *Improving the Performance of Echo State
Networks Through Feedback*. arXiv:2312.15141.

---

## Appendix: FSF for Dummies

**V** is a random ±1 pattern from seed (score + paint). **B_fsf** is the inject
weight block. **stage_scale** loudness of the pad field; **fsf_scaling** loudness
of the inject weights.

### Each step

```text
  1. Full state x
  2. φ  =  V · x
  3. pad[v]  =  stage_scale · φ · V[v]
  4. Gather neighbors’ pads × B_fsf into each neuron
  5. + recurrent + input (+ external feedback) → new state
```

### One-line summary

**V** from seed in [−1,1]; **stage_scale** sizes pads; **fsf_scaling** sizes B_fsf.
No second N-vector. No Set/Get.
