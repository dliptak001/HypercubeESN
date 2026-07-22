# Full-state linear feedback — theory and landed API

> Status: **landed (mechanism)**.  
> Form **φ = V · x** (scalar), gather φ through **B_fsf**. Equivalent to a
> uniform external-feedback D=1 field without a length-N staging buffer.
> Config on `ReservoirConfig` / `ESN`.

## Design (canonical)

```
Construction (fsf_seed, standalone — not reservoir.seed):
  V[i]   ~ U(-1, 1)                         // first N draws; φ only
  B_fsf  ~ U(-1, 1) * fsf_scaling / √dim    // next N·dim; only FSF strength knob

Each Step:
  φ = V · x                                 // one scalar for the whole reservoir
  UpdateState: s += φ * B_fsf[v,i]          // over dim neighbors (≡ φ · row-sum)
```

| Piece | Role |
|--------|------|
| **V** | Only builds φ (`fsf_v_`) |
| **φ** | Single-channel drive this step (`fsf_phi_`) |
| **B_fsf** | Inject weights (like external-feedback weights) |
| **`fsf_scaling`** | Only loudness knob (on B_fsf, like `input_scaling`) |

No length-N `vtx_fsf_` buffer: a uniform D=1 external-feedback fill would write
the same φ into every entry, so the neighbor gather collapses to multiplying
φ by each B_fsf weight (or by the row sum). No Set/Get of V. No separate stage
scale (it only multiplied `fsf_scaling`).

```cpp
ESNConfig cfg;
cfg.reservoir.full_state_feedback = true;
cfg.reservoir.fsf_seed            = 1;
cfg.reservoir.fsf_scaling         = 0.05f;
ESN esn(cfg);
```

## Drive-port map

```
  input              — InjectInput → vtx_input_ → gather B_in
  external feedback  — InjectExternalFeedback → vtx_ext → gather B_ext
  FSF                — φ = V·x (scalar) → gather B_fsf
```

![FSF one timestep](drive_ports_flow.jpg)

## Reference

Ehlers, P. J., Nurdin, H. I., & Soh, D. *Improving the Performance of Echo State
Networks Through Feedback*. arXiv:2312.15141.

---

## Appendix: FSF for Dummies

1. **φ = V · x** — only use of V; one scalar for the whole reservoir this step.  
2. **Gather** with B_fsf (no length-N staging buffer):

```cpp
s += fsf_phi_ * fsw[i];   // for each of dim neighbors
// equivalent: s += fsf_phi_ * sum(fsw[0..dim))
```

Strength: **`fsf_scaling`** on B_fsf only.
