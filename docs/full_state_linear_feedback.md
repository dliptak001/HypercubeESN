# Full-state linear feedback — theory and landed API

> Status: **landed (mechanism)**.  
> Form **φ = V · x**, inject like **external feedback with one channel**: fill
> `vtx_fsf_` with φ, XOR-gather × **B_fsf**. Config on `ReservoirConfig` / `ESN`.

## Design (canonical)

```
Construction (fsf_seed, standalone — not reservoir.seed):
  V[i]   ~ U(-1, 1)                         // first N draws; φ only
  B_fsf  ~ U(-1, 1) * fsf_scaling / √dim    // next N·dim; only FSF strength knob

Each Step:
  φ = V · x
  vtx_fsf_[0 .. N) = φ                      // ext-fb D=1 style
  UpdateState: s += vtx_fsf_[neighbor] * B_fsf[v,i]
```

| Piece | Role |
|--------|------|
| **V** | Only builds φ |
| **φ** | Single-channel drive this step |
| **`vtx_fsf_`** | Staged field (all entries φ) |
| **B_fsf** | Inject weights (like external-feedback weights) |
| **`fsf_scaling`** | Only loudness knob (on B_fsf, like `input_scaling`) |

No Set/Get of V. No separate stage scale (it only multiplied `fsf_scaling`).

```cpp
ESNConfig cfg;
cfg.reservoir.full_state_feedback = true;
cfg.reservoir.fsf_seed            = 1;
cfg.reservoir.fsf_scaling         = 0.5f;
ESN esn(cfg);
```

## Drive-port map

```
  input              — InjectInput → vtx_input_ → gather B_in
  external feedback  — InjectExternalFeedback → vtx_ext → gather B_ext
  FSF                — φ = V·x → fill vtx_fsf_ with φ → gather B_fsf
```

![FSF one timestep](drive_ports_flow.jpg)

## Reference

Ehlers, P. J., Nurdin, H. I., & Soh, D. *Improving the Performance of Echo State
Networks Through Feedback*. arXiv:2312.15141.

---

## Appendix: FSF for Dummies

1. **φ = V · x** — only use of V.  
2. **Write φ into every entry of `vtx_fsf_`** — same as one external-feedback channel.  
3. **Gather** with B_fsf:

```cpp
s += vtx_fsf_[v ^ NearestMask(i)] * fsw[i];
```

Strength: **`fsf_scaling`** on B_fsf only.
