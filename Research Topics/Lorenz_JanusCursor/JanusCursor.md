# JanusCursor

## Concept

A pair of **counter-moving cursors** over one linear index space, used to half-anchor a free-run:

- **Past cursor** — Anchor — always real history on the input port; free-run stabilizer; may walk below the training window into a past runway.
- **Future cursor** — Horizon — teacher in train; self-prediction in free-run; walking past the window’s upper edge is the generative signal.

Both cursors share one window `[lb, ub]` centered at `center_index`. `Reset()` seats them at **opposite edges**; each `Step()` moves them in **opposite directions** so they sweep toward and then past each other.

```text
 array index n:   0 ······· lb ····· center ····· ub ········· stream end
                  │         │           │           │            │
                 seed   train edge   midpoint   train edge   eval / free-run
                        (lb)        (center)      (ub)         tail

 window [lb, ub]     — training / washout region the cursors start on
 [0, lb)             — past free-run runway (anchor history past may enter)
 (ub, end]           — prediction / eval runway (future is generative here)
```

```text
stream:  0 ....... lb =========== center =========== ub ....... N
                    ^future                          ^past       (after Reset)
         future ──────────────────>          <────────────── past  (each Step)
```

The class is **system-agnostic**: any array addressable by `int32_t` index can sit under the dual indices. Lorenz-63 is only the in-tree instantiation.

### Hypothesis: The past cursor’s job is **dual**

1. **Horizon / VPT** — real history on the input port can delay the first threshold-crossing.  
2. **Lock / re-lock** — a continuous real past is a **drive** signal. While free-run is generative on the future port, the reservoir is still driven by on-attractor truth through the past port.

---
