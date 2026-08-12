# Lorenz - Janus Cursor Research Archive

Internal snapshot of the Janus dual-cursor free-run harness.

Closed-loop experiment: train a HypercubeESN online on a Lorenz-63 orbit with a
**Janus cursor** presentation, then free-run with **self-feedback on the
future port** while the past port stays anchored to real history.

This is **assisted / half-anchored** free-run — continuous partial observation
on the past — **not** classical unassisted autonomous generation (no true drive
at all).

---

## 1. Janus cursor concept

The harness is built around a **pair of counter-moving indices** over one forward orbit.

| Cursor | Role | Motion |
|--------|------|--------|
| **Past** | **Anchor** — always real history on the input port; free-run stabilizer | Starts at `ub`, each `Step()` decrements |
| **Future** | **Horizon** — teacher in train; self-prediction in free-run | Starts at `lb`, each `Step()` increments |

`Reset()` seats them at **opposite edges** of a shared window. Each `Step()` walks
them toward and then past each other.

```text
stream:  0 ....... lb =========== center =========== ub ....... N
                    ^future                          ^past       (after Reset)
         future ──────────────────>          <────────────── past  (each Step)
```

---

## 2. Pipeline at a glance

```text
 LorenzAttractor (RK4 Lorenz-63; σ, ρ, β, dt as configured)
        │
        ▼
 LorenzDatastream  — integrate once, midpoint-offset + shared-scale → float S[·] ≈ [-1,1]
        │            inherits JanusCursor (past @ ub←, future @ lb→)
        │
        │  input port              (4): past   [x, y, z, x·z]  always real history
        │  external-feedback port  (4): future [x, y, z, x·z]  real in train; prediction in free-run
        ▼
 ESN  — fixed hypercube reservoir + online HCNN readout (3 outputs: x, y, z)
        │
        ├─ Train()   multi-epoch teacher-forced sweeps (horizon-1, prequential)
        └─ FreeRun() washout → generative self-feedback; VPT + free-run RMSE
```
