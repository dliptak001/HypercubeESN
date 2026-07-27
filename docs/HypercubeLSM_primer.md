# LSM for Hypercube people — clocked LIF + features

> Status: **concept primer** (no landed code). Companion to
> [HypercubeLSM.md](HypercubeLSM.md).  
> Audience: people who already know HypercubeESN (fixed hypercube reservoir,
> freeze dynamics, train readout).  
> Goal: one short page of **what moves when**, in ESN vocabulary.

---

## 30-second map

| HypercubeESN | HypercubeLSM (this page) |
|--------------|---------------------------|
| Continuous field `state[v]` | Membrane voltage `V[v]` + spikes |
| `tanh` + leak blend | Integrate, threshold, reset, refractory |
| Dense float every step | Spikes are **rare**; head reads **traces** |
| Spectral radius scales recurrence | Regime protocol scales E/I weights (separate note) |
| Freeze reservoir → train readout | Freeze **liquid** → train readout |

Same crystal: vertex `v`, neighbors `v XOR (1<<i)`, N = 2^dim. Different phase:
**events in time**, not a smooth rate field.

---

## Units on the cube

- One neuron per vertex `v = 0 .. N-1`.
- Recurrent synapses only on Hamming edges `(v, u)` with `u = v XOR (1<<i)`.
- Weights `W[v, i]` frozen after construction (excitatory or inhibitory).
- Time is **clocked**: fixed step `dt`. No event queue in v1.

Per vertex, hold at least:

| Symbol | Meaning |
|--------|---------|
| `V[v]` | Membrane voltage |
| `ref[v]` | Refractory countdown (steps left, or 0 if free) |
| `I[v]` | Input current this step (synaptic + external) |
| `spike[v]` | 1 if this step emitted a spike, else 0 |

---

## Clocked LIF (equations)

All of this is **per vertex**, every `dt`. Prefer simple Euler; exact form can
change when code lands — the shape should not.

### 1. Leak and integrate (if not refractory)

While `ref[v] == 0`:

```
V[v]  ←  V[v] + (dt / tau_m) * ( -V[v] + I[v] )
```

Interpretation:

- `tau_m` — membrane time constant (seconds). Larger → slower fade of past drive.
- Rest is written at 0 for simplicity; a rest potential `V_rest` is an affine
  shift (`V − V_rest` in the leak term) if you want one later.
- `I[v]` already includes gain (weights absorb “resistance”).

Equivalent one-liner with decay factor `alpha = exp(-dt / tau_m)` (same idea,
slightly different discretization):

```
V[v]  ←  alpha * V[v] + (1 - alpha) * I[v]
```

Pick **one** discrete form in the implementation and stick to it.

### 2. Fire and reset

```
if V[v] >= theta:
    spike[v] = 1
    V[v]     = V_reset
    ref[v]   = refr_steps          # floor(t_ref / dt), at least 1 if t_ref > 0
else:
    spike[v] = 0
```

| Symbol | Role |
|--------|------|
| `theta` | Spike threshold |
| `V_reset` | Voltage after spike (often 0 or below rest) |
| `t_ref` / `refr_steps` | Hard silence after a spike — caps max rate |

### 3. Refractory tick

If `ref[v] > 0` at the start of the step: do **not** integrate (or clamp `V`),
do not fire; then `ref[v] ← ref[v] - 1`.

### 4. Build current `I[v]` from spikes (recurrent)

**v1 synaptic model (instant jump + optional filter):**

Each step, start from external drive only, then add recurrent contributions from
**previous-step** spikes (synchronous update — same discipline as ESN’s “read
published state, write new state”):

```
I[v] = I_ext[v]

for i = 0 .. dim-1:
    u = v XOR (1 << i)
    if spike_prev[u] == 1:
        I[v] += W[v, i]
```

Optional **exponential synapse** (smoother, often nicer dynamics) — keep a
synaptic variable `S[v]`:

```
# decay always
S[v] ← S[v] * exp(-dt / tau_s)

# on previous spike at neighbor u along axis i:
S[v] ← S[v] + W[v, i] * spike_prev[u]

I[v] = I_ext[v] + S[v]
```

`tau_s` is the synaptic time constant. Instant-jump is `tau_s → 0` (all mass in
one step).

### 5. External input (encoding lives outside)

`I_ext[v]` comes from the host’s encoding of the task stream into currents or
forced spikes, for example:

- **Current injection:** continuous signal written onto a vertex block (ESN-like
  channel layout), then maybe no forced spikes.
- **Forced spikes:** Poisson or latency encoder sets `spike` / adds to `I` on
  chosen vertices.

Encoding is **not** the liquid. The liquid only sees `I_ext` and recurrent
spikes.

### One-step order (recommended)

```
1. Build I[v] from I_ext + recurrent (using spike_prev, W)
2. For each v: refractory? else integrate V
3. For each v: threshold → spike, reset, set refractory
4. spike_prev ← spike
5. Update readout traces (next section)
```

Synchronous, O(N · dim) per step — same complexity class as one ESN slice gather
without history depth M.

---

## Preferred readout features (multi-τ traces)

The head should **not** train on the raw spike raster alone for the product
face. Spikes are sparse and binary; HypercubeCNN wants a **smooth field on
vertices**.

### Trace definition

Maintain K exponential **firing traces** per vertex (K ≥ 1). On each step,
after spikes are known:

```
for k = 0 .. K-1:
    for v = 0 .. N-1:
        r[k][v] ← r[k][v] * exp(-dt / tau_trace[k])
        if spike[v] == 1:
            r[k][v] ← r[k][v] + 1
```

| Symbol | Role |
|--------|------|
| `r[k][v]` | Trace channel k at vertex v |
| `tau_trace[k]` | Time constant of that channel (e.g. short / medium / long) |
| `+1` on spike | Unit impulse; amplitude can be a scale later |

**Readout input at time t** (intent):

```
features = pack channels k=0..K-1 of r[k][*]   # length K * N
# or for HCNN: start_dim = dim, input_channels = K  (full capacity N)
```

| Recipe | When to use |
|--------|-------------|
| **Multi-τ traces (above)** | Preferred product face; HCNN-friendly |
| Spike counts in a window | Baseline / ridge diagnostic |
| Membrane `V[*]` | Fallback; less “spike-native” |

Training contract unchanged: **freeze liquid, train readout** on recorded
`features` after warmup (linear first; HCNN when K-channel fields are smooth).

### Why this matches HypercubeESN intuition

| ESN | LSM traces |
|-----|------------|
| Dense `Outputs()` length N | Dense `r[k]` length N (× K channels) |
| Multi-slice B packs ages into a bigger cube | Multi-τ packs **timescales** into channels (v1) or later into geometry |
| HCNN on vertex field | HCNN on trace field — same topology story |

Traces are the liquid’s “published state” for learning. Spikes are how the
liquid talks to itself.

---

## Minimal parameter set (survey later, not magic)

| Group | Knobs |
|-------|--------|
| Time | `dt` |
| Neuron | `tau_m`, `theta`, `V_reset`, `t_ref` |
| Synapse | weight draw + E/I balance; optional `tau_s` |
| Regime | global scale so mean rate under a probe drive sits in a band |
| Features | `K`, `tau_trace[0..K)` |
| Size | `dim` → N = 2^dim |

ESN analogy: you survey seed / spectral radius / M. LSM: you survey seed /
**regime scale** / τ’s / trace stack. Same product habit, different knobs.

---

## What this page deliberately skips

- Event-driven queues and neuromorphic deployment  
- Axonal delays (possible later analogue of ESN history depth)  
- STDP / training the liquid  
- Exact regime algorithm (mean-rate band vs branching — open in
  [HypercubeLSM.md](HypercubeLSM.md))  
- Benchmark numbers  

---

## One diagram

```
  I_ext (encoded task)
       │
       ▼
  ┌─────────────────────────────────────┐
  │  Hypercube liquid (frozen)          │
  │  V, spikes on XOR edges, clocked dt │
  └─────────────────────────────────────┘
       │ spike[v]
       ▼
  multi-τ traces r[k][v]     ← preferred "state" for the head
       │
       ▼
  readout (linear or HCNN) ──▶ y
```

Only the readout learns. The crystal is the hypercube; the phase is spikes.

---

## Related

| Doc | Role |
|-----|------|
| [HypercubeLSM.md](HypercubeLSM.md) | Product concept, scope, success criteria |
| [Reservoir.md](Reservoir.md) | Rate-based topology twin |
| [Readout.md](Readout.md) | How a vertex field becomes y |
