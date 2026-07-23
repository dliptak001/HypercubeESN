# HypercubeLSM — concept

> Status: **design concept** (not landed; not an in-tree HypercubeESN feature).  
> **Product / project name (working):** **HypercubeLSM** — intended as its own
> project in the HypercubeML ecosystem, **sibling** to HypercubeESN (not an
> example folder inside ESN).  
> **Paradigm:** liquid state machine (LSM) — spiking reservoir computing on the
> Boolean hypercube.  
> This file lives under HypercubeESN `docs/` as a **concept record** so the
> family story stays next to the rate-based reservoir docs. Implementation,
> packaging, and API belong in a future HypercubeLSM repo.

---

## One-line definition

**HypercubeLSM** is a liquid state machine whose recurrent “liquid” sits on
the vertices of a Boolean hypercube: spikes travel by XOR-addressed
Hamming-distance-1 edges, the liquid is frozen after construction, and only a
readout is trained.

Same topology contract as HypercubeESN; **spikes and timing** instead of
smooth rate units.

### Shelf line

> **Same crystal as Hopfield / CNN / ESN. Different phase of matter.**

Use that as the product face after the name: one shared Boolean-hypercube
geometry (the crystal); unit model and dynamics differ (the phase) — here,
spikes instead of rate fields or energy memory. Accurate enough for branding;
not a formal physics claim. Pair with the training contract nearby: freeze the
liquid, train the readout.

**Suggested pairing:**

> **HypercubeLSM** — a spiking liquid on the Boolean hypercube.  
> *Same crystal as Hopfield / CNN / ESN. Different phase of matter.*  
> Freeze the liquid; train the readout.

---

## What LSM means

**LSM** = **liquid state machine** (Maass, Natschläger, Markram and the
broader spiking-RC literature).

| Piece | Role |
|-------|------|
| **Liquid** | Recurrent network of **spiking** neurons; weights / delays fixed (or only lightly adapted) |
| **Input** | Spike trains (or encodings that become spikes) |
| **State** | High-dimensional snapshot of recent liquid activity (counts, traces, voltages, …) |
| **Readout** | Trained map from liquid state → labels or continuous targets |

It is the **spiking cousin** of echo state networks / reservoir computing:
rich fixed dynamics do the hard temporal lift; learning stays in a thin head.

Classical references (field context, not Hypercube-specific):

- W. Maass, T. Natschläger, and H. Markram, *Real-time computing without
  stable states: a new framework for neural computation based on
  perturbations*, Neural Computation 14(11):2531–2560, 2002.
- H. Jaeger, *The “echo state” approach…*, GMD Report 148, 2001 (rate-based
  RC parallel).
- M. Lukoševičius and H. Jaeger, *Reservoir computing approaches to recurrent
  neural network training*, Computer Science Review 3(3):127–149, 2009.

---

## Why a hypercube liquid

HypercubeESN already showed that the Boolean hypercube is a strong
**implicit graph** for reservoirs:

- **N = 2^DIM** neurons on vertices  
- Neighbors = single bit flips: `v XOR (1 << i)` for `i = 0 .. DIM-1`  
- **No adjacency list** — topology is computed, not stored  
- Local fan-in = **DIM** (exactly), vertex-transitive, multi-scale bit structure  

HypercubeLSM reuses that geometry for **events**:

| Property | LSM payoff |
|----------|------------|
| XOR routing | Spike delivery is an address op, hardware-friendly |
| Fixed fan-in DIM | Bounded local work per event / per step |
| Hamming locality | Nearby codes interact first; bit hierarchy = multi-scale pathways |
| Same N as ESN/CNN/Hopfield family | Shared mental model, optional shared readout geometry (e.g. HCNN on vertex fields) |

The product claim is not “yet another SNN framework.” It is:

> **This liquid is a hypercube** — same lattice as Hopfield / CNN / ESN, unit
> model = spikes.

Or, shorter:

> **Same crystal as Hopfield / CNN / ESN. Different phase of matter.**

---

## HypercubeESN vs HypercubeLSM

| | **HypercubeESN** | **HypercubeLSM** |
|--|------------------|------------------|
| Unit | Continuous / rate (e.g. leaky tanh field) | Spike times / binary events (+ optional traces) |
| State | Dense float vector on vertices (+ history slices) | Sparse events; features derived as counts, filters, or membranes |
| Drive | `InjectInput` / staged fields each step | Spike trains into vertices (or encoded from analog streams) |
| Recurrence | Weighted neighbor gather on published slices | Synaptic events along XOR edges (optional delays) |
| “Memory” knobs | Spectral radius, leak, history depth | Time constants, refractory, delays, mean rate / branching |
| Strength | Smooth dynamics, easy numerics, mature host | Timing codes, sparsity, neuromorphic / energy narrative |
| Training contract | Freeze reservoir, train readout | **Same** — freeze liquid, train readout |

**Sibling products**, not a mode flag inside ESN. Dependency direction when
built: HypercubeLSM may **depend on** shared ideas or readout libraries
(e.g. HypercubeCNN); it should not force spiking into the ESN core.

---

## Ecosystem placement

| Product | Paradigm slot |
|---------|----------------|
| **HypercubeHopfield** | Associative / energy memory on the cube (**first** in the family) |
| **HypercubeCNN** | Hierarchical spatial / conv features on the cube |
| **HypercubeESN** | Rate-based reservoir / temporal dynamics |
| **HypercubeMLP** (+ RIMT) | Static / MLP-class tasks via temporalization into ESN |
| **HypercubeLSM** (concept) | Spiking liquid / event-native reservoir on the cube |

Geometry story:

> Hamming-space computing on one lattice — memory, fields, rate dynamics,
> static maps, and (planned) **spike liquids**.

---

## Core scheme (conceptual)

### Liquid

- One spiking unit per hypercube vertex `v ∈ {0, …, N-1}`.  
- Recurrent synapses only along Hamming edges `(v, v XOR (1<<i))`.  
- Weights (and optional axonal delays) drawn once, **frozen**.  
- Minimal v1 neuron: **LIF** (leaky integrate-and-fire) or simpler
  threshold + leaky trace — not multi-compartment.

### Time

Two implementation families (pick one for v1 and document it):

| Style | Idea | Tradeoff |
|-------|------|----------|
| **Clocked** | Fixed `dt`; every step update membranes, emit spikes | Easier tooling, closer to ESN mental model |
| **Event-driven** | Priority queue of spike events | Closer to neuromorphic pure story; harder host integration |

**Recommendation for v1:** clocked LIF liquid — prove topology + readout path
before building a full event runtime.

### Input encoding

External signals become spikes, for example:

- rate / Poisson encoding of continuous channels  
- latency coding  
- direct event sensors (DVS-style) when available  

Static vectors (HypercubeMLP-class tasks) need an explicit **spike encoding**
or a bridge (including possible future use of RIMT-like ideas to temporalize
then spike — out of scope until both products exist).

### Readout features

The liquid must expose a **fixed-size feature field** the head can train on,
preferably aligned with vertex geometry:

| Feature recipe | Notes |
|----------------|-------|
| Spike counts in a window per vertex | Simple, robust |
| Exponential synaptic / firing traces | Smooth field for HCNN |
| Membrane voltage snapshot | Dense; less “spike-native” |
| Multi-window / multi-channel traces | Richer; higher dim |

**Training contract (classic LSM):** freeze liquid; train linear or
HypercubeCNN readout on collected features after warmup.

### Per-sample / streaming episode (sketch)

```
Build frozen hypercube liquid (DIM, seed, weight scales, τ, refractory, …)

For each trial or stream segment:
    Clear liquid state (or defined rest protocol)
    for t in time:
        inject input spikes for this dt (if any)
        liquid step (or process events)
        if t >= T_warmup:
            record readout features
    train / evaluate readout on recorded features → targets
```

---

## Design knobs (v1-oriented)

| Knob | Role |
|------|------|
| `DIM` / `N` | Cube size; liquid size |
| Neuron model | LIF parameters: τ_m, threshold, reset, refractory |
| Synaptic weights | Draw + scale on XOR edges; excitatory/inhibitory balance |
| Synaptic filter | Instantaneous vs exponential EPSC/IPSC |
| Delays | None in v1; optional later (powerful, costly) |
| `dt` | Clocked integration step |
| Input encoding | How continuous or symbolic input becomes spikes |
| `T_warmup` / collection window | Washout and feature horizon |
| Readout | Linear first; HCNN when vertex fields are smooth enough |
| Seed | Reproducible liquid draw |

**Non-goals for v1:** full STDP library, multi-chip runtime, surrogate-gradient
end-to-end training of the liquid, generic SNN framework APIs.

---

## Why it might matter (niche, not mass-market)

| Niche | Why LSM-on-cube |
|-------|-----------------|
| Neuromorphic R&D | Event-driven story + structured XOR routing |
| Timing-sensitive tasks | Spike times carry information rate units smear |
| Edge / energy narrative | Sparse events when rates are low |
| Ecosystem completeness | Spiking twin of HypercubeESN on the same lattice |
| Shared readout culture | Same “freeze dynamics, train head” economics as ESN |

**Not** a claim to replace general deep SNN toolkits or rate ESNs on every
benchmark. HypercubeLSM wins when **geometry + spikes + frozen liquid** is
the product reason.

---

## Relation to HypercubeESN (this repo)

| | |
|--|--|
| This document | Concept only, under `docs/` for family continuity |
| HypercubeESN code | Remains **rate-based** reservoir host |
| Future HypercubeLSM | Own project; may cite ESN docs for topology intuition and readout patterns |
| Shared pieces (optional later) | HypercubeCNN as readout; common DIM/N conventions; org-level branding |

Do **not** implement LSM as a silent mode inside `Reservoir` without an
explicit product decision — unit model, time, and cost models diverge enough
to deserve a separate core.

---

## Minimal success criteria (when built)

1. XOR-wired LIF (or simpler) liquid, DIM neighbors only.  
2. Inject spike trains, run for T steps, collect per-vertex features.  
3. Train a readout (linear acceptable for first proof; HCNN next).  
4. At least one temporal spike-pattern or classification demo.  
5. Short written comparison note vs HypercubeESN on a shared task family
   (fairness optional; narrative clarity required).  
6. README states: sibling of HypercubeESN; spikes not rates; freeze liquid.

---

## Open questions

- Clocked vs event-driven for the first public core  
- Exact v1 neuron and synapse equations  
- Whether delays ship in v1 or v2  
- Feature tensor layout for HypercubeCNN (channels × vertices)  
- Package name lock: **HypercubeLSM** vs broader HypercubeSpiking  
- How static data (HypercubeMLP) encodes into spikes if both products coexist  

---

## Naming

| Term | Use |
|------|-----|
| **HypercubeLSM** | Working product / project name (preferred: paradigm-familiar) |
| **Liquid state machine** | Full paradigm name in docs |
| **LSM** | Short handle |
| **Liquid** | The frozen spiking recurrent graph on the cube |
| **Crystal / phase** | Brand metaphor: shared hypercube lattice vs unit-model dynamics (see shelf line) |
| **HypercubeSpiking** | Broader alt name — only if the product outgrows pure LSM scope |

**Shelf line (locked for voice kit):** *Same crystal as Hopfield / CNN / ESN.
Different phase of matter.*

---

## Related HypercubeESN reading

| Document | Relevance |
|----------|-----------|
| [Reservoir.md](Reservoir.md) | Rate-based hypercube reservoir — topology twin |
| [Readout.md](Readout.md) | HCNN readout patterns usable for liquid feature fields |
| [Rotating-input-map-temporalization.md](Rotating-input-map-temporalization.md) | Static→temporal bridge (HypercubeMLP); possible future spike-encoding neighbor |
| [docs/README.md](README.md) | Documentation map |

---

## Status and non-claims

**This is a concept note.** It does not claim:

- landed code, API, or config in HypercubeESN  
- a chosen neuron model beyond “LIF-class for v1”  
- benchmark numbers vs ESN or other SNN stacks  
- that HypercubeLSM replaces HypercubeESN for rate tasks  

It **does** claim a coherent product direction:

> Put a **spiking liquid** on the **Boolean hypercube**, freeze it, train a
> readout — the event-native sibling of HypercubeESN in the HypercubeML
> family.

> **Same crystal as Hopfield / CNN / ESN. Different phase of matter.**
