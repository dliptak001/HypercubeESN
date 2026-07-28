# HypercubeLSM — concept

> Status: **design concept** (not landed; not an in-tree HypercubeESN feature).  
> **Product / project name (working):** **HypercubeLSM** — its own project in
> the HypercubeML ecosystem, **sibling** to HypercubeESN (not an example folder
> inside ESN).  
> **Paradigm:** liquid state machine (LSM) — spiking reservoir computing on the
> Boolean hypercube.  
> This file lives under HypercubeESN `docs/` as a **concept record** so the
> family story stays next to the rate-based reservoir docs. Implementation,
> packaging, and API belong in a future HypercubeLSM repo.

---

## One-line definition

**HypercubeLSM** is a liquid state machine whose recurrent **liquid** sits on
the vertices of a Boolean hypercube: spikes travel by XOR-addressed
Hamming-distance-1 edges, the liquid is frozen after construction, and only a
readout is trained.

Same topology contract as HypercubeESN; **spikes and timing** instead of
smooth rate units. Not “ESN with a hard threshold” — a sibling product with its
own unit model, time model, and construction culture.

### Shelf line

> **Same crystal as Hopfield / CNN / ESN. Different phase of matter.**

One shared Boolean-hypercube geometry (the **crystal**); unit model and
dynamics differ (the **phase**) — here, spikes instead of rate fields or energy
memory. Accurate enough for branding; not a formal physics claim. Always pair
with the training contract: **freeze the liquid; train the readout.**

**Suggested pairing:**

> **HypercubeLSM** — a spiking liquid on the Boolean hypercube.  
> *Same crystal as Hopfield / CNN / ESN. Different phase of matter.*  
> Freeze the liquid; train the readout.

### What must be true for the product to matter

Wiring LIF units on XOR edges is easy. The product earns its name only if three
things land together:

1. **A reproducible liquid-regime protocol** at construction (the LSM analogue of
   ESN spectral-radius rescale) — not ad-hoc gain fiddling per demo.
2. **Spike-native features** on the cube (preferably smooth multi-τ traces the
   HCNN can read), not only dense membrane dumps.
3. **Demos that need timing or sparsity** — not day-one NARMA parity with the
   rate ESN (fairness optional; narrative clarity required).

Without those, HypercubeLSM collapses to a thinner rebrand of ESN with a hard
threshold. With them, geometry + spikes + frozen liquid is a real reason to
exist.

---

## What LSM means

**LSM** = **liquid state machine** (Maass, Natschläger, Markram and the broader
spiking-RC literature).

| Piece | Role |
|-------|------|
| **Liquid** | Recurrent network of **spiking** neurons; weights (and optional delays) fixed after construction |
| **Input** | Spike trains, or encodings that become spikes |
| **State** | High-dimensional snapshot of recent liquid activity (counts, traces, voltages, …) |
| **Readout** | Trained map from liquid features → labels or continuous targets |

It is the **spiking cousin** of echo state networks / reservoir computing: rich
fixed dynamics do the hard temporal lift; learning stays in a thin head.

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

HypercubeESN already showed that the Boolean hypercube is a strong **implicit
graph** for reservoirs:

- **N = 2<sup>dim</sup>** neurons on vertices
- Neighbors = single bit flips: `v XOR (1 << i)` for `i = 0 .. dim-1`
- **No adjacency list** — topology is computed, not stored
- Local fan-in = **dim** (exactly), vertex-transitive, multi-scale bit structure

HypercubeLSM reuses that geometry for **events**:

| Property | LSM payoff |
|----------|------------|
| XOR routing | Spike delivery is an address op — hardware-friendly story |
| Fixed fan-in dim | Bounded local work per event / per clocked step |
| Hamming locality | Nearby codes interact first; bit hierarchy = multi-scale pathways |
| Same N as ESN / CNN / Hopfield | Shared mental model; optional shared readout geometry (HCNN on vertex fields) |

The product claim is not “yet another SNN framework.” It is:

> **This liquid is a hypercube** — same lattice as Hopfield / CNN / ESN; unit
> model = spikes.

Or, shorter: *Same crystal. Different phase of matter.*

### What does **not** automatically transfer from HypercubeESN

| ESN idea | LSM stance |
|----------|------------|
| Spectral radius rescale of a linear companion | Different stability culture — need an explicit **regime protocol** (below) |
| Addressable delay line M (history depth) | v1 memory is **τ-based** (membrane / synaptic); axonal delays are a later analogue of M, not a free port of the ring buffer |
| Soft modularity essay (random weights → multi-scale soft cuts) | Rate-graph story; re-validate under E/I spikes or leave as research — do not copy unchanged |
| Smooth tanh field → HCNN | HCNN needs a **smooth vertex field**; raw spike rasters may prefer linear readout until traces exist |

---

## HypercubeESN vs HypercubeLSM

| | **HypercubeESN** | **HypercubeLSM** |
|--|------------------|------------------|
| Unit | Continuous / rate (leaky tanh field) | Spike times / binary events (+ traces) |
| State | Dense float vector on vertices (+ history slices) | Sparse events; features = counts, multi-τ traces, optional membranes |
| Drive | `InjectInput` / staged fields each step | Spike trains into vertices (or encoded from analog streams) |
| Recurrence | Weighted neighbor gather on published slices | Synaptic events along XOR edges (delays optional, not v1) |
| Construction “hit target” | Secant scale recurrent block → spectral radius | Scale E/I weights toward a **measurable regime** (mean rate / branching under probe drive) |
| Memory knobs | Spectral radius, leak, history depth M | τ_m, synaptic τ, refractory; delays later |
| Strength | Smooth dynamics, easy numerics, mature host | Timing codes, sparsity; neuromorphic narrative **after** event runtime |
| Training contract | Freeze reservoir, train readout | **Same** — freeze liquid, train readout |

**Sibling products**, not a mode flag inside ESN. Dependency direction when
built: HypercubeLSM may depend on shared readout libraries (e.g. HypercubeCNN);
it must **not** force spiking into the ESN core.

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

- One spiking unit per hypercube vertex `v` in `{0, …, N−1}`.
- Recurrent synapses only along Hamming edges `(v, v XOR (1<<i))`.
- Weights drawn once, **frozen** (optional axonal delays: v2+).
- Minimal v1 neuron: **LIF** (leaky integrate-and-fire) — not multi-compartment.

### Time — lock for v1

| Style | Idea | Tradeoff |
|-------|------|----------|
| **Clocked** (v1) | Fixed `dt`; update membranes, emit spikes each step | Easier tooling; closer to ESN mental model |
| **Event-driven** (later) | Priority queue of spike events | Closer to pure neuromorphic story; harder host integration |

**v1 decision:** clocked LIF liquid. Prove topology + regime protocol + readout
path before building a full event runtime.

Honesty rule for messaging: clocked LIF on N vertices is **closer to a discrete
simulator** than to Loihi-class event machines. Keep the shelf line; the README
should say **clocked simulation first; event runtime is a later phase** so the
energy narrative does not outrun the code.

### Construction regime (the LSM “spectral radius”)

ESN has a usable first knob: rescale the recurrent block so estimated ρ ≈
target. LSM stability is mean rate, E/I ratio, synaptic gain, τ, refractory —
a different tuning culture. Without a construction contract, every demo becomes
manual firefighting.

**v1 intent (to be made precise in the LSM repo):**

1. Draw signed weights on XOR edges (excitatory / inhibitory balance as knobs).
2. Fan-in normalize by degree (dim) the way ESN normalizes drive ports.
3. Apply a **global scale** (and/or E/I ratio) so that under a fixed **probe
   drive** the liquid sits in a measurable band — e.g. target mean firing rate,
   or a crude branching-ratio proxy.
4. Record seed + protocol so liquids are reproducible across machines.

Exact metrics and tolerances are open (see [Open questions](#open-questions));
the non-negotiable is that **some** hit-target protocol exists before demos
claim “surveyed liquids.”

### Memory story

| Phase | Source of temporal depth |
|-------|---------------------------|
| **v1** | Membrane τ, synaptic / firing-trace τ, refractory period — fading memory by construction of the neuron model |
| **v2+** | Optional **axonal delays** on cube edges (possibly axis-keyed) — geometric analogue of ESN history depth, costly but natural |

Do not port ESN’s M-deep ring buffer as the identity of HypercubeLSM. Addressable
lags may return as delays; until then, sell τ honestly.

### Input encoding

External signals become spikes, for example:

- rate / Poisson encoding of continuous channels
- latency coding
- direct event sensors (DVS-style) when available

**Demo strategy:** for continuous benchmarks (NARMA-class, Lorenz-class),
Poisson/rate encoding can erase the timing advantage and make LSM look like a
worse ESN. Prefer tasks where spikes are native or timing-shaped first
(temporal spike-pattern classification, classic spoken-digit-style protocols,
synthetic temporal codes). Shared-family comparisons with HypercubeESN are
welcome later — **narrative clarity over forced parity**.

Static vectors (HypercubeMLP-class tasks) need an explicit spike encoding or a
bridge (including possible future RIMT-like temporalization then spike) — out of
scope until both products exist.

### Readout features

The liquid must expose a **fixed-size feature field** the head can train on,
aligned with vertex geometry:

| Feature recipe | Notes | v1 priority |
|----------------|-------|-------------|
| **Multi-τ exponential firing traces** per vertex | Smooth field; natural multi-channel cube for HCNN | **Preferred product face** |
| Spike counts in a window per vertex | Simple, robust; often linear-readout territory | Baseline / diagnostic |
| Membrane voltage snapshot | Dense; less spike-native | Fallback |
| Multi-window stacks | Richer; higher dim | Later |

**Training contract (classic LSM):** freeze liquid; train linear readout first
for proof; HypercubeCNN when vertex fields are smooth enough (multi-τ traces).

Feature tensor for HCNN (intent): **channels × vertices** with capacity N =
2<sup>dim</sup> (and optional channel count = number of τ’s) — same full-capacity culture
as ESN’s readout input.

### Lifecycle (API-shaped sketch)

Mirror HypercubeESN language where it helps; different types underneath.

```
LiquidConfig / Create(cfg)     // draw weights, hit regime protocol, rest state
InjectSpikes(...)              // stage events for this dt (or this window)
Step()                         // clocked LIF update; clear or age stages
// optional: Features() / CopyTraces()  — vertex field for the head

// Batch-style episode
Clear()
for t in stream:
    InjectSpikes(...)
    Step()
    if t >= T_warmup: record features
Train readout on recorded features → targets
Predict / score
```

No learning inside the liquid in v1. Closed-loop feedback (if any) is
**caller-owned** — same philosophy as ESN external feedback: an input path, not
a second path to y.

### Per-sample / streaming episode

```
Build frozen hypercube liquid (dim, seed, weight scales, τ, refractory, regime …)

For each trial or stream segment:
    Clear liquid state (or defined rest protocol)
    for t in time:
        inject input spikes for this dt (if any)
        liquid step
        if t >= T_warmup:
            record readout features
    train / evaluate readout on recorded features → targets
```

---

## Design knobs (v1-oriented)

| Knob | Role |
|------|------|
| `dim` / N | Cube size; liquid size |
| Neuron model | LIF: τ_m, threshold, reset, refractory |
| Synaptic weights | Draw + scale on XOR edges; E/I balance |
| Regime protocol | Target mean rate / branching under probe drive |
| Synaptic / firing filter | Exponential traces (preferred for HCNN) |
| Delays | **None in v1**; optional later |
| `dt` | Clocked integration step |
| Input encoding | How continuous or symbolic input becomes spikes |
| `T_warmup` / collection window | Washout and feature horizon |
| Trace τ stack | One or more exponential timescales → feature channels |
| Readout | Linear first; HCNN when fields are smooth |
| Seed | Reproducible liquid draw (named substreams, ESN-style) |

**Non-goals for v1:** full STDP library, multi-chip runtime, surrogate-gradient
end-to-end training of the liquid, generic SNN framework APIs, silent mode
inside HypercubeESN `Reservoir`.

---

## What to lock before writing a core

1. **Clocked LIF**, dim-neighbor synapses only, **no delays**.
2. **Init / regime protocol** — draw E/I weights, scale to a measurable
   operating point under a fixed probe; document seed + metrics.
3. **State for readout** — multi-τ exponential firing traces per vertex
   (channels × N); counts as baseline.
4. **First demo** — temporal spike-pattern classification (not NARMA day one).
5. **API shape** — `LiquidConfig` / `Create` / `InjectSpikes` / `Step` /
   features / freeze / train readout (ESN lifecycle language, different unit).
6. **Repo boundary** — own project; optional HypercubeCNN dependency; no
   spiking path forced into HypercubeESN.

---

## Why it might matter (niche, not mass-market)

| Niche | Why LSM-on-cube |
|-------|-----------------|
| Structured neuromorphic R&D | XOR routing + fixed fan-in; event runtime later |
| Timing-sensitive tasks | Spike times carry information rate units smear |
| Sparse / edge narrative | Low mean rates → sparse events (honest only when rates are low) |
| Ecosystem completeness | Spiking twin of HypercubeESN on the same lattice |
| Shared readout culture | Same “freeze dynamics, train head” economics; HCNN when traces are smooth |

**Not** a claim to replace general deep SNN toolkits or rate ESNs on every
benchmark. HypercubeLSM wins when **geometry + spikes + frozen liquid** is the
product reason — and when demos show it.

---

## Relation to HypercubeESN (this repo)

| | |
|--|--|
| This document | Concept only, under `docs/` for family continuity |
| HypercubeESN code | Remains **rate-based** reservoir host |
| Future HypercubeLSM | Own project; may cite ESN docs for topology intuition and readout patterns |
| Shared pieces (optional later) | HypercubeCNN as readout; common dim/N conventions; org-level branding |

Do **not** implement LSM as a silent mode inside `Reservoir` without an
explicit product decision — unit model, time, and cost models diverge enough
to deserve a separate core.

---

## Minimal success criteria (when built)

1. XOR-wired LIF liquid, dim neighbors only; **documented regime protocol** at
   construction.
2. Inject spike trains, run for T steps, collect **per-vertex multi-τ traces**
   (counts acceptable as baseline).
3. Train a readout (linear for first proof; HCNN when traces are smooth).
4. At least one **temporal spike-pattern / timing-shaped** classification demo.
5. Short written comparison note vs HypercubeESN on a shared task family
   (fairness optional; narrative clarity required) — **not** the only demo.
6. README states: sibling of HypercubeESN; spikes not rates; freeze liquid;
   clocked first; event runtime later.

---

## Open questions

**Locked for v1 (intent):** clocked time base; no axonal delays; linear readout
acceptable first; preferred features = multi-τ traces; first demos timing-shaped.

**Still open:**

- Exact LIF / synapse equations and discrete-time form
- Concrete regime metric (mean rate band? branching ratio? other?) and
  tolerances
- Trace τ values and channel count default
- Whether delays ship in v2 with axis-keyed structure
- Feature tensor layout details for HypercubeCNN (channels × vertices)
- Package name lock: **HypercubeLSM** vs broader HypercubeSpiking if scope grows
- How static data (HypercubeMLP) encodes into spikes if both products coexist
- Whether soft modularity under random E/I weights is empirically real on the
  cube (research; not a v1 claim)

---

## Naming

| Term | Use |
|------|-----|
| **HypercubeLSM** | Working product / project name (preferred: paradigm-familiar) |
| **Liquid state machine** | Full paradigm name in docs |
| **LSM** | Short handle |
| **Liquid** | The frozen spiking recurrent graph on the cube |
| **Crystal / phase** | Brand metaphor: shared hypercube lattice vs unit-model dynamics |
| **Regime protocol** | Construction-time hit-target for liquid operating point |
| **HypercubeSpiking** | Broader alt name — only if the product outgrows pure LSM scope |

**Shelf line (locked for voice kit):** *Same crystal as Hopfield / CNN / ESN.
Different phase of matter.*

---

## Related HypercubeESN reading

| Document | Relevance |
|----------|-----------|
| [HypercubeLSM_primer.md](HypercubeLSM_primer.md) | **Primer** — clocked LIF equations + multi-τ features in ESN vocabulary |
| [Reservoir.md](Reservoir.md) | Rate-based hypercube reservoir — topology twin (not unit-model twin) |
| [Readout.md](Readout.md) | HCNN readout patterns usable for liquid feature fields |
| [CPP_SDK.md](CPP_SDK.md) | ESN lifecycle language to mirror, not copy |
| [Rotating-input-map-temporalization.md](Rotating-input-map-temporalization.md) | Static→temporal bridge (HypercubeMLP); possible future spike-encoding neighbor |
| [docs/README.md](README.md) | Documentation map |

---

## Status and non-claims

**This is a concept note.** It does not claim:

- landed code, API, or config in HypercubeESN
- final neuron equations or a finished regime metric
- benchmark numbers vs ESN or other SNN stacks
- that HypercubeLSM replaces HypercubeESN for rate tasks
- event-driven energy wins while the core is still clocked

It **does** claim a coherent product direction:

> Put a **spiking liquid** on the **Boolean hypercube**, freeze it, train a
> readout — the event-native sibling of HypercubeESN in the HypercubeML
> family. Prove **regime + spike-native features + timing-shaped demos** before
> claiming more.

> **Same crystal as Hopfield / CNN / ESN. Different phase of matter.**
