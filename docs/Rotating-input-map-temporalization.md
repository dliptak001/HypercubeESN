# Rotating input-map temporalization

> Status: **design proposal** (not landed).  
> **Mechanism name:** rotating input-map temporalization (**RIMT**).  
> **Product / project name (locked):** **HypercubeMLP** — separate repo,
> built **atop** HypercubeESN (not an in-tree ESN example).  
> Role: spatial → temporal bridge so static / non-sequence (MLP-class) tasks
> run on the hypercube stack in the HypercubeML ecosystem.

This document is the detailed record of the scheme: motivation, mechanics,
assumptions, rotation semantics, warmup/readout contract, performance framing,
related literature (with citations), baselines, cost, open knobs, and how it
relates to the existing HypercubeESN drive ports.

---

## One-line definition

**Rotating input-map temporalization** encodes a static vector by repeatedly
driving a reservoir with that *same* vector under an **orbit of a frozen random
input map** (map addresses XOR’d with an incrementing pass counter), then
training a readout on the post-warmup state trajectory.

The sample does not change. The **coupling** of the sample into the reservoir
does.

### Job (performance frame)

**RIMT’s job is: turn static `x` into a rich driven trajectory so a frozen
hypercube + trained readout can separate classes. Performance tracks how good
that trajectory is as a feature map.**

Everything else in this design — pass-counter XOR, warmup, multi-phase readout,
map draw, episode length — is in service of that trajectory. Accuracy is not a
property of the rotate operator in isolation; it is a property of whether the
post-warmup orbit is a **separable lift** of the sample for the task at hand.

---

## Why this exists

### Two paradigms

| Paradigm | Data | Natural machine |
|----------|------|-----------------|
| **Temporal** | sequences, streams, dynamical systems | ESN / reservoir (HypercubeESN) |
| **Spatial / static** | fixed feature vectors, images, tables, embeddings | classical MLP-style map: `x → y` with no time axis |

HypercubeESN is built for the first column: inject a signal each step, evolve
the hypercube, train a readout on the resulting trajectory. It does **not**
natively accept a one-shot spatial sample as a first-class object.

### The bridge problem

To run MLP-class work on HypercubeESN you must turn a static sample into a
**sequence the reservoir can step on**. That conversion is not a detail — it
*is* the architectural claim.

Naive options:

1. **Single presentation** — inject `x` once, hope the state is usable. Too
   thin; no washout, no trajectory.
2. **Constant drive** — hold `x` fixed for `T` steps under a fixed `W_in`.
   Standard RC-for-static-data. Usable, but the drive is constant, so the
   state mostly settles toward a fixed-point-like response to that sample.
3. **Feature scan** — present coordinates (or patches) of `x` one after
   another. Domain-order dependent; invents a temporalization that may have
   nothing to do with the hypercube.

RIMT is a fourth option: keep `x` fixed, **XOR-reindex a frozen random input
map** with an incrementing pass counter so the *effective drive* is
time-varying. Spatial structure becomes temporal structure without a
domain-specific scan order.

---

## Core scheme

### Assumptions (canonical form)

1. **Size match.** Let `N = 2^DIM` be the number of hypercube vertices. The
   static input `x` is a real vector of length **N** (or is projected /
   padded to length N *before* this layer — how you get there is out of
   scope for this document).
2. **Frozen random map.** A static random structure `W` (the “input map”) is
   drawn once at construction and never trained — same discipline as
   hypercube recurrent / input weights in HypercubeESN.
3. **Rotation over time.** A per-episode **pass counter** `c` (unsigned integer)
   starts at 0 and increments **once every pass**. Weight indices / vertex
   addresses are read through `index XOR c` (see below). That *is* the rotate.
4. **Hold and repeat.** The same `x` is presented for every step of a
   per-sample episode: warmup, then readout collection.
5. **Reset between samples.** Each new spatial sample starts from a clean
   reservoir state (or a fixed protocol) and a fresh pass counter
   (`c = 0` unless an experiment deliberately continues it). Samples do not
   leak into each other through residual dynamics unless that is explicit.

### Per-sample episode

```
Given static sample x  (|x| = N)
Given frozen map W     # addressed by vertex / weight index

Clear reservoir state
c = 0                  # unsigned pass counter
for t = 0 .. T_total - 1:
    drive(t) = ApplyMap(W, x, c)   # every weight/address lookup uses (i XOR c)
    Inject drive into reservoir
    Step()
    c = c + 1          # one increment per pass

    if t >= T_warmup:
        record state (and/or readout features) for this sample
```

- **`T_warmup`** — washout / settle under the rotating drive. Initial
  condition and early transient are discarded.
- **`T_read`** — steps after warmup used for readout features
  (`T_total = T_warmup + T_read`, or overlapping variants — see knobs).
- **Readout engagement** — train or evaluate only on post-warmup material
  (single snapshot, multi-slice window, or average — same family of choices
  as temporal ESN work).

### What is being encoded

The code for sample `x` is **not** “one vector of activations after one
inject.” It is the **driven trajectory** of the reservoir under the forcing
built from `x` and the orbit of map alignments induced by `c`.

Different `x` produce different orbits of drives → different state
trajectories → a trainable readout can separate them. Same spirit as
reservoir computing’s usual bet: rich, fading dynamics + simple trained
head.

---

## What “rotate the input map” means

### Canonical: pass-counter XOR on weight addresses

**Rotate** means one thing in this design:

1. Keep a single frozen weight structure `W`, addressed by unsigned indices
   (vertex ids, weight-table rows — the same integer addresses the hypercube
   already uses).
2. Hold an **unsigned integer pass counter** `c`.
3. On every weight / address lookup for the input map, use **`i XOR c`**
   instead of `i`.
4. After each **pass** (one drive + `Step` of the episode), do **`c += 1`**.

No cyclic shift table, no copy of `W`, no automorphism list. One integer and
an XOR — the same primitive the reservoir already uses for neighbor addressing
(`v XOR (1 << bit)`).

Sketch (length-N diagonal map; same idea for denser tables):

```
// W[0 .. N) frozen at construction
// x[0 .. N) held fixed for the episode
// c increments once per pass

for v = 0 .. N-1:
    drive[v] = W[v XOR c] * x[v]
```

Dual form (XOR the sample index instead of the weight index) is a different
alignment; pick **weight-index XOR** as the default unless an experiment
deliberately compares:

```
drive[v] = W[v] * x[v XOR c]    # dual — not the default
```

If the map is denser (e.g. per-vertex neighbor weights `W[v][i]`), apply the
XOR to the **vertex/address axis** of the table on every lookup — still one
frozen tensor, reindexed by `c`, never rewritten.

### Why this is the right simple operator

| Point | Detail |
|-------|--------|
| **Hypercube-native** | XOR is the group operation on vertex labels; RIMT speaks the same language as neighbor gather |
| **O(1) state** | One `uint` per episode (or per running spatial stream) |
| **No materialization** | `W` is never rotated in memory; only the **lookup address** changes |
| **Full orbit in N passes** | For `N = 2^DIM`, the low `DIM` bits of `c` run through every residue class mod N. After N increments, `(i XOR c)` has walked a complete translation orbit of the address space (period N in the map alignment, barring extra symmetry in `W` or `x`) |
| **Implementation** | Mask to the address width if needed: `(i XOR c) & (N - 1)` when indices are `0 .. N-1` and `c` may grow past `N` |

When `c` exceeds `N - 1`, either let it run (mask on use) or wrap
`c = (c + 1) & (N - 1)`. Behavior of the **alignment** is period-N either way
if only the low `DIM` bits matter for indexing.

### Pass vs step

**Pass** = one presentation of the held sample under the current `c`, then one
reservoir `Step`, then `c += 1`. In the canonical episode loop, pass and
timestep are the same counter.

Do **not** increment `c` inside the per-vertex loop — once per pass, not once
per neuron.

### Period

With address-width `DIM` and mask `(N - 1)`, the map alignment is **period N**.
After warmup the reservoir is forced by a structured, reproducible orbit of
couplings whose period is tied to cube size. That should interact honestly
with leak rate, spectral radius, and `T_read` (prefer readout windows that
see more than one phase of the orbit).

### Non-canonical variants (optional only)

These are **not** the design default; listed only so they are not rediscovered
as if missing:

| Variant | Idea |
|---------|------|
| Cyclic shift | `(i + c) mod N` instead of `i XOR c` — arithmetic, not cube-native |
| Gray-code `c` | Advance `c` in Gray order so successive masks differ by one bit |
| Increment by `1 << k` | Walk a single bit plane, then the next |

First implementation and all baseline claims should use **incrementing
unsigned `c` with `i XOR c`**.

---

## Relation to HypercubeESN as it exists today

This section is a **map**, not a claim that RIMT is already implemented.

### Drive ports (landed)

HypercubeESN already has three drive concepts (see
[Reservoir.md](Reservoir.md), [full_state_linear_feedback.md](full_state_linear_feedback.md)):

| Port | Who stages it | Role |
|------|---------------|------|
| **Input** | Caller, each step | Primary external drive (`InjectInput`) |
| **External feedback** | Caller | Closed-loop / teacher / self-drive path |
| **FSF** | Internal (`φ = V · x`) | Full-state linear feedback |

RIMT lives on the **input** path (or a dedicated “spatial episode” driver that
stages the input buffer before each `Step`). It does not require FSF or
external feedback, though either could be combined later for other
architectures.

### How input is staged today

Current API: `InjectInput(channel, value)` writes a scalar onto a contiguous
block of vertices; `num_inputs` must divide `N`. Neighbor gather then mixes
that field through frozen `W_in[v][i]` on Hamming neighbors.

RIMT’s conceptual “map each coordinate of `x` to a node through a weight
vector” is **compatible in spirit** with a full-length field on the cube, but
it is **not** the same as today’s multi-channel block broadcast. A landed
RIMT layer would likely:

1. Build an length-N **drive field** from `x` and `W(t)` outside or beside
   the current channel API, then inject that field in one shot; or
2. Specialize a spatial-episode mode that owns staging for `|x| = N`.

Either way, the reservoir step itself stays the same: stage drive → `Step()`
→ read state / history slices.

### Neighbor gather vs diagonal map

HypercubeESN’s input path is a **neighbor gather**, not a pure diagonal
`drive[v] = w[v] * x[v]`. Design choice when implementing RIMT:

| Style | Meaning |
|-------|---------|
| **Diagonal spin** | Form a length-N field, then let existing `W_in` gather it (two random layers) |
| **Map-is-the-weights** | Treat the rotating structure as the only input coupling (may bypass or replace the usual `W_in` role for that product) |
| **Hybrid** | Rotating outer map + frozen hypercube input weights as today |

**HypercubeMLP** should pick one and document it; this design doc treats
“frozen map + address XOR pass-counter + hold `x`” as the RIMT invariant.

---

## Warmup and readout

### Why warmup

Under any repeated drive, the early trajectory still depends on the initial
state. Warmup is the washout: run the rotating drive long enough that the
**echo of the initial condition** is negligible relative to the forced
orbit of `x`. Only then is the state a function (approximately) of the
sample and the map orbit — which is what the readout is allowed to trust.

### What to feed the readout

After `T_warmup`, options (from thin to rich):

1. **Last state only** — one length-N vector (plus history slices if the
   host already exposes multi-slice state to the HCNN readout).
2. **Fixed window** — concatenate or multi-slice-read `T_read` post-warmup
   states.
3. **Orbit features** — sample several phases of the period-N XOR-alignment
   orbit (e.g. every `N/k` passes) so the readout sees the orbit, not one
   phase of `c`.
4. **Pool over the period** — mean / max over one full period of `c` after
   warmup (phase-insensitive summary).

RIMT **pays rent** when the readout uses more than a single arbitrary phase
of `c`: the whole point of XOR-spinning the map is a **trajectory**.
Multi-slice / multi-phase features are the natural fit with HypercubeESN’s
existing history-depth and multi-slice readout ideas.

### Training loop (sketch)

```
for each labeled sample (x, y):
    run RIMT episode (warmup + collect)
    append collected reservoir features and target y

train readout on collected set   # same HCNN / host training as temporal tasks
```

Inference: same episode protocol, no teacher signal required on the input
path unless the product adds a closed loop for other reasons.

---

## Why rotation helps (and when it might not)

### Against constant drive

Constant drive: `drive = ApplyMap(W, x)` every step with **fixed** `c` (or no
XOR). The forced system tends toward a steady response to that fixed field.
Classification can still work (different `x` → different fixed points / short
transients), but the temporal axis is underused.

XOR-rotating map: effective coupling changes every pass while `x` is fixed →
a **closed orbit of forcings** (period N in the low-bit address) → a richer
state trajectory for the same sample budget of steps.

### Against feature scan

Feature scan imposes an order on coordinates. That order is often arbitrary
for generic vectors and can smuggle in a false 1D locality. RIMT’s order is
**in the pass-counter orbit of map addresses**, not in a story about which
feature is “next.” The random frozen `W` is the inductive bias; `c` and XOR
are how you temporalize it.

### Failure modes

| Failure | Symptom / cause |
|---------|-----------------|
| **Degenerate W** | Near-constant or highly symmetric map → orbit of drives collapses |
| **Too short warmup** | Readout sees initial-state junk |
| **T_read ignores period** | Samples only one phase of the period-N `c` orbit and loses diversity |
| **Increment in the wrong place** | `c++` inside the per-vertex loop (once per neuron) instead of once per pass |
| **Forgot address mask** | `i XOR c` without `& (N-1)` when `c` or width can exceed DIM bits |
| **Leaky sample protocol** | No clear / no `c` reset between samples → train/test contamination |
| **\|x\| ≠ N without a defined projection** | Silent padding/truncation policy bugs |

---

## Cost and scaling

| Piece | Cost |
|-------|------|
| Storage of `W` | O(size of one map) — **not** O(N × period) |
| Pass counter | One unsigned integer; `c += 1` once per pass |
| Per-lookup rotate | One XOR (and optional `& (N - 1)`) on the weight address |
| Per-pass field build | O(N) if forming an explicit drive field from `W[v XOR c] * x[v]` |
| Per-pass reservoir | Same as any HypercubeESN step (neighbor gathers, history, activation) |
| Per sample | `(T_warmup + T_read)` passes |

`N` is already exponential in DIM for the reservoir. RIMT does not change
that asymptotics; it multiplies work by the episode length. Episode length
is a **product hyperparameter**, not a hidden quadratic in a materialized
weight tower.

---

## Design knobs (for implementers)

| Knob | Role | Notes |
|------|------|-------|
| `DIM` / `N` | Hypercube size; target length of `x` | `N = 2^DIM` |
| Map structure | Vector vs matrix vs neighbor-weight block | Start simple: length-N vector |
| Rotate | **`c` unsigned; lookup `i XOR c`; `c++` once per pass** | Canonical; not optional for the name RIMT |
| Address mask | `(i XOR c) & (N - 1)` | Required when `c` may leave `0 .. N-1` |
| Map draw | Distribution, scaling, seed | Match host input-scaling discipline where possible |
| `T_warmup` | Washout length (passes) | ≥ a few times characteristic memory / period fraction |
| `T_read` | Collection length (passes) | Prefer ≥ 1 full period of `c` if using orbit features |
| Readout feature recipe | Last / window / multi-phase / pool | Multi-phase is the RIMT-native choice |
| Between-sample reset | Full clear + `c = 0` vs partial | Full clear and `c = 0` are the safe default |
| Projection to N | How non-`N` data becomes length N | Out of scope here; must be explicit in a product |

---

## Naming and placement (locked)

| Term | Role | Use |
|------|------|-----|
| **HypercubeMLP** | **Product / project / repo** | Distribution name, CMake/PyPI identity, org listing, user-facing README |
| **Rotating input-map temporalization** | **Mechanism** (full) | Papers, design docs, theory sections |
| **RIMT** | **Mechanism** (short) | Code namespaces, APIs, cross-links, implementer prose |
| **Spin drive** | Informal only | Not canonical — avoid in public API names |

**Locked split:** product name ≠ mechanism name.

- **HypercubeMLP** = shelf name: static / non-sequence learning on the Hypercube
  stack (familiar “MLP-class tasks,” not a classical backprop dense net).
- **RIMT** = how HypercubeMLP temporalizes a fixed vector into a driven
  reservoir trajectory.

**One-liner (product face):**

> **HypercubeMLP** — static / non-sequence learning on the Hypercube stack:
> RIMT turns a fixed vector into a driven reservoir trajectory; HypercubeESN
> + readout do the rest.

**Placement (locked intent):**

| Item | Where |
|------|--------|
| HypercubeMLP implementation, packaging, end-user API | **Own project** (depends on HypercubeESN; does not live as `examples/` in ESN) |
| RIMT design record (this file) | HypercubeESN `docs/` for now; re-home or dual-link into HypercubeMLP when that repo exists |
| Dependency direction | HypercubeMLP → HypercubeESN (never the reverse) |

Honest README constraint for HypercubeMLP: first screen must say this is
**not** a classical multilayer perceptron trainer — it is **MLP-class tasks**
(`static x → y`) via RIMT + frozen hypercube + trained readout.

---

## Performance expectations (design-time)

No RIMT benchmarks exist yet. Framing follows the job line above: judge the
**driven trajectory as a feature map**, not the XOR operator in isolation.

### Why the trajectory can be a good feature map

- **Static-pattern RC is already a working practice** — hold a sample, run the
  reservoir, washout, train a readout [1][2][3][4]. RIMT inherits that pipeline.
- **Time-varying frozen input coupling is proven** in delay-based / masked RC:
  a frozen random mask temporalizes how the input hits a dynamical node
  [5][6][7]. RIMT is the hypercube-native cousin: orbit of map *addresses*, not
  virtual-node expansion of a scalar stream.
- **Multi-phase features harvest the orbit.** Last-state-only often discards
  the reason the map was spun; multi-slice / multi-phase readout is where RIMT
  should separate from constant drive.

### Why the trajectory can fail as a feature map

- Rotation does not invent information in `x` — it only spreads a fixed random
  coupling through nonlinear dynamics.
- Period-N forcing can lock into useful sample-specific cycles or into
  redundant phases if leak / spectral radius / drive scale are wrong.
- Projection of real data onto length `N` often dominates accuracy; RIMT is
  not a substitute for a spatial front-end.
- Readout capacity caps the stack at RC-class static performance unless the
  head is deepened for a product reason.

### Expected standing (qualitative)

| Baseline | RIMT likely… |
|----------|----------------|
| Linear model on raw `x` | Better (nonlinear driven lift) |
| Extreme learning machine / one-shot random projection [8] | Competitive; win when multi-step mixing helps |
| Constant-drive static RC [1][2][3] | Slightly better to clearly better **if** orbit features are used; flat if last-state only |
| Feature-scan / raster into RC | Similar or better on generic vectors; scan may win when true spatial order exists |
| Small trained MLP / CNN | Often worse on hard spatial tasks; competitive when frozen-core RC economics are the product constraint |
| Delay-mask photonic-style RC [5][6] | Different architecture; shared spirit, not a transferred accuracy claim |

**Bottom line:** RIMT should behave like a solid static-pattern reservoir method
with a better temporalization than constant drive — not like a foundation model.
Performance tracks trajectory quality; measure that with a controlled bake-off
(constant drive vs RIMT; last-state vs multi-phase; same `N` and readout).

---

## Related literature

RIMT is a **composition** of known pieces, not a rediscovery of one classical
named algorithm. Full package — hold length-`N` `x`, reindex a frozen map by
incrementing `c` and `i XOR c`, warmup, readout on a size-matched hypercube —
was **not** found as a standard named technique. Prior art falls into clear
bins.

### Field context

| | Reference |
|--|-----------|
| [9] | H. Jaeger, *The “echo state” approach to analysing and training recurrent neural networks*, GMD Report 148, 2001. Foundational ESN; washout, echo state property, frozen reservoir + trained readout. |
| [10] | M. Lukoševičius and H. Jaeger, *Reservoir computing approaches to recurrent neural network training*, Computer Science Review 3(3):127–149, 2009. Survey; situates static and temporal uses of RC. |
| [11] | W. Maass, T. Natschläger, and H. Markram, *Real-time computing without stable states: a new framework for neural computation based on perturbations*, Neural Computation 14(11):2531–2560, 2002. Liquid state machines; parallel RC lineage. |

### Static patterns into reservoirs (constant-drive family)

Closest **problem** prior art: use RC when the sample has no intrinsic time
axis. Typical recipe is hold input (or present once and let dynamics settle),
discard washout, train readout — **without** an orbit of input-map alignments.

| | Reference |
|--|-----------|
| [1] | M. J. Embrechts, L. A. Alexandre, and J. D. Linton, *Reservoir computing for static pattern recognition*, ESANN 2009. Introduces RC for static classification; reservoir outputs after dynamics stabilize under static input. |
| [2] | L. A. Alexandre, M. J. Embrechts, and J. D. Linton, *Reservoir size, spectral radius and connectivity in static classification problems*, ICANN 2009 (Springer LNCS). Hyperparameters for static RC classification. |
| [3] | A. J. Wootton, *Optimising Echo State Networks for Static Pattern Recognition*, PhD thesis, Keele University (and related papers). ESNs tuned for static pattern tasks; comparisons to SVM / ELM-style baselines. |
| [4] | C. Emmerich, R. F. Reinhart, and J. J. Steil, *Recurrence enhances the spatial encoding of static inputs in reservoir networks*, Proc. ICANN / related 2010. How recurrence helps encode static spatial inputs. |

**Relation to RIMT:** same job framing (static sample → reservoir features →
readout). RIMT differs by **time-varying map alignment** so the feature map is
a **driven orbit**, not only a fixed-point-like response to constant drive.

### Time-varying frozen input maps (mask / delay-RC family)

Closest **mechanism** prior art: a frozen random structure that changes how
input couples **over time**. Standard in delay-based and photonic reservoir
computing as an **input mask**.

| | Reference |
|--|-----------|
| [5] | L. Appeltant et al., *Information processing using a single dynamical node as complex system*, Nature Communications 2:468, 2011. Single nonlinear node + delayed feedback; input **time-multiplexed** with a random mask into virtual nodes. |
| [6] | L. Appeltant et al., *Constructing optimized binary masks for reservoir computing with delay systems*, Scientific Reports 4:3629, 2014. Design of binary input masks for delay RC. |
| [7] | F. Duport et al., *Analog input layer for optical reservoir computers*, arXiv:1406.3238, 2014 (and related photonic mask literature). Input mask as periodic random function in optical RC. |

**Relation to RIMT:** shared spirit — frozen random **time-varying input
coupling**. Mask literature usually expands a scalar (or low-dim) stream into
virtual time; RIMT holds a full length-`N` vector and **XOR-translates map
addresses** on a hypercube. Same economics (draw once, reindex forever);
different geometric object.

### One-shot random maps (contrast, not trajectory)

| | Reference |
|--|-----------|
| [8] | G.-B. Huang, Q.-Y. Zhu, and C.-K. Siew, *Extreme learning machine: theory and applications*, Neurocomputing 70(1–3):489–501, 2006. Frozen random hidden layer + trained output; static `x`, **no** multi-pass driven orbit. |

**Relation to RIMT:** same “freeze random map, train head” discipline. ELM is
a **single-shot** feature map; RIMT’s performance claim is specifically about
the **trajectory** under a changing alignment.

### False friends (related keywords, different claim)

| Topic | Why it appears in searches | Why it is not RIMT |
|-------|----------------------------|--------------------|
| Circular / cycle / simple-cycle ESNs | “Circular,” ring topologies | Structure of **recurrent** `W`, not rotating input-map addresses |
| Circulant reservoir matrices | Memory-capacity and structured RC | Again reservoir topology / spectrum, not input-map spin |
| Temporal XOR / parity benchmarks | “XOR” in RC papers | Task on input **bits**, not XOR of weight **indices** |
| Raster / row-scan of images into RC | Spatial → temporal by feature order | Temporalizes **coordinates**, not a frozen map orbit |

### How to cite RIMT relative to this map

Suggested positioning for papers and product docs:

> **HypercubeMLP** provides MLP-class (static / non-sequence) learning on the
> Hypercube stack. Its temporalization engine is **RIMT**, not claimed as a
> rediscovery of a single classical algorithm. RIMT sits between
> **static-pattern reservoir computing** [1][2][3][4] and **masked
> time-multiplexed RC** [5][6][7], specialized to a size-matched hypercube via
> **pass-counter XOR address translation** of a frozen input map. RIMT’s job
> is to turn static `x` into a rich driven trajectory so a frozen hypercube +
> trained readout can separate classes; performance tracks how good that
> trajectory is as a feature map.

---

## Related HypercubeESN reading

| Document | Relevance |
|----------|-----------|
| [Reservoir.md](Reservoir.md) | Topology, timestep, input gather, history depth |
| [Readout.md](Readout.md) | HCNN readout; multi-slice features |
| [full_state_linear_feedback.md](full_state_linear_feedback.md) | Separate drive port (not RIMT) |
| [reservoir_feedback_mechanism.md](reservoir_feedback_mechanism.md) | External feedback port |
| [docs/README.md](README.md) | Documentation map |

---

## Status and non-claims

**This is a design proposal.** It does not claim:

- a landed API or config flag in HypercubeESN
- a benchmark result against constant-drive or MLP baselines
- that integer-increment `c` is optimal among ways to advance the XOR mask
  (e.g. Gray code) — only that it is the **canonical simple** choice
- a specific projection from arbitrary spatial objects onto length-N vectors
- identity with input-mask delay RC [5][6] or with constant-drive static RC [1]

It **does** claim a coherent, implementable temporalization:

> Hold the spatial sample fixed; drive the hypercube under a frozen random
> input map whose addresses are XOR’d with an incrementing pass counter;
> warmup; read the trajectory.

**RIMT’s job is: turn static `x` into a rich driven trajectory so a frozen
hypercube + trained readout can separate classes. Performance tracks how good
that trajectory is as a feature map.**

That is the hinge between HypercubeESN’s temporal engine and **HypercubeMLP**
in the HypercubeML ecosystem.
