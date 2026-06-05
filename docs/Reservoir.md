# Reservoir — The Hypercube Echo-State Network

The reservoir is the engine of the whole system: a fixed recurrent network that
turns an input stream into a high-dimensional state the readout can learn from.
What makes this one unusual is the shape it takes. Its neurons sit on the vertices
of a Boolean hypercube, wired to their single-bit-flip neighbors by XOR — a
topology that is never stored, only computed.

That one decision ripples outward into three properties worth the read:

- **A topology you don't store.** Connectivity is implicit in the vertex
  indices — no adjacency list, at any size.
- **Hidden multi-scale structure.** Full neighbor connectivity with random
  weights turns the cube into a continuum of nested clusters — local, regional,
  and global at once — that nobody designed in.
- **Memory you can address.** Each vertex carries a delay line of its own recent
  past, so the reservoir remembers *specific* lags by construction, not by lucky
  echoes.

The rest of this document builds these from the ground up: the graph and its
wiring, the structure that emerges from it, the per-vertex memory — then the
mechanics of a timestep, input, spectral radius, and cost.

## The reservoir, briefly

In reservoir computing, a recurrent network is used as a fixed nonlinear
dynamical system: its internal weights are set once and frozen, and only the
readout is trained. The reservoir's job is to lift a low-dimensional input into a
high-dimensional state that nonlinearly encodes the input's recent history; the
readout — here a learned convolutional network on the hypercube
([Readout.md](Readout.md)), not the traditional ridge regression — reads the
features it needs back out.

A reservoir is only as good as two properties:

1. **Rich dynamics** — different inputs trace distinguishably different state
   trajectories.
2. **Fading memory** — the state reflects recent history, and lets both the
   distant past and the initial condition wash out.

Everything that follows serves those two.

## A topology you don't store

Most echo-state networks wire their neurons at random — generate a sparse graph,
store it, and trust it. HypercubeESN doesn't store a graph at all. Its N = 2^DIM
neurons sit on the vertices of a **Boolean hypercube**: each vertex is addressed
by a DIM-bit binary index, and two vertices are neighbors exactly when their
indices differ in a single bit.

That definition *is* the whole graph. A neighbor along bit `i` is just
`v XOR (1 << i)` — one instruction, computed on demand. No adjacency list is
built, stored, or serialized; the entire connectivity lives in the binary
representation of the vertex addresses. Double the neurons and the "wiring
diagram" still costs nothing — because there is no wiring diagram, only
arithmetic.

| DIM | N (neurons) | Connections/neuron | Recurrent edges/slice (N·DIM) |
|-----|-------------|--------------------|-------------------------------|
| 5   | 32          | 5                  | 160                           |
| 6   | 64          | 6                  | 384                           |
| 7   | 128         | 7                  | 896                           |
| 8   | 256         | 8                  | 2,048                         |
| 9   | 512         | 9                  | 4,608                         |
| 10  | 1024        | 10                 | 10,240                        |

The last column is the directed-edge count of a *single* recurrent slice — pure
topology (N vertices × DIM neighbors). The number of stored *weights* is larger
and grows with `history_depth` M: each of the M slices gets its own N·DIM
recurrent block plus one N·DIM input block, for a total of **N · DIM · (M+1)**
(e.g. DIM 10, M 16 → 10,240 × 17 ≈ 174K). See
[Deep vertices](#deep-vertices-memory-you-can-address).

DIM is constrained to **[5, 16]** — 32 to 65,536 neurons, the practical range for
reservoir computing.

### Every single-bit-flip neighbor

Each neuron receives from *all* DIM of its Hamming-distance-1 neighbors — every
vertex one bit-flip away, nothing further:

```
Mask for neighbor i: 1 << i, for i = 0 to DIM - 1
  i=0: mask=1            (flip bit 0)
  i=1: mask=2            (flip bit 1)
  i=2: mask=4            (flip bit 2)
  ...
  i=DIM-1: mask=1<<(DIM-1) (flip the top bit)
```

The masks are computed inline from the loop index — a single XOR per neighbor
lookup, no adjacency storage. Using the *full* DIM neighbors rather than a
truncated subset is not an arbitrary choice: it's what gives the next section its
structure. Hold onto that — it's where the hypercube stops being a tidy diagram
and starts doing something surprising.

## The hidden structure: multi-scale soft modularity

Here is the surprising part the last section promised. Full neighbor
connectivity is geometrically uniform — every vertex wired identically to its DIM
neighbors — yet the *weighted* graph is anything but. Random weights, read the
right way, hide a whole hierarchy of structure inside the plain cube.

Start with a single edge. Recurrent weights are drawn from a symmetric uniform
distribution, so plenty of them land near zero. A near-zero `weight[v][i]` is a
**soft cut** of the bit-`i` edge into vertex `v` — signal crossing it is
attenuated to noise. These soft cuts happen independently per `(vertex, bit)`, so
they line up along no global axis at all.

Now sweep a threshold. Pick any cutoff on weight magnitude and keep only the edges
above it: the surviving subgraph's connected components partition the cube into
clusters. These are *not* neat geometric sub-hypercubes — this is **bond
percolation on the cube**, with a critical threshold near `1/DIM`. Slide the
cutoff and the graph slides with it, smoothly, between a dust of fragments and a
single giant component.

The real payoff is that **every vertex lives in all of these partitions at once**.
Its DIM incoming weights span every bit direction with different magnitudes, so:

- at a **strict** cutoff (only the strongest weights), the vertex sits in a small,
  tight cluster of its most-connected neighbors — *local* structure;
- as the cutoff **loosens**, that cluster merges outward through whichever
  directions carry middling weights — *regional* structure;
- at **zero** cutoff, everything reconnects into the full hypercube — *global*
  structure.

There is no single "correct" partition — only a continuum of nested clusterings,
local at strong weights, regional at moderate, global at weak. The reservoir is
compartmentalized and fully connected at the same time, depending on which scale
you read it.

Contrast the obvious alternative: wire only DIM−1 of the bit directions
globally — say, omit bit 0 at every vertex. That yields a *deterministic, globally
disjoint* split: two exact (DIM−1)-subcubes with no edges between them, no mixing,
ever. Information injected in one half can never reach the other. Full-DIM
connectivity with random weights is qualitatively different — it buys both the
**compartmentalization** of random clusters (local regions free to specialize
their dynamics) and the **scale-spanning overlap** that lets information cross
between them through whichever directions carry strong weights.

This multi-scale soft modularity is plausibly a real part of why full-DIM
connectivity beats dimensionally-truncated variants — and the claim is testable: a
DIM−1-truncated reservoir, matched on every other hyperparameter, should fail on
any task whose input and target require crossing the omitted axis.

## Deep vertices: memory you can address

A standard echo-state network keeps exactly one step of feedback: each neuron
reads its neighbors' *last* output and nothing older. Anything the reservoir
"remembers" further back has to survive by recirculating through the recurrent
loop — an exponentially-fading echo with no addressable past. The `history_depth`
extension (codename *deep vertices*) replaces that echo with an explicit,
addressable **delay line**: the reservoir retains the M most-recent output slices,
and every vertex update sums over both its DIM spatial neighbors *and* those M
temporal slices.

### From a vector of neighbors to a kernel over space × time

At M = 1, each vertex applies a length-DIM weight vector to its neighbors — a
purely spatial filter. At depth M it applies an **M × DIM weight bank** instead:
one independent weight per `(slice, axis)` pair. Read as a small convolution,
every vertex becomes a fixed **spatiotemporal FIR kernel**, sweeping the cube's
local neighborhood across the last M steps. The recurrent weight count grows from
N·DIM to N·DIM·M, initialized at scale `1/√(DIM·M)` so the per-vertex drive
variance is independent of both DIM and M *before* the spectral-radius rescale.

### Depth resolves; leak blurs

This is a different lever from the leaky integrator, and the contrast is the whole
point. `leak_rate` smears the past into the present with a single exponential
time-constant — it **blurs**. The delay line exposes the past as M distinct,
separately-weighted taps the readout can address one by one — it **resolves**. The
two compose: leak sets how fast a single slice forgets; depth sets how many
cleanly-separated slices exist to be read at all. A target that depends on a
*specific* lag — not a smeared average of recent input — is exactly the kind a
delay line can serve and a leaky integrator cannot.

Stacking M slices also changes what "spectral radius" means: the reservoir's true
state becomes the MN-dimensional stack of all slices, and the recurrence turns
into a companion-form operator over that stack. That story — and how
`Initialize()` hits a target radius for M > 1 — lives in
[Spectral radius](#spectral-radius-tuning-the-edge-of-chaos).

### One ring, one memcpy, and a clean M = 1 fallback

The M slices live in a single `M·N` buffer addressed through M rotating pointers,
so aging the history costs one `memcpy` per step regardless of M (the mechanics
are in [Anatomy of a timestep](#anatomy-of-a-timestep)). `history_depth` may be
any integer in **[1, 64]** — it need not be a power of two — and at M = 1 the ring
degenerates to a single slot: the rotation is a no-op, the weight layout and RNG
draws match the legacy single-step reservoir, and dynamics are preserved
**bit-for-bit**.

One honest caveat: depth is **not** a hypercube trick. Adding a delay line is a
generic recurrent move; nothing about it exploits XOR addressing. Its value is
*orthogonal* to the topology — depth adds representational reach at each vertex
(its own multi-step state and kernel), while the topology work above sharpens the
cube's spatial graph. The two compose cleanly because the kernel is still
*labelled by axis*: any hypercube-native idea — per-axis weight sharing,
Walsh-band analysis, time-gated axes — lifts straight into the `M × DIM` weight
bank.

## Anatomy of a timestep

With the concepts in hand, here is the machinery. Each vertex carries three
per-step quantities:

- `vtx_state_[v]` — scratch for *this* step's newly computed state, written by
  `UpdateState` and published at the end of `Step()`.
- `vtx_output_history_` — the ring of the M = `history_depth` most-recent published
  output slices (M = 16 by default), addressed through M rotating `slice_ptrs_`.
  Slice 0 is the latest output — what neighbors read and what `Outputs()` returns;
  slice `j` is the output from `j` steps ago.
- `vtx_input_[v]` — the per-step injected field set by `InjectInput()`. It is
  summed through its **own** input-weight block (decoupled from the recurrent
  weights) inside `UpdateState`, never written into the output ring, and zeroed
  after every `Step()`.

The split between `vtx_state_` (the write target) and the output ring (the read
source) is what makes the update **synchronous**: every vertex reads only the
published slices and the injected input, and writes only to `vtx_state_`. No
vertex ever sees a half-updated neighbor, so the per-vertex updates are mutually
independent — and trivially parallelizable.

Two small design choices keep the history clean. First, the injected input lives
in its own buffer with its own weight block, so the M output slices hold *pure
activations* — no input contamination, which matters the moment M > 1 and those
slices become a delay line the readout reads. Second, because input is a
separately-weighted term (never folded into the published output), the leaky
integrator's carryover is simply slice 0 — last step's clean activation, read
directly — with no separate "previous" buffer and no risk of double-counting the
drive.

### Phase 1 — compute new states (independent across vertices)

For each vertex `v`, the drive `s` is two sums: an input term through its own
weight block, and a recurrent term over the M history slices, each slice carrying
its own DIM weights.

```
s = 0
# (a) input term — decoupled input weights W_in (an N·DIM block)
for i = 0 to DIM - 1:
    s += input[v XOR (1<<i)] * W_in[v][i]
# (b) recurrent term — M slices × DIM axes, one weight per (slice, axis)
for j = 0 to M - 1:
    for i = 0 to DIM - 1:
        s += slice_j[v XOR (1<<i)] * W_rec[v][j][i]
activation = tanh(s)
state[v] = (1 - leak_rate) * slice_0[v] + leak_rate * activation
```

`slice_j` is `slice_ptrs_[j]`, and `slice_0[v]` — last step's output — is the
leaky-integrator carryover. At M = 1 the recurrent term collapses to a single
slice (the [deep-vertices](#deep-vertices-memory-you-can-address) bit-for-bit
fallback). `leak_rate` (default 1.0) sets how fast a neuron replaces its state: at
1.0 the old state is fully overwritten each step; at 0.3, 70% persists — a leaky
integrator that smooths dynamics and stretches temporal memory.

### Phase 2 — age the ring and publish

```
rotate slice_ptrs_ by one       # oldest physical slot becomes the new slice 0
memcpy(slice_ptrs_[0], state, N * sizeof(float))   # publish into new slice 0
memset(input, 0, N * sizeof(float))                # consume the injected drive
```

Aging is a **pointer rotation**, not a data shuffle: the M slice pointers rotate
so what was slice `j` becomes slice `j+1`, the oldest slot is recycled as the new
slice 0, and the freshly computed state is `memcpy`'d into it — one `memcpy` per
`Step()` regardless of M, the older slices never moving. Because every vertex read
only last step's published slices (never another vertex's just-computed
`vtx_state_`), the loop is embarrassingly parallel; the shipped `Step()` runs it
serially.

## Input injection

Input enters through `InjectInput(channel, value)`, called **before** each
`Step()`. The raw scalar for channel `c` is broadcast across that channel's
contiguous vertex block — with a single input, that's the whole cube:

```
input[v] = value     for v in [c·B, (c+1)·B),   B = N / num_inputs
```

The reservoir never clamps the value; callers pass already-bounded signals. The
randomness enters later, inside `UpdateState`, where each vertex sums its DIM
neighbors' injected fields through the **decoupled input-weight block** W_in (the
first N·DIM weights, drawn uniform[−1, 1] × `input_scaling`/√DIM):

```
s += input[v XOR (1<<i)] · W_in[v][i]     for i = 0 .. DIM-1
```

So a flat single-channel drive still lands as a *distinct per-vertex random
projection* — each vertex's own random sum of DIM input weights. The buffer is
zeroed after every `Step()` and never written into the output ring, keeping the M
history slices free of input contamination. And because the `1/√DIM` factor
normalizes the input fan-in, a given `input_scaling` delivers the same `tanh`
drive at every DIM — DIM-invariant by construction (default `0.5`).

### Multi-input mode

For K channels (`num_inputs = K`), each drives a **contiguous block** of B = N/K
vertices:

- channel 0 → vertices [0, B)
- channel 1 → vertices [B, 2B)
- channel k → vertices [k·B, (k+1)·B)

K must divide N evenly (validated at construction). The channels don't stay walled
off: cross-channel mixing happens for free through the recurrent nearest-neighbor
bit-flips, which connect vertices across adjacent block boundaries — so every
channel still reaches the whole reservoir within a few steps.

## Spectral radius: tuning the edge of chaos

The spectral radius is the reservoir's master dial — it sets where the dynamics
sit between dead and chaotic:

- **Too low** — the state forgets almost immediately. Short fading memory, weak on
  anything that needs history.
- **Just right** — rich dynamics with stable fading memory: the "edge of chaos"
  where reservoir computing earns its keep. The sweet spot is topology- and
  task-dependent.
- **Too high** — useful structure dissolves into chaos that no longer tracks the
  input.

### What operator, exactly?

At M = 1 the radius is just the dominant eigenvalue of the N × N recurrent matrix.
At depth M it is something subtler, and getting it right is what makes deep
vertices behave. The reservoir's true state is the stacked **MN-dimensional**
vector of all M slices, and one `Step()` is a tanh of a linear map whose linear
part is a **companion-form operator**: the top block mixes the slices by the
recurrent weights, while the lower blocks merely shift slice `j` into slice `j+1`
— the aging. The spectral radius is therefore estimated by power iteration on this
MN × MN companion operator (`EstimateSpectralRadius`, up to 500 iterations with a
convergence check), so the reported value is the true spectral abscissa of the
*M-step* recurrence, not of any single slice.

### Hitting the target

Recurrent weights start from uniform[−1, 1] scaled by `1/√(DIM·M)`, then the whole
recurrent block is rescaled so the companion operator's radius matches the target.
Only the recurrent block is touched; the decoupled input block stays fixed at
`input_scaling`/√DIM, so input drive and recurrent dynamics tune independently.

The rescale is a **secant root-find** on the weight scale, not a single multiply —
because for M > 1 the companion operator's dominant eigenvalue is a *nonlinear*
function of that scale (a longer operator's spectrum doesn't move linearly under
uniform scaling). At M = 1 the relationship is exactly linear and the root-find
lands on target in one step. See `Reservoir.cpp` (`Initialize()` and
`EstimateSpectralRadius()`) for the full rationale.

One coupling to keep in mind: the same input drive now feeds the edge-of-chaos
balance through M times as many recurrent pathways, so the surveyed spectral
radius and the chosen depth **interact** — they do not tune independently of each
other.

## Computational properties

- **O(N · DIM · (M+1)) per step** (M = `history_depth`, default 16) — each vertex
  sums DIM weighted neighbor reads for the input term plus DIM for each of the M
  history slices, against O(N²) for a dense ESN.
- **N · DIM · (M+1) weights total** — an N·DIM input block plus an N·DIM·M
  recurrent block, and still zero adjacency storage (neighbors computed by XOR).
- **No pointer-chasing.** A neighbor address is arithmetic — `v XOR (1<<i)` — not
  a load from an adjacency table, so the table a random graph must keep hot in
  cache simply doesn't exist, and there's no dependent indirect load. The
  per-vertex weight blocks are walked contiguously, which streams cleanly. The
  neighbor *state* reads are a different story — high-bit flips land N/2 apart in
  the buffer, so that gather is a strided scatter, not a cache-local sweep; the
  cache win is in the wiring, not the gather.
