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
mechanics of a timestep, input, optional feedback and bias, spectral radius, and
cost. The implementation is `Reservoir` / `ReservoirConfig` in `Reservoir.h` and
`Reservoir.cpp`; instances come only from `Reservoir::Create(cfg)`.

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

## ReservoirConfig (defaults and ranges)

| Field | Default | Range / rule |
|-------|---------|----------------|
| `dim` | 10 | **[5, 16]** → N = 2^dim neurons |
| `seed` | 73895 | master seed; named substreams via SplitMix64 (recurrent / input / external-feedback / bias / SR probe) |
| `spectral_radius` | 0.99 | **> 0** (target for recurrent-block rescale) |
| `leak_rate` | 1.0 | **(0, 1]** — 1 = full replacement |
| `input_scaling` | 0.5 | drive strength; weights × `input_scaling`/√dim (fan-in variance; retune per task/DIM) |
| `num_inputs` | 1 | **≥ 1** and must **divide N** evenly |
| `history_depth` (M) | 16 | **[1, 64]** — delay-line length |
| `verbose` | true | construction banner to stderr |
| `num_external_feedback_channels` (D) | 0 | **0** = no external-feedback path; else **[1, N]** (need **not** divide N) |
| `external_feedback_scaling` | 0.5 | like input: × scaling/√dim (only if D > 0) |
| `bias_scaling` | 0.02 | U(−1,1)×scale per neuron; **0 disables** bias |

`GetConfig()` returns these fields with `spectral_radius` = the **configured
target**, not the realized estimate — use `GetRealizedSpectralRadius()` for the
post-secant value. `Create(GetConfig())` rebuilds matching weight blocks from
`seed`.

## A topology you don't store

Most echo-state networks wire their neurons at random — generate a sparse graph,
store it, and trust it. HypercubeESN doesn't store a graph at all. Its N = 2^DIM
neurons sit on the vertices of a **Boolean hypercube**: each vertex is addressed
by a DIM-bit binary index, and two vertices are neighbors exactly when their
indices differ in a single bit.

That definition *is* the whole graph. A neighbor along bit `i` is just
`v XOR (1 << i)` (`NearestMask(i)`) — one instruction, computed on demand. No
adjacency list is built, stored, or serialized.

| DIM | N (neurons) | Connections/neuron | Edges/slice (N·DIM) |
|-----|-------------|--------------------|---------------------|
| 5   | 32          | 5                  | 160                 |
| 6   | 64          | 6                  | 384                 |
| 7   | 128         | 7                  | 896                 |
| 8   | 256         | 8                  | 2,048               |
| 9   | 512         | 9                  | 4,608               |
| 10  | 1024        | 10                 | 10,240              |

The last column is the directed-edge count of a *single* recurrent slice. Stored
**weights** are larger and depend on M and optional feedback (see
[Weight layout](#weight-layout)).

DIM is constrained to **[5, 16]** — 32 to 65,536 neurons.

### Every single-bit-flip neighbor

Each gather (input, feedback, recurrent) uses *all* DIM Hamming-distance-1
neighbors:

```
mask for neighbor i: 1 << i,  i = 0 .. DIM-1
```

Using the full DIM neighbors (not a truncated subset) is what gives the next
section its structure.

## The hidden structure: multi-scale soft modularity

Full neighbor connectivity is geometrically uniform — every vertex wired
identically to its DIM neighbors — yet the *weighted* graph is not. Random
weights, read the right way, hide a hierarchy of structure inside the plain cube.

Recurrent weights are drawn from a symmetric uniform distribution, so plenty land
near zero. A near-zero weight on bit-`i` into vertex `v` is a **soft cut** of that
edge. Soft cuts happen independently per `(vertex, bit)`.

Sweep a magnitude cutoff and keep only edges above it: the surviving subgraph’s
components partition the cube (bond percolation, critical scale near `1/DIM`).
Every vertex lives in **all** of these partitions at once — tight local clusters
at a strict cutoff, regional merges as the cutoff loosens, full cube at zero
cutoff.

A global DIM−1 truncation (omit one bit axis everywhere) would split the cube into
two disjoint subcubes forever. Full-DIM random weights buy both compartmentalization
and scale-spanning overlap. That multi-scale soft modularity is a product story
about the random graph, not a separate code path.

## Deep vertices: memory you can address

A standard ESN keeps one step of feedback. `history_depth` M keeps an addressable
**delay line** of the M most-recent published output slices. Each vertex update
sums over DIM spatial neighbors **and** those M temporal slices.

### Space × time kernel

At M = 1 each vertex applies a length-DIM weight vector to its neighbors. At depth
M it applies an **M × DIM weight bank** — one weight per `(slice, axis)`. Recurrent
weights for that bank are filled at scale `1/√(DIM·M)` (before the spectral-radius
rescale).

### Depth resolves; leak blurs

`leak_rate` smears past into present with one exponential time-constant. The delay
line exposes M separately weighted taps. They compose: leak sets how fast one
slice forgets; depth sets how many cleanly separated slices exist.

The true linear state for spectral-radius work is the **MN-dimensional** stack of
slices; see [Spectral radius](#spectral-radius-tuning-the-edge-of-chaos).

### Ring buffer

Slices live in one `M·N` buffer with M rotating pointers (`slice_ptrs_`). Aging
costs one pointer rotation + one `memcpy` of N floats per `Step()`, independent of
M. `history_depth` ∈ **[1, 64]**. At M = 1 the ring is a single slot and dynamics
match a single-step reservoir.

Depth is a generic recurrent move, orthogonal to XOR topology: the kernel is still
labelled by axis, so hypercube structure remains readable.

## Weight layout

`vtx_weight_` is one contiguous float array:

```
[ input:  N × DIM ]
[ external feedback: N × DIM ]  # only if num_external_feedback_channels > 0
[ recurrent: N × M × DIM ]  # layout [vertex][slice][axis], matches UpdateState
```

| Block | Size | Init scale (pre-SR) | In SR rescale? |
|-------|------|---------------------|----------------|
| Input | N·DIM | `input_scaling / √DIM` | **No** |
| External feedback | N·DIM or 0 | `external_feedback_scaling / √DIM` | **No** |
| Recurrent | N·DIM·M | `1/√(DIM·M)` | **Yes** (whole block, one scalar) |

Total weights:

```
N · DIM · (M + 1 + [ext-fb ? 1 : 0])
```

Example: DIM 10, M 16, no feedback → 10,240 × 17 ≈ 174K floats.

## Anatomy of a timestep

Per-step quantities:

- `vtx_state_[v]` — this step’s newly computed state (write target of `UpdateState`).
- `vtx_output_history_` / `slice_ptrs_` — ring of M published slices. **Slice 0** is
  newest (`Outputs()`, recurrent age 0). Slice `j` is age `j`.
- `vtx_input_[v]` — staged input field; cleared after every `Step()`.
- `vtx_ext_feedback_[v]` — staged external feedback if D > 0; cleared every `Step()`.
- `vtx_bias_[v]` — fixed U(−1,1)×`bias_scaling` (or zero if scale is 0); **not**
  cleared by `Clear` / not in snapshots.

Updates are **synchronous**: every vertex reads only published slices and staged
drives, writes only `vtx_state_`.

### Contract

```
InjectInput(...)       // optional, each channel you need this step
InjectExternalFeedback(...)  // optional, if D > 0
Step()                 // update all neurons, age ring, clear staged drives
// read Outputs() / SliceAt(age)
```

Staged input / external-feedback are **consumed and zeroed** by `Step()` — re-stage
every timestep.

### Phase 1 — compute new states (per vertex `v`)

```
s = 0
# (a) input — neighbor gather on vtx_input_
for i = 0 .. DIM-1:
    s += input[v XOR (1<<i)] * W_in[v][i]

# (b) external feedback — same gather (omitted if D == 0)
for i = 0 .. DIM-1:
    s += ext_fb[v XOR (1<<i)] * W_ext[v][i]

# (c) recurrent — M slices × DIM axes
for j = 0 .. M-1:
    for i = 0 .. DIM-1:
        s += slice_j[v XOR (1<<i)] * W_rec[v][j][i]

activation = tanh(s) + bias[v]
state[v]   = (1 - leak_rate) * slice_0[v] + leak_rate * activation
```

The nonlinearity is plain **`tanh(s)`**. Bias is added **after** the nonlinearity,
before the leak blend. (An experimental central-slope envelope was studied and
removed; see the archived notes in [ActivationFunctionA.md](ActivationFunctionA.md).)

For a single input channel the injected field is uniform, so the input gather is
equivalent to one scalar times the sum of the row of `W_in`; the code still uses
the general multi-input form (neighbors can straddle channel blocks when
`num_inputs > 1`).

### Phase 2 — age the ring and publish

```
rotate slice_ptrs_ by one          # oldest physical slot becomes new slice 0
memcpy(slice_ptrs_[0], state, N)   # publish
memset(input, 0, N)
if ext-fb configured: memset(ext_fb, 0, N)
```

Shipped `Step()` runs the vertex loop **serially** (embarrassingly parallel in
principle).

## Input injection

`InjectInput(channel, value)` before `Step()`. Channel `c` writes `value` onto
vertices `[c·B, (c+1)·B)` with `B = N / num_inputs` (exact division required).

The reservoir does not clamp; callers pass already-bounded signals. Randomness is
in the **weights**, not the broadcast. After `Step()`, the input buffer is zero —
history slices stay free of raw input (only post-activation state is published).

### Multi-input mode

K channels → K contiguous blocks of B = N/K vertices. Cross-channel mixing still
occurs through recurrent bit-flips near block boundaries.

## External feedback injection (optional)

When `num_external_feedback_channels = D > 0`, a second driver path mirrors input:

- Own weight block (outside spectral-radius rescale — does **not** guarantee
  closed-loop stability).
- `InjectExternalFeedback(channel, value)` or `InjectExternalFeedback(ptr, count)`
  with `count == D`.
- Block size `floor(N/D)`: if D does not divide N, the tail vertices stay at
  reset-zero as *sources* but still receive drive via neighbor gather.

Typical closed-loop use: stage last step’s readout-derived signal (y(t−1)), then
`Step()`.

## Per-neuron bias (optional)

Drawn once at construction: `U(−1,1) * bias_scaling`. Default scale **0.02**
(set `0` for no bias). Fixed model parameter — survives `Clear()`, omitted from
`TakeSnapshot` / `RestoreSnapshot`. Outside the spectral-radius operator.

## Spectral radius: tuning the edge of chaos

The spectral radius sets where dynamics sit between dead and chaotic:

- **Too low** — short fading memory.
- **Just right** — rich dynamics with stable fading memory (“edge of chaos”).
- **Too high** — chaos that no longer tracks the input.

### What operator?

At M = 1 the radius is the dominant eigenvalue of the N × N recurrent matrix. At
depth M the linear state is the MN-vector of all slices; one step’s linear map is a
**companion-form** operator (top block mixes slices by recurrent weights; lower
blocks shift ages). Only the **recurrent** weight block participates;
input/feedback/bias are outside the estimate.

### Hitting the target

1. Fill recurrent weights ~ U(−1,1) / √(DIM·M).  
2. **Secant root-find** on a global scale so estimated ρ ≈ target (relative tol
   0.1%, max 20 secant evals). M = 1 is nearly one-shot; M > 1 is nonlinear in the
   scale.  
3. `EstimateSpectralRadius` uses power iteration on the companion operator with
   **Gelfand (geometric-mean) growth rates** (complex dominant pairs / rotation),
   burn-in, and stability checks — hard cap **1500** iterations per eval (not a
   fixed 500-step naive power method).

`verbose` prints one line: DIM, M, seed, leak, input_scaling, SR target / post,
secant iters.

Input drive and recurrent dynamics still interact through the nonlinearity and
depth: surveyed spectral radius and M are not fully independent in practice.

## Snapshots

`TakeSnapshot()` captures `vtx_state_` and the history ring in **logical age
order** (slice 0 first). Staged input / external feedback buffers are not
included (empty between steps). Weights and bias are not included — restore only
onto the same (or identically configured) reservoir.

`RestoreSnapshot` copies state + history, re-homes the ring to canonical rotation,
and clears staged drives. Replaying the same injections afterward is bit-exact.

`SliceAt(age)` is the live delay-line view (age 0 ≡ `Outputs()`); throws if
`age ≥ history_depth`. Prefer this over raw buffer layout (ring rotation makes
physical block order meaningless).

## Computational properties

- **Time per step:** O(N · DIM · M) recurrent + O(N · DIM) per enabled drive port
  (input always; + external feedback) — i.e.
  **O(N · DIM · (M + 1 + [ext]))**, vs O(N²) for a dense ESN.
- **Weights:** **N · DIM · (M + 1 + [ext])** as above; zero adjacency storage.
- **Neighbor addresses** are arithmetic (`v XOR (1<<i)`); weight rows are contiguous.
  Neighbor *state* gathers are still strided (high-bit flips jump by N/2).
- **Parallelism:** vertex updates are independent; the released `Step()` is serial.

## Public surface (quick index)

| API | Role |
|-----|------|
| `Create(cfg)` | only constructor path |
| `InjectInput` / `InjectExternalFeedback` | stage drives |
| `Step` | one timestep |
| `Outputs` / `SliceAt` | read state / delay line |
| `Clear` | zero state + history; keep weights & bias |
| `GetConfig` / `GetRealizedSpectralRadius` | inspect |
| `TakeSnapshot` / `RestoreSnapshot` | branch / replay |
| `Dim` / `Size` | DIM / N |

ESN owns a `Reservoir` and layers warmup, multi-slice readout packing, and the
HCNN readout on top — see [Readout.md](Readout.md) and [CPP_SDK.md](CPP_SDK.md).
