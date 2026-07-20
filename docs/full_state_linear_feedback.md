# Implementing Full-State Linear Feedback in HypercubeESN

> Status: **proposal / guidance**. This is not landed behavior. It specifies how to
> integrate the full-state linear feedback scheme of Ehlers, Nurdin & Soh
> (arXiv:2312.15141) on top of HypercubeESN. The reservoir already has a
> source-agnostic external-drive port
> ([reservoir_feedback_mechanism.md](reservoir_feedback_mechanism.md)); this
> document defines a *policy* for what to inject — a trainable full-state gain
> **V** — not a second internal feedback path inside the reservoir.

## Overview

The core idea is to optimize a feedback vector **V** in R^n (where n = N = 2^dim
is the reservoir size) that computes a linear combination of the current
reservoir state **x**_k and adds it to the input **u**_k. That effectively
modifies the recurrent dynamics:

```
A' = A + B V^T
```

without altering the hypercube topology, rewiring neighbors, or re-solving the
open-loop spectral-radius target on the recurrent block.

This maximizes flexibility for task-specific adaptation while preserving the
scale-invariant hyperparameters that HypercubeRC is built around (e.g. spectral
radius near 0.90, small DIM-invariant input scaling).

```
  paper (dense ESN)                 HypercubeESN (policy on existing ports)

  u_k ──► + ──► B ──┐               u_k ──► InjectInput ──────────┐
          ▲         │                        ▲                    │
          │ V^T x   │                        │ V^T x (same B)     │
          │         ▼                        │  -or- φ via        ▼
         x_k ◄── g(A x + ·)                 │  InjectFeedback   Step ──► x_k
          │         │                        │  (B_fb ≠ B_in)     │
          └─────────┘                        └────────────────────┘
```

## Mathematical Formulation (ESN with Feedback)

### Standard ESN

```
x_{k+1} = g(A x_k + B u_k)
ŷ_k     = W^T x_k + C
```

### Full-state feedback (Ehlers et al.)

```
u'_k    = u_k + V^T x_k
x_{k+1} = g(A x_k + B u'_k)
        = g((A + B V^T) x_k + B u_k)
```

| Symbol | Role |
|--------|------|
| g(·) | Element-wise nonlinearity (paper uses sigmoid; HypercubeESN default is `tanh`, optional Lorentzian envelope A) |
| A | Fixed recurrent weights (dense random in the paper; hypercube XOR neighbor weights here) |
| B | Fixed input weights |
| V | Trainable full-state feedback gain in R^n (or R^{n×D} for D drive channels) |
| W, C | Linear readout in the paper; HypercubeESN uses a hypercube CNN readout ([Readout.md](Readout.md)) |

**What changes and what does not:**

- **Changes:** the *effective* linear map seen by g — A' = A + B V^T — by
  reusing the existing input (or feedback) drive path.
- **Does not change:** hypercube wiring, the construction-time spectral-radius
  rescale of the *recurrent* weight block, or the fact that only a small set of
  parameters is trained (V plus the readout).

### Why this is not output feedback

Classic Jaeger-style output feedback injects ŷ (or a function of ŷ) back into
the reservoir. Full-state linear feedback injects a *learned projection of the
entire state* V^T x. That is a strictly richer family of closed-loop maps: every
output-feedback drive is a special case when V is constrained to the span of the
readout weights, but V may point anywhere in R^n.

The paper proves (Theorem 1) that for almost any fixed (A, B) and finite
training set, there exists a V that *strictly* improves the minimized training
cost relative to V = 0, while still admitting a convergent choice of V when A
itself is convergent.

## Why Full-State Feedback?

- Broadest linear rank-1 adjustment of the recurrent map that still uses only
  the existing input channel and the already-measured state.
- Near-universal improvement in training NMSE for almost all reservoirs (paper
  theorem; not a task-specific claim).
- Paper reports average error reductions of roughly 30–60% across three
  representative problem classes, with a boost comparable to roughly *doubling*
  reservoir size — at far lower cost than growing N.
- Fits HypercubeESN's design bias: keep the reservoir fixed and cheap; put
  task-specific adaptation in a thin trainable layer outside the topology.

## Mapping onto HypercubeESN

HypercubeESN is not a dense ESN. The same *math idea* maps cleanly; the
*realization* has a few structural differences you must keep straight.

### What the reservoir actually does

See [Reservoir.md](Reservoir.md). In short:

- Neurons sit on the vertices of a Boolean hypercube; neighbors are
  `v XOR (1 << i)`, never stored as a dense A.
- Each step gathers input, optional feedback, and history-depth recurrent
  slices through that XOR fan-in, then applies leak / activation / bias.
- Construction rescales **only the recurrent weight block** to a target
  `spectral_radius`. Input and feedback blocks sit **outside** that rescale.

### Two legal realizations of φ = V^T x

| Path | What you do each step | Effective map | Faithfulness to the paper |
|------|------------------------|---------------|---------------------------|
| **Same-B (recommended for theory match)** | `u'_k = u_k + V^T x_k` on the *input* channels, then `InjectInput` / `ReservoirStep(inputs)` | A' = A + B_in V^T | Exact paper form when u is scalar (or when V is n×K for K input channels) |
| **Feedback-port (recommended for decoupling)** | Keep u unchanged; set φ_k = V^T x_k and call `InjectFeedback` / `ReservoirStep(inputs, φ)` with `num_feedback_channels = D` | A' = A + B_fb V^T | Same *structure*, different B (feedback weight block, own `feedback_scaling`) |

Both are valid. Prefer **same-B** when you want the paper's theorem statement
literally. Prefer the **feedback port** when you want V-driven drive to stay
orthogonal to the task input path (separate scaling, separate seed substream,
clear open-loop baseline with `num_feedback_channels = 0` or V = 0).

The existing port contract still holds either way: the reservoir does not own
the policy. Something *outside* `Reservoir` owns V and stages the drive before
each `Step`. See [reservoir_feedback_mechanism.md](reservoir_feedback_mechanism.md).

### Multi-channel generalization

The paper treats scalar u and B in R^n. HypercubeESN allows K input channels
and D feedback channels (block-broadcast layout).

Natural generalizations:

```
scalar paper form:     φ_k = V · x_k            ∈ R,    V ∈ R^n
K-channel same-B:      φ_k = V^T x_k            ∈ R^K,  V ∈ R^{n×K}
D-channel feedback:    φ_k = V^T x_k            ∈ R^D,  V ∈ R^{n×D}
```

Start with **D = 1 or K = 1** (single scalar full-state gain). Multi-column V
is more expressive and more expensive to train; promote only if single-column
plateaus.

### Readout difference (important)

The paper's cost S is the residual of a *linear* readout after exact least
squares (W, C). HypercubeESN's production readout is a small hypercube CNN
([Readout.md](Readout.md)). Consequences:

1. Nested optimization still applies: for fixed V, train the readout to
   convergence (or for a fixed epoch budget), then evaluate task loss.
2. The paper's closed-form ∇_V S via the linear projection Π_x does **not**
   transfer unchanged. Either:
   - **(A) Linear probe for V** — optimize V against a temporary ridge / linear
     readout on collected states (faithful to the paper), then freeze V and
     train the HCNN; or
   - **(B) Task loss on the real readout** — treat V as outer-loop parameters;
     each trial of V re-runs the reservoir under that V and retrains / fine-tunes
     the HCNN, with finite-difference or RTRL-style gradients if needed.
3. Do not silently claim the paper's NMSE numbers for an HCNN readout; report
   the metric the actual pipeline optimizes (RMSE / NRMSE / R² / VPT / …).

## Optimization of V

### Nested loop (paper practice)

S(V) is **not convex** in V (paper Fig. 1). There is no closed-form global
optimum. The practical procedure is batch gradient descent with a nested linear
(or readout) fit:

```
  V ← 0
  for iter = 1 .. max_iters:
      1. Warm up the reservoir under u'_k = u_k + V^T x_k   (or φ_k = V^T x_k)
      2. Record states over the training window
      3. Fit readout (paper: exact W, C; HypercubeESN: HCNN Train / probe)
      4. Estimate g = ∇_V S_min  (analytic for linear probe; FD otherwise)
      5. V ← V − η g
      6. Project / correct V if closed-loop dynamics leave the safe set
  freeze V; train final readout for deployment
```

ASCII lifecycle:

```
  construct ESN
       │
       ▼
  ┌─────────────────────────────────────────┐
  │  outer loop over V                      │
  │    warm up → run → fit readout → loss   │
  │    update V (GD / FD / RLS-style)       │
  │    enforce stability correction         │
  └─────────────────────────────────────────┘
       │
       ▼
  freeze V, final readout train, eval on hold-out
```

### Practical choices for this codebase

| Knob | Suggested default | Notes |
|------|-------------------|--------|
| Init | V = 0 | Paper start; recovers open-loop baseline |
| Learning rate η | small, task-tuned | Start ~1e-3–1e-2 on normalized states; back off if loss jumps |
| Iterations | tens to low hundreds | Non-convex; early-stop on hold-out |
| Gradient | finite differences first | n = 2^dim FD costs n re-runs; use for dim ≤ 8 prototypes. Analytic RTRL later if needed |
| Linear probe | ridge on collected X | Cheap nested W,C; good V-direction even if final readout is HCNN |
| Stability | soft ‖V‖ cap + reject step | Paper's singular-value surface on dense A' is expensive here (A is implicit). Prefer: (1) reject updates that blow up state norms, (2) optional hard ‖V‖_∞ or ‖V‖_2 cap, (3) keep `spectral_radius` / `input_scaling` / `feedback_scaling` at known-good open-loop values |

### Stability notes specific to HypercubeESN

- Open-loop SR is still measured on the **recurrent block only**. Full-state
  feedback, like the existing feedback port, sits outside that budget.
- The paper's sufficient condition (singular values of A' below a bound set by
  g) is denser-matrix oriented. For the hypercube, treat closed-loop
  convergence as an empirical contract: after each V update, check that
  parallel trajectories from different initial states collapse under the same
  drive (echo-state smoke test), and that ‖x‖ stays bounded on long rollouts.
- Leaky integration and the activation's global Lipschitz constant still matter;
  do not "fix" instability by cranking spectral radius upward while searching V.

## Implementation Guidance

### 1. Core step (caller-side policy)

Do **not** rewire `Reservoir::UpdateState` to own V. Keep the reservoir
source-agnostic. Own V in a small helper (example-level or a thin ESN
extension) that stages drive before each step.

Same-B form (scalar channel):

```cpp
// Pseudocode — policy layer, not Reservoir internals
// x: current published state, length N
// V: feedback gain, length N  (default all zeros)
const float phi = Dot(V.data(), x, N);   // V · x
const float u_prime = u + phi;
reservoir.InjectInput(/*channel*/ 0, u_prime);
reservoir.Step();
// x advances; next step uses the new state
```

Feedback-port form (D = 1):

```cpp
const float phi = Dot(V.data(), x, N);
reservoir.InjectInput(/*channel*/ 0, u);
reservoir.InjectFeedback(/*channel*/ 0, phi);  // requires num_feedback_channels >= 1
reservoir.Step();
```

Or through the ESN seam already shipped:

```cpp
// ESN::ReservoirStep(inputs, feedback) — only closed-loop entry at ESN layer
float phi = Dot(V.data(), esn.GetState(), N);
esn.ReservoirStep(&u, &phi);   // open-loop: pass feedback = nullptr
```

Default V = 0 ⇒ φ = 0 ⇒ open-loop baseline bit-identical to today's path
(feedback port unallocated when `num_feedback_channels = 0`).

### 2. Where V lives

Suggested ownership (proposal, not API yet):

```
struct FullStateFeedback {
    std::vector<float> V;     // size N, or N * D
    size_t channels = 1;      // 1 for scalar φ
    enum class Path { SameInput, FeedbackPort } path = Path::FeedbackPort;
    float learning_rate = 1e-3f;
};

// per step
float* phi = ComputePhi(fb, reservoir.GetState());  // V^T x
// stage via chosen path, then Step
```

Persist V with the trained model if the deployment loop needs it (binary dump
or sidecar). Reservoir weights remain seed-reconstructible; V does not.

### 3. Training algorithm (minimal viable)

Phase the work so each step is independently testable:

1. **Inject path only** — hardcode a fixed nonzero V, confirm states diverge
   from V = 0 and that V = 0 matches today's open-loop golden.
2. **Nested linear probe** — collect states under V, solve ridge for (W, C),
   report NMSE vs V = 0 (paper-comparable).
3. **Outer GD on V** — finite-difference or coordinate descent for dim ≤ 8;
   log train/hold-out loss per iteration; enforce ‖V‖ cap.
4. **HCNN readout** — freeze best V, train production readout; A/B against
   V = 0 with matched compute (same epochs, same seed).
5. **Optional** — multi-column V, analytic gradients, online RLS-style V
   updates for streaming.

### 4. Config surface (proposed)

Keep defaults off so nothing changes until opted in:

| Field | Default | Meaning |
|-------|---------|---------|
| `full_state_feedback` | false | Master enable |
| `fsf_path` | `FeedbackPort` | `SameInput` or `FeedbackPort` |
| `fsf_channels` | 1 | Columns of V / width of φ |
| `fsf_learning_rate` | 1e-3 | Outer-loop η |
| `fsf_max_iters` | 50 | Outer GD iterations |
| `fsf_v_l2_cap` | (none / large) | Soft stability clamp on ‖V‖_2 |
| `fsf_probe` | `LinearRidge` | Nested readout used while training V |

Open-loop hyperparameters (`spectral_radius`, `input_scaling`,
`feedback_scaling`, `history_depth`, seed) stay at their existing scale-invariant
defaults unless a sweep says otherwise — **do not** retune the reservoir to
"make room" for V as a first resort.

### 5. Causality / timing

φ_k = V^T x_k uses the **current** published state *before* the step that
consumes it (same causal pattern as any external drive):

```
  have x_k
     │
     ├─► φ_k = V^T x_k
     ├─► stage u_k (and φ_k)
     └─► Step  ──► x_{k+1}
```

That matches the paper's simultaneous use of x_k in both the recurrence and the
feedback term. Do not delay V by an extra step unless you are deliberately
studying a discrete lag.

Warmup must run **under the same V** you will train with. A V trained after a
V = 0 warmup is a different dynamical system than one trained with consistent
closed-loop warmup.

### 6. Parallelism

If you parallelize FD probes or multi-seed sweeps, use `std::thread` /
`std::async` with **one ESN instance per thread**. Do not introduce OpenMP in
this codebase. `ESN` is not instance-thread-safe.

### 7. What not to do

- Do not bake V into `Reservoir` internals or treat full-state feedback as a
  second topology.
- Do not fold V into the spectral-radius secant solve (recurrent block only).
- Do not claim paper NMSE deltas until the same nested linear probe is in place.
- Do not justify API shape from a single benchmark task; keep the mechanism
  task-agnostic (policy + port).

## Verification Plan

Minimum bar before calling the feature real:

| ID | Check | Pass criterion |
|----|-------|----------------|
| V0 | V = 0 path | Bit-identical states to current open-loop for fixed seed and input |
| V1 | Fixed nonzero V | States differ from V = 0; ‖x‖ remains finite over long run |
| V2 | Nested linear probe GD | Train NMSE(V*) < NMSE(0) on a simple synthetic sequence (not used as design justification — only as a smoke test) |
| V3 | Hold-out | Improvement not pure train memorization |
| V4 | Stability reject | Oversized V step is projected / rejected; no NaNs |
| V5 | HCNN A/B | Frozen V* + production readout vs V = 0 control, matched budget |
| V6 | Port parity | FeedbackPort path with φ = V^T x behaves as documented in the substrate doc |

Science beyond the engineering bar (optional): multi-seed sweeps, dim scaling,
same-B vs FeedbackPort ablation, and whether a linear-probe V transfers to the
HCNN without re-optimizing V.

## Workflow (operator view)

```
  1. Build ESN with desired dim / SR / input_scaling
  2. Choose path: SameInput or FeedbackPort (set num_feedback_channels if needed)
  3. Optimize V (outer loop + nested readout / probe)
  4. Freeze V
  5. Final readout train under frozen V
  6. Deploy: each live step compute φ = V^T x, stage, Step, Predict
```

Inference cost is one dot product of length N per step — negligible next to the
reservoir gather (O(N · dim · history_depth)).

## Relation to Existing Docs

| Document | Relationship |
|----------|----------------|
| [reservoir_feedback_mechanism.md](reservoir_feedback_mechanism.md) | Landed **substrate** (external drive port). This doc is a **policy** that can use that port. |
| [Reservoir.md](Reservoir.md) | Hypercube dynamics, SR, input fan-in — unchanged by V. |
| [Readout.md](Readout.md) | Production trainable head; nested or final stage when training V. |
| [ActivationFunctionA.md](ActivationFunctionA.md) | Optional g(·); full-state feedback is independent of activation choice. |
| [CPP_SDK.md](CPP_SDK.md) / [Python_SDK.md](Python_SDK.md) | Update only after an API lands. |

## Reference

Ehlers, P. J., Nurdin, H. I., & Soh, D. (2023/2024).
*Improving the Performance of Echo State Networks Through Feedback*.
arXiv:2312.15141.

Primary takeaways used here:

- Definition u' = u + V^T x and equivalence A' = A + B V^T
- Universal superiority theorem (almost all reservoirs improve for some V)
- Non-convex S(V); batch GD from V = 0 with nested W, C fit
- Empirical boost comparable to roughly doubling reservoir size on the paper's
  three problem classes
