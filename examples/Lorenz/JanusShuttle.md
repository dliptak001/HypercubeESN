# The Janus Shuttle Method for Half-Anchored Generative Free-Run

> Design spec for the **Janus Shuttle**: a dual-cursor scheme for reservoir/ensemble free-run
> prediction. The mechanism below is settled; open design points are
> tracked in §7 and `?` marks anything still inferred rather than decided.

**Method vs. instantiation.** The Janus Shuttle is **system-agnostic** — it works for any time
series an array can hold (the cursor's only contract is `value(position)`). This document
instantiates it on **Lorenz-63** throughout: `S[·]` is a Lorenz orbit, the channels are
`(x, y, z, x·z)`, and the normalization figures are Lorenz's. Read the *mechanism* as generic and
the *Lorenz specifics* as the worked example. (When this spec becomes the Lorenz module's README,
that split must be stated explicitly, not just implied.)

## Name & rationale

**Janus Shuttle** (use "the Janus Shuttle method" when context needs the noun). The **method** is
"Janus Shuttle" (this spec, `JanusShuttle.md`); the **class** implementing only the cursor traversal
is `JanusCursor` (`JanusCursor.{h,cpp}`). Two names, two scopes, deliberately kept distinct.

- **Janus** — the two-faced Roman god of thresholds, one face to the past, one to the future. The
  center seam is that threshold: `past_cursor_` faces the **known past** (anchor), `future_cursor_`
  faces the **generated future**. Owns the *free-run* phase.
- **Shuttle** — the reflecting back-and-forth scan that bounces off `[lb_, ub_]` (triangle
  wave). Owns the *training* phase.
- One name, both phases, both mechanisms; also nods to the two reflections in play —
  boundary reflection (the shuttle) and center-mirror reflection (the `±i` symmetry of the
  two cursors about "now").

## 0. One-paragraph statement

We drive an ensemble ESN with **two cursors** that index a single **precomputed** positive-time
Lorenz trajectory `S[·]`, moving in opposite directions from a shared **center** index. Each cursor emits a
4-vector `(x, y, z, x·z)`, so the input is **8-D**. In training the cursors stay inside a
**reflecting window** (bounded shuttle). In free-run they break out of the window and run
away from center: the **future** cursor walks into the *unknown future* and becomes
**generative** (fed by the ESN's own prediction), while the **past** cursor walks into
the *known past* and stays **ground-truth** — a real signal that anchors half the input
space and keeps the generative half from drifting off the attractor.

Everything is positive time. The orbit is integrated **once** from the seed into an array;
the cursors only ever **read indices** — no `step(-dt)`, no backward integration, no recompute.

**Cursor names vs. roles.** The two cursors are named by the *temporal direction* they scan
(invariant across both phases) and described by *role* (which only applies in free-run):

```
 future_cursor_ (= +i, samples S[N_c + i])  →  GENERATIVE half in free-run   (upper 4 inputs)
 past_cursor_   (= −i, samples S[N_c − i])  →  ANCHOR half, always real       (lower 4 inputs)
```

Temporal direction is the **name** because it holds in every phase: in training *both* cursors read
real Lorenz inside the window — neither generates. "Generative" and "anchor" name what each
half *becomes* once free-run starts; use them for the half/role, never as a synonym for the
cursor itself.

## 1. The axis: four regions, one center

```
 array index n:   0 ······· lb ····· center ····· ub ········· ub+E
 S[n]:           [S₀ ······ Sₗ ······· S_c ······· Sᵤ ········· Sₑ ]   ← integrated ONCE from seed
                  │         │           │           │            │
                 seed   train edge   anchor pt   train edge   eval horizon
                 T=0       (lb)       (center)      (ub)        (ub+E)

 center = N_c (an INDEX, not a state)     window = [lb, ub] = [N_c−H, N_c+H],  H = span/2
   future cursor sample → S[N_c + i]   (real lookup while inside the window)
   past   cursor sample → S[N_c − i]              (i = shared shuttle displacement)
   region [0, lb)    = past free-run runway (real history left of the window)
   region [lb, ub]   = training window (reflecting shuttle; both cursors live here)
   region (ub, ub+E] = PREDICTION / EVALUATION window — the future cursor goes GENERATIVE
                       here, so the upper-4 inputs are the ensemble's own output (NOT written
                       back into S). The TRUE orbit S[N_c + i] is still precomputed across this
                       band purely as a yardstick: it lets us score generated xyz against
                       expected xyz to gauge training effectiveness (when a true tail exists).
   region (ub+E, ∞)  = open-ended free-run — generative with NO reference past the horizon.
```

**What `S[n]` is, and how it is made.** `S[n]` is the **raw Lorenz state** at integration step
`n` — the bare 3-tuple `(xₙ, yₙ, zₙ)`. It is *not* the 8-D input vector and *not* a reservoir
state; both of those are built downstream from it. The array is produced **once, at setup**, by
fixed-step RK4 from the seed state at `T = 0`, marching forward with a single `dt` to the array's
upper limit (`ub`, or `ub+E` when an eval tail is available offline — §5):

```
 seed s₀ (e.g. {1,1,1})  ──RK4(dt)──▶ s₁ ──▶ s₂ ──▶ … ──▶ s_{ub(+E)}
 stored as:               S[0]        S[1]   S[2]         S[N_c+H(+E)]
 physical time:           t = n·dt    (the index n IS the clock — one dt for the whole array)
```

Everything is **positive time**: every index `n ≥ 0` is a real, forward-integrated state. There
is no negative time and no backward integration anywhere — the **past** cursor's name refers only
to the *direction* it scans this one array (`S[N_c − i]`), never to a `step(−dt)`.

**Role — one array, two readers.** `S` is the single shared source of truth. Both cursors are
pure **index readers** into it: `future_cursor_ = S[N_c + i]`, `past_cursor_ = S[N_c − i]`. The 8-D input
is simply two reads of the *same* array — lower 4 from the past read, upper 4 from the
future read — each passed through the per-variable normalization of §4b. Because both cursors
ride one array built with one `dt`, they sit on the identical orbit by construction: no drift, no
per-cursor integrator to keep in sync, no chance of the two halves diverging numerically. `S`
stores only the 3 raw coordinates; the 4th channel `x·z` is formed at lookup time from the
(scaled) `x` and `z`, never stored.

**Setup scan for per-channel extremes.** The same one-time setup pass that fills `S` then
**scans the full array** (eval band included, when present) for each channel's `min`/`max`, and
from those derives the normalization offset/scale `(c_v, h_v)` of §4b. So setup is two cheap
sweeps over `S`: **integrate** to fill it, **scan** to calibrate it — both completed before any
warmup or training touches the reservoir, and both frozen for the rest of the run.

The prediction/evaluation region `(ub, ub+E]` is where we read off whether training worked:
compare the future cursor's *generated* `(x,y,z)` against the true `S[N_c + i]`, step for step.
But scoring needs that true future to *exist*, and the `(ub, ub+E]` band is **not** guaranteed —
it is there only when the orbit was precomputed past `ub` (an offline simulation, or recorded data
with a held-out tail). On real-time data the future has not happened yet: there is no `Sₑ` tail, so
the region collapses into open-ended generative free-run. Crucially, that costs only the *score*,
not the *run* — the past cursor's anchor (`S[N_c − i]`, the lower 4 inputs) is real history and is
**always** available, so the half-anchoring keeps the free-run tethered to ground truth whether or
not the future eval tail exists.

## 2. Two phases = the two existing cursor moves

| phase     | cursor move        | code today            | i behavior                  |
|-----------|--------------------|-----------------------|-----------------------------|
| TRAINING  | reflecting shuttle | `StepBounded()`       | triangle wave inside [lb,ub]|
| FREE-RUN  | one-way ramp       | `StepUnbounded()`     | grows monotonically past ub |

After `StepUnbounded()`, `OOB()` reports `+1` once the future cursor passes `ub`. That `+1` is
exactly the **"future cursor crossed the right reflection limit"** signal → the moment
the future cursor flips from reading data to being generative.

**Training is multi-sweep, NOT one pass.** Both cursors start at center as mirrors
(`future_cursor_=+i`, `past_cursor_=−i`) and oscillate: `center → ub_ → lb_ → ub_ → …`, reflecting at
*both* boundaries and crossing center every half-sweep, repeated a **specified number of
times**. The whole window is swept back and forth repeatedly (the multi-epoch presentation).
`JanusCursor::StepBounded()` already reflects forever — "N sweeps" is just how many
`StepBounded()` calls the training loop drives; the cursor itself is unchanged. The sweep
count is **chosen so training terminates with both cursors back at center** (`i = 0`,
`future_cursor_` poised to step positive), handing the warm reservoir off seamlessly to free-run (§3).

```
  ub_ ┤   ╱╲        ╱╲        ╱╲          future_cursor_ (= +i)
  ctr ┼──╱──╲──────╱──╲──────╱──╲──────
  lb_ ┤ ╱    ╲╱        ╲╱        ╲        past_cursor_ = mirror about ctr
      └──────────────────────────► step   (× N sweeps)
```

## 3. Free-run dynamics: diverge from center, then one switch

Free-run does **not** start at the window edges — **both cursors start at the center** and
move apart:

```
 t=0 of free-run:   both at center (i=0)
 then every step:   future_cursor_ index += 1 (→ ub_ and beyond)
                    past_cursor_   index -= 1 (→ lb_ and beyond)   [symmetric, mirror]
 center → boundary: "natural washout" — still inside [lb_, ub_], still real Lorenz,
                    reservoir stays on in-distribution data before anything generative.
 they reach lb_/ub_ together (symmetric divergence).
```

**The reservoir state is NEVER reset between phases — it carries straight through.** Training
is configured to *end* with both cursors back at center (`i = 0`, `future_cursor_` poised to step
positive), so free-run resumes from exactly the warm, in-distribution state the last training
step left behind. The "washout" above is therefore **continuity, not a cold start**: the
center→boundary stretch is real Lorenz keeping an already-settled reservoir in-distribution
right up to the generative switch — there is no transient to discard and nothing to re-warm.

**Data source by cursor / region** — note only ONE thing ever switches:

```
                     in [lb_, ub_]  (train + washout)     past the boundary (free-run)
 past_cursor_   (lower 4)   Lorenz(pos)                        Lorenz(pos)   ← STILL REAL (anchor)
 future_cursor_ (upper 4)   Lorenz(pos)                        ENSEMBLE out  ← SWITCH (generative)
```

- **past_cursor_ never switches.** Past `lb_` it keeps reading `S[N_c − i]` — real history, the
  **anchor**. Half the input is always ground truth, all the way down to the seed (index 0).
- **future_cursor_ switches once,** the instant it passes `ub_` (`i > H`): it stops reading `S` and
  feeds the upper 4 channels from the **ensemble's own output**. **Generative** from there on.

"Lorenz(pos)" is just an **array lookup**: `future_cursor_ = S[N_c + i]`, `past_cursor_ = S[N_c − i]`,
where `S` is the orbit precomputed once from the seed (§5). No `step(-dt)`, no recompute —
both cursors only read indices. (`JanusCursor::StepBounded()` / `StepUnbounded()` already do
exactly this — pure index arithmetic on `idx_`, with no `step(±dt)` anywhere.)

## 4. The 8-input vector (fixed split)

```
 input[0..3] = LOWER  ← past_cursor_  : ( xb, yb, zb, xb·zb )   anchor   (always real)
 input[4..7] = UPPER  ← future_cursor_: ( xf, yf, zf, xf·zf )   generative past ub_
```

**Targets:** readout/ensemble predicts **3** outputs — the future cursor's `(x, y, z)` one
step ahead. `x·z` is a **derived input feature only**, never a target. In generative mode
the upper-4 = `(x̂, ŷ, ẑ, x̂·ẑ)` built from the ensemble's 3 predictions; in-window it's the
same 4-tuple built from real Lorenz. Same shape, only the source of the first three differs.

## 4b. Per-variable normalization (raw `S` → `[−1, 1]`)

The four channels span wildly different scales (Appendix A: `x, y, z` are O(tens), `x·z` is
O(hundreds)), and the reservoir's input projection is a **fixed random matrix scaled by a
single scalar** `input_scaling` — it has no per-channel knob and cannot rebalance the channels
on its own. So **between the array lookup and the reservoir** we apply a per-variable affine map
that pulls each channel into `[−1, 1]`. Every value shown in §4 is the *normalized* value; the
raw `S` lookup never reaches the reservoir directly.

**Notation:** a hat means *normalized*. `x` is raw Lorenz; `x̂ = (x − c_x)/h_x ∈ [−1, 1]`. In
generative mode the hatted values come from the readout (which predicts in normalized space); in
the window they come from scaling the real lookup — same symbol because it is the same quantity.

**The map.** Each channel is pulled into `[−1, 1]` by an affine map `v̂ = (v − c_v) / h_v`, with
the offset/scale set by channel type from that stream's measured extremes `[v_min, v_max]`:

```
 x, y  (bipolar, ~symmetric):  c_v = 0                    h_v = |v|_max = max(|v_min|, |v_max|)
 z     (positive, offset):     c_z = (z_max + z_min) / 2  h_z = (z_max − z_min) / 2
```

x and y keep `c = 0` so `v = 0 ↦ 0` (sign symmetry preserved — scaled by the largest excursion,
not the half-range); only z, sitting up at ~+24, carries a nonzero center.

```
 channel │ offset c_v │ scale h_v  │ normalized value
 ────────┼────────────┼────────────┼─────────────────────────────────────
 x       │    0       │  |x|_max   │  x̂ = x / h_x          (already symmetric)
 y       │    0       │  |y|_max   │  ŷ = y / h_y          (already symmetric)
 z       │   ≈ +24    │  half-rng  │  ẑ = (z − c_z) / h_z  (centered → straddles 0)
 x·z     │     —      │     —      │  x̂ · ẑ  ∈ [−1, 1]     (product of scaled; no own scale)
```

- The `z` **offset** `c_z` is the "make it bimodal" step: subtracting the DC center drops `z`
  from sitting up at +24 to straddling zero like `x, y` already do.
- The **4th channel needs no scaling of its own.** `|x̂·ẑ| ≤ |x̂|·|ẑ| ≤ 1`, so the product of two
  already-scaled values is bounded by construction — define it as `x̂·ẑ` and it lands in `[−1, 1]`
  for free.

**Extremes are per-stream, scanned over the full `S`.** `c_v` and `h_v` are computed **once at
setup by scanning the entire precomputed `S[·]`, including the eval band `(ub, ub+E]` when
present** (an offline tail); when there is none, it is the whole provided series. They are
**never hardcoded** — the Appendix-A figures illustrate one `{1,1,1}` run, they are not
constants. A different seed, or a real-world series, yields its own eight values
`(c_x, h_x, c_y, h_y, c_z, h_z)`. Scanning the *full* `S` (eval band included) is deliberate: it
guarantees the frozen `[−1, 1]` envelope covers exactly the orbit region the generative rollout
is trying to match, so the true comparison orbit is always in range. Once scanned, the eight
values are **frozen** and applied identically everywhere: anchor lookups, training inputs,
generative reconstruction, and the denormalization used for scoring.

**Pipeline placement:**

```
 S[N_c ± i]  ──affine(c,h)──▶  (x̂, ŷ, ẑ)  ──┬─────────────▶  (x̂, ŷ, ẑ, x̂·ẑ)  ──▶ reservoir
   raw lookup                                └──▶ x̂·ẑ  (4th channel)
```

The readout predicts normalized `(x̂, ŷ, ẑ)`; the 4th input is rebuilt as a bare multiply `x̂·ẑ`.

**Two payoffs.**

1. **Generative reconstruction is trivially correct.** Because the 4th channel is *defined* as
   the product of scaled values, free-run simply multiplies the two normalized predictions
   `x̂·ẑ`. The `denorm → multiply → renorm` round-trip — the bug that would stay invisible until
   free-run — **cannot occur**, because there is no renormalization step left to get wrong.

2. **The `z`-centering is harmless.** Centered `z` means the product encodes `x·(z − c_z)`:

   ```
    x̂ · ẑ  =  (x·z  −  c_z·x) / (h_x · h_z)
   ```

   not raw `x·z`. But `x̂` is *also* an input, so raw `x·z` is a linear combination of features
   the reservoir already holds (`x̂·ẑ` and `x̂`) — fully recoverable, **no information lost**. The
   Q3 native-nonlinearity rationale survives; centering only shifts the product by a recoverable
   multiple of `x`. (Mild cost: with `z` centered the product rarely reaches ±1 — when `|x|`
   peaks, `z` is mid-range — so the 4th channel under-fills `[−1, 1]`; harmless, `input_scaling`
   absorbs the slack.)

**Denormalization.** To score a generated `x̂` against the true orbit in physical units (the §1
eval yardstick), or to emit physical-space predictions, invert the map: `v = h_v · v̂ + c_v`. The 4th
channel is never denormalized — it is a derived feature, never a target (§4, Q4).

## 5. What this forces on the implementation

1. **Precompute the orbit once.** At setup, integrate forward from the seed state (T=0; the
   `{1,1,1}` used elsewhere is just one example seed) to the array's upper limit, which is one
   of two:
   - **`N_c+H` (= `ub`)** — when there is no reference tail to score against: a real-time
     free-run into the genuine future, or any run where no tail is held out. The array stops at
     the right edge of the training window.
   - **`N_c+H+E` (= `ub+E`, the eval horizon)** — when the future past `ub` is known offline (a
     precomputed simulation, or recorded data with a held-out tail). The extra band `(ub, ub+E]`
     is the true orbit we score the generative rollout against (§1).

   `LorenzAttractor::trajectory()` returns exactly such an array. One fixed `dt`, one array —
   both cursors index it, so they ride the identical orbit by construction.
2. **Cursors read indices, not integrators.** `future_cursor_ = S[N_c + i]`,
   `past_cursor_ = S[N_c − i]` — pure lookups, with **no** `step(±dt)` path that would force one
   cursor to integrate backwards and blow up. The center is not a stored state to reset to; it is
   just the index `N_c`, and the seam case is `S[N_c]`.
3. **Array bounds = the runtime envelope.** Right end = `ub`: the future cursor goes generative past it,
   and generated values are **not** written back into `S`. Left end = index 0 = seed = the
   past cursor's floor. Margin `N_c − H` (seed → `lb`) = the free-run anchor runway.
4. **`WarmupReservoir()` decouples from "find the center"** (now trivial — it's an index). It
   becomes purely: run the reservoir over the leading array region to settle its internal
   state before training/scoring.
5. **Future-cursor source switch** at the right-limit crossing: `S[…]` → ensemble output. The
   past cursor never switches.
6. `StepBounded` (train) / `StepUnbounded` (free-run) still model the two phases — they now
   advance an **index**, not an integrator.

## 6. Why it should work (intuition)

A pure generative free-run feeds the model only its own output — errors compound and it
slides off the attractor (the failure mode we keep hitting). Here, **4 of 8 inputs are
always real**. The reservoir state is continuously re-tethered to a true Lorenz signal, so
the generative half has far less room to run away. The past/anchor signal is a "free"
ground truth we happen to already know — we spend it as a stabilizer.

## 7. Open questions (let's resolve these)

- **Q1 — RESOLVED.** Mirror offsets `±i` off a shared index. Training: both shuttle inside
  `[lb_, ub_]`. Free-run: both start at center, diverge symmetrically (future +i, past −i).
- **Q2 — anchor runway / floor behavior.** The past cursor floors at array index 0 (the
  seed). Runway = `N_c − H` (seed-to-`lb` margin); deeper pre-roll = longer runway vs. a bigger
  array (negligible for Lorenz). Open: behavior at the floor — stop / clamp / end the run?
- **Q3 — RESOLVED.** The 4th channel is `x·z`, not `x·y·z`. `x·z` is a *native* Lorenz
  nonlinearity — it is the bilinear term in `ẏ = ρx − x·z − y`. Under the wing-swap symmetry
  `(x,y,z) → (−x,−y,z)`, `x·z` is **odd**, matching the odd targets `x,y`; the triple product
  `x·y·z` is **even** (redundant parity with `x·y`) and appears nowhere in the field, so it was
  rejected. Input feature only, never a target (see Q4). The 4-channel split `(x,y,z,x·z)` is
  preserved (8-D total, hypercube-friendly).
- **Q4 — RESOLVED.** 3 outputs (future x,y,z); `x·z` derived; anchor half never a target.
- **Q5 — RESOLVED.** Center is just the array index `N_c`. The seam "re-anchor" is simply the
  `i = 0` case → `S[N_c]`; there is no center *state* to recompute or reset.
- **Q6 — ensemble coupling.** Where does κ-consensus feedback sit relative to the future
  generative channel — does consensus replace/blend the future 4 inputs, or ride the
  separate feedback weight block alongside them?

## Appendix A — channel scales

With standard Lorenz params (σ=10, ρ=28, β=8/3), the four channels live on very different
scales. This is why input normalization is load-bearing for the `x·z` channel (the fixed
random input matrix + single scalar `input_scaling` cannot rebalance a channel that is ~10×
the scale of the others and heavier-tailed). Two ranges matter: the **typical** band where the
signal sits (≈±2σ) sets the input_scaling sweet spot; the **max** band (measured absolute
extremes) is what must not saturate tanh on a worst-case excursion — and for `x·z` the max
sits ~3.6σ out vs ~2.5–3σ for `x, y, z`, which is exactly the heavy tail biting.

```
 channel │ typical (≈±2σ)  │ max (extremes)  │ mean   │ std    │ distribution
 ────────┼─────────────────┼─────────────────┼────────┼────────┼────────────────────────
 x       │ [ -16,   16]    │ [ -19.5,  19.6] │   ~0   │   7.9  │ bimodal, ~symmetric
 y       │ [ -18,   18]    │ [ -27.0,  27.2] │   ~0   │   9.0  │ bimodal, ~symmetric
 z       │ [   6,   41]    │ [   1.0,  47.8] │ +23.6  │   8.6  │ positive, offset (z > 0)
 x·z     │ [-490,  490]    │ [-882.0, 891.8] │   ~0   │ 245.8  │ heavy-tailed (~3.6σ extremes)
```

*Measured by RK4 integration (dt=0.005) over ~10⁴ time units from the `{1,1,1}` seed, transient
included. The max envelope is stable run-to-run; the `x·z` extreme tracks the moments when `|x|`
and `z` peak together (out on a wing).*
