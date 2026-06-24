# The Janus Shuttle Method

> Design spec for the **Janus Shuttle**: a dual-cursor, half-anchored generative free-run for
> reservoir/ensemble prediction. The mechanism below is settled; open design points are
> tracked in §7 and `?` marks anything still inferred rather than decided.

## Name & rationale

**Janus Shuttle** (use "the Janus Shuttle method" when context needs the noun).

- **Janus** — the two-faced god of thresholds, one face to the past, one to the future. The
  center seam is that threshold: `backward_` faces the **known past** (anchor), `forward_`
  faces the **generated future**. Owns the *free-run* phase.
- **Shuttle** — the reflecting back-and-forth scan that bounces off `[lb_, ub_]` (triangle
  wave). Owns the *training* phase.
- One name, both phases, both mechanisms; also nods to the two reflections in play —
  boundary reflection (the shuttle) and center-mirror reflection (the `±i` symmetry of the
  two cursors about "now").

## 0. One-paragraph statement

We drive an ensemble ESN with **two cursors** that index a single **precomputed** positive-time
Lorenz trajectory `S[·]`, moving in opposite directions from a shared **center** index. Each cursor emits a
4-vector `(x, y, z, x*y*z)`, so the input is **8-D**. In training the cursors stay inside a
**reflecting window** (bounded shuttle). In free-run they break out of the window and run
away from center: the **forward** cursor walks into the *unknown future* and becomes
**generative** (fed by the ESN's own prediction), while the **reverse** cursor walks into
the *known past* and stays **ground-truth** — a real signal that anchors half the input
space and keeps the generative half from drifting off the attractor.

Everything is positive time. The orbit is integrated **once** from the seed into an array;
the cursors only ever **read indices** — no `step(-dt)`, no backward integration, no recompute.

## 1. The axis: three regions, one center

```
 array index n:   0 ··········· lb ········ center ········ ub
 S[n]:           [S₀ ·········· Sₗ ········· S_c ·········· Sᵤ ]   ← integrated ONCE from seed
                  │             │            │              │
                 seed       train edge    anchor pt     train edge
                 T=0           (lb)        (center)        (ub)

 center = N_c (an INDEX, not a state)     window = [lb, ub] = [N_c−H, N_c+H],  H = span/2
   forward  cursor sample → S[N_c + i]
   backward cursor sample → S[N_c − i]              (i = shared shuttle displacement)
   region [0, lb)  = backward free-run runway (real history left of the window)
   region (ub, ∞)  = forward generative — NOT stored in S (ensemble output)
```

Key invariant for "positive time only": the array starts at the seed (index 0), so the
backward cursor's floor is `T=0`. Array length = `N_c + H + 1`; the margin `N_c − H` between
the seed and `lb` is the free-run anchor runway.

## 2. Two phases = the two existing cursor moves

| phase     | cursor move        | code today            | i behavior                  |
|-----------|--------------------|-----------------------|-----------------------------|
| TRAINING  | reflecting shuttle | `BoundedStep()`       | triangle wave inside [lb,ub]|
| FREE-RUN  | one-way ramp       | `UnBoundedStep()`     | grows monotonically past ub |

`UnBoundedStep()` already returns the out-of-bounds report (+1 past `ub`). That `+1` is
exactly the **"forward cursor crossed the right reflection limit"** signal → the moment
forward flips from reading data to being generative.

**Training is multi-sweep, NOT one pass.** Both cursors start at center as mirrors
(`forward_=+i`, `backward_=−i`) and oscillate: `center → ub_ → lb_ → ub_ → …`, reflecting at
*both* boundaries and crossing center every half-sweep, repeated a **specified number of
times**. The whole window is swept back and forth repeatedly (the multi-epoch presentation).
`JanusShuttleCursor::step()` already reflects forever — "N sweeps" is just how many
`BoundedStep()` calls the training loop drives; the cursor itself is unchanged.

```
  ub_ ┤   ╱╲        ╱╲        ╱╲          forward_ (= +i)
  ctr ┼──╱──╲──────╱──╲──────╱──╲──────
  lb_ ┤ ╱    ╲╱        ╲╱        ╲        backward_ = mirror about ctr
      └──────────────────────────► step   (× N sweeps)
```

## 3. Free-run dynamics: diverge from center, then one switch

Free-run does **not** start at the window edges — **both cursors start at the center** and
move apart:

```
 t=0 of free-run:   both at center (i=0)
 then every step:   forward_ index += 1 (→ ub_ and beyond)
                    backward_ index -= 1 (→ lb_ and beyond)   [symmetric, mirror]
 center → boundary: "natural washout" — still inside [lb_, ub_], still real Lorenz,
                    reservoir settles on in-distribution data before anything generative.
 they reach lb_/ub_ together (symmetric divergence).
```

**Data source by cursor / region** — note only ONE thing ever switches:

```
                     in [lb_, ub_]  (train + washout)     past the boundary (free-run)
 backward_ (lower 4)   Lorenz(pos)                        Lorenz(pos)   ← STILL REAL (anchor)
 forward_  (upper 4)   Lorenz(pos)                        ENSEMBLE out  ← SWITCH (generative)
```

- **backward_ never switches.** Past `lb_` it keeps reading `S[N_c − i]` — real history, the
  **anchor**. Half the input is always ground truth, all the way down to the seed (index 0).
- **forward_ switches once,** the instant it passes `ub_` (`i > H`): it stops reading `S` and
  feeds the upper 4 channels from the **ensemble's own output**. **Generative** from there on.

"Lorenz(pos)" is just an **array lookup**: `forward_ = S[N_c + i]`, `backward_ = S[N_c − i]`,
where `S` is the orbit precomputed once from the seed (§5). No `step(-dt)`, no recompute —
both cursors only read indices. (The current `JanusShuttle::advance_()` still integrates with
`step(±dt)`; that is the code replaced by index lookups.)

## 4. The 8-input vector (fixed split)

```
 input[0..3] = LOWER  ← backward_: ( xb, yb, zb, xb*yb*zb )   anchor   (always real)
 input[4..7] = UPPER  ← forward_ : ( xf, yf, zf, xf*yf*zf )   generative past ub_
```

**Targets (RESOLVED):** readout/ensemble predicts **3** outputs — forward `(x, y, z)` one
step ahead. `x*y*z` is a **derived input feature only**, never a target. In generative mode
the upper-4 = `(x̂, ŷ, ẑ, x̂·ŷ·ẑ)` built from the ensemble's 3 predictions; in-window it's the
same 4-tuple built from real Lorenz. Same shape, only the source of the first three differs.

## 5. What this forces on the implementation

1. **Precompute the orbit once.** At setup, integrate forward from the `{1,1,1}` seed (T=0)
   up to `t@ub`, storing `S[0 .. N_c+H]`. `LorenzAttractor::trajectory()` already returns
   exactly this array. One fixed `dt`, one array — both cursors index it, so they ride the
   identical orbit by construction (no drift; the "all cursors share one `dt`" worry vanishes).
2. **Cursors read indices, not integrators.** `forward_ = S[N_c + i]`,
   `backward_ = S[N_c − i]`. The old `advance_()` `step(±dt)` — which forced one copy to run
   `-dt` and blow up — is **retired**. `center_` is no longer a stored state to reset to; it's
   the index `N_c`, and the seam case is just `S[N_c]`.
3. **Array bounds = the runtime envelope.** Right end = `ub`: forward goes generative past it,
   and generated values are **not** written back into `S`. Left end = index 0 = seed = the
   backward cursor's floor. Margin `N_c − H` (seed → `lb`) = the free-run anchor runway.
4. **`WarmupReservoir()` decouples from "find the center"** (now trivial — it's an index). It
   becomes purely: run the reservoir over the leading array region to settle its internal
   state before training/scoring.
5. **Forward source switch** at the right-limit crossing: `S[…]` → ensemble output. The
   backward cursor never switches.
6. `BoundedStep` (train) / `UnBoundedStep` (free-run) still model the two phases — they now
   advance an **index**, not an integrator.

## 5b. Data source: functional now, streams later (scope)

The cursor's contract is just **`value(position)`**. Two ways to satisfy it — JS must support
**both** eventually; we build only the first now:

| mode | how a sample is produced | status |
|------|--------------------------|--------|
| **functional** | `f(position)` on the fly (Lorenz-63 RK4) — compact, swappable, has reference results | **focus now** |
| **stream / replay** | index a provided real-world series (weather, sensors) — no closed form | future, required |

Two reasons this split is clean rather than a retrofit:

1. **Functional and stream are the *same code*.** Both modes produce an array `S[·]` that the
   cursors index; functional **computes** `S` from `f` (Lorenz RK4), stream is **handed** `S`
   (a real series). The cursor/index logic downstream is identical — stream mode is a
   different array *source*, not a different mechanism. Nothing to retrofit.
2. **Functional-first gives us a yardstick.** In stream mode the future past the known horizon
   has *no* ground truth, so generative free-run is the only option (and unscoreable against
   truth). In functional mode the true future *does* exist (evaluate `f` past `ub_`), which is
   exactly what lets us **score** the generative rollout — the Lorenz reference comparison.
   Calibration we can't get from raw real-world data. So functional-first is a measurement
   decision, not just a simplicity one.

## 6. Why it should work (intuition)

A pure generative free-run feeds the model only its own output — errors compound and it
slides off the attractor (the failure mode we keep hitting). Here, **4 of 8 inputs are
always real**. The reservoir state is continuously re-tethered to a true Lorenz signal, so
the generative half has far less room to run away. The reverse/past signal is a "free"
ground truth we happen to already know — we spend it as a stabilizer.

## 7. Open questions (let's resolve these)

- **Q1 — RESOLVED.** Mirror offsets `±i` off a shared index. Training: both shuttle inside
  `[lb_, ub_]`. Free-run: both start at center, diverge symmetrically (forward +i, backward −i).
- **Q2 — anchor runway / floor behavior.** The backward cursor floors at array index 0 (the
  seed). Runway = `N_c − H` (seed-to-`lb` margin); deeper pre-roll = longer runway vs. a bigger
  array (negligible for Lorenz). Open: behavior at the floor — stop / clamp / end the run?
- **Q3 — `x*y*z` rationale.** Why the triple product as the 4th channel (vs. `x*y` which
  appears in `ż`, or nothing)? Input feature only, or also predicted? (ties to Q4)
- **Q4 — RESOLVED.** 3 outputs (forward x,y,z); xyz derived; anchor half never a target.
- **Q5 — RESOLVED.** Center is just the array index `N_c`. The seam "re-anchor" is simply the
  `i = 0` case → `S[N_c]`; there is no center *state* to recompute or reset.
- **Q6 — ensemble coupling.** Where does κ-consensus feedback sit relative to the forward
  generative channel — does consensus replace/blend the forward 4 inputs, or ride the
  separate feedback weight block alongside them?
```
