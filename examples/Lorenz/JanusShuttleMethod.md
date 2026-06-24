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

We drive an ensemble ESN with **two cursors** that ride a single positive-time Lorenz
trajectory, moving in opposite directions from a shared **center**. Each cursor emits a
4-vector `(x, y, z, x*y*z)`, so the input is **8-D**. In training the cursors stay inside a
**reflecting window** (bounded shuttle). In free-run they break out of the window and run
away from center: the **forward** cursor walks into the *unknown future* and becomes
**generative** (fed by the ESN's own prediction), while the **reverse** cursor walks into
the *known past* and stays **ground-truth** — a real signal that anchors half the input
space and keeps the generative half from drifting off the attractor.

Everything is positive time. No `step(-dt)`, no backward integration anywhere.

## 1. The axis: three regions, one center

```
 T=0                                  center                              T = known_max
  |              KNOWN HISTORY           |          (future)                   |
  •──────────────────────────────[  lb ··· 0 ··· ub  ]─────────────────────────•───────►  t
  |        anchor lives here →           |  training    |        ← generative lives here
  |                                      |   window     |          (beyond known_max)
                                         └── reflecting bounce region ──┘

 cursor index i (displacement from center, in steps):   t(i) = t_center + i·dt
   forward cursor sample  →  state at  t_center + i·dt
   reverse cursor sample  →  state at  t_center − i·dt
   center / seam  = i = 0      window = i ∈ [lb, ub] = [−span/2, +span/2]
```

Key invariant for "positive time only": the center must sit far enough into the run that the
reverse cursor never reaches `T=0`:  `t_center ≥ (reverse travel)·dt`.

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

- **backward_ never switches.** Past `lb_` it keeps emitting `(x,y,z,x*y*z)` evaluated at
  its position — real Lorenz, the **anchor**. Half the input is always ground truth.
- **forward_ switches once,** the instant it passes `ub_`: it stops reading Lorenz and feeds
  the upper 4 channels from the **ensemble's own output**. **Generative** from there on.

"Lorenz(pos)" = the known positive-time orbit evaluated at that cursor position. For now
(functional mode, **no buffer**) backward_ gets an earlier true point by **re-integrating
FORWARD from the `{1,1,1}` seed** to that position — NOT `step(-dt)` (the exploding /
negative-time direction). A replay buffer is the *later* optimization (= stream mode). This
is the one place the current `JanusShuttle::advance_()` (`backward_.step(-dt)`) disagrees
with the concept.

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

1. **On-the-fly anchor (no buffer, for now).** The anchor's earlier points are produced by
   re-integrating FORWARD from the `{1,1,1}` seed to the cursor position. This forces a
   **pre-roll**: center sits `N_c` steps downstream of the seed (so there's real history to
   its left), and **center ≠ seed** — the constructor's `{1,1,1}` is the *seed*, and
   `WarmupReservoir()` is the pre-roll that produces the center. Pre-roll depth trades against
   anchor longevity in free-run (and against per-sample recompute cost). Buffer/
   `trajectory()` is the later optimization = stream mode.
2. **`advance_()` is rebuilt (two coupled changes).**
   - *Seam branch* — `reset(center_)` becomes correct once `center_` is the **pre-rolled**
     state (`integrate_forward(seed, N_c)`), not raw `{1,1,1}`.
   - *Else branch* — the `step(±dt)` is the bug: every leg, one copy runs `-dt` (the exploding
     direction, and not the true past). Replace with forward-from-seed:
     `forward_ = integrate_forward(seed, N_c + i)`, `backward_ = integrate_forward(seed, N_c − i)`.
     No copy ever steps `-dt`; the seam reset is just the `i = 0` case of the same formula.
3. **A source switch on the forward cursor** at the right-limit crossing: stored/live Lorenz
   → ESN feedback. The reverse cursor needs no switch.
4. `BoundedStep` (train) vs `UnBoundedStep` (free-run) already model the two phases — keep
   both, drop the backward integration inside `advance_()`.

## 5b. Data source: functional now, streams later (scope)

The cursor's contract is just **`value(position)`**. Two ways to satisfy it — JS must support
**both** eventually; we build only the first now:

| mode | how a sample is produced | status |
|------|--------------------------|--------|
| **functional** | `f(position)` on the fly (Lorenz-63 RK4) — compact, swappable, has reference results | **focus now** |
| **stream / replay** | index a provided real-world series (weather, sensors) — no closed form | future, required |

Two reasons this split is clean rather than a retrofit:

1. **The anchor half is already a stream, conceptually.** `backward_` reads the *known orbit
   at an earlier position* (not `step(-dt)`; see §3/§5) — i.e. it indexes a stored series.
   That's the stream abstraction, present from day one. Only the `forward_`/generative half
   differs between modes.
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
- **Q2 — reverse cursor hitting T=0.** If free-run runs long enough the reverse cursor
  reaches the start of known history. Stop? Clamp? Reflect? How long do we expect free-run
  to last vs. how much history we pre-roll?
- **Q3 — `x*y*z` rationale.** Why the triple product as the 4th channel (vs. `x*y` which
  appears in `ż`, or nothing)? Input feature only, or also predicted? (ties to Q4)
- **Q4 — RESOLVED.** 3 outputs (forward x,y,z); xyz derived; anchor half never a target.
- **Q5 — center placement.** Is `t_center` fixed, or does the seam re-anchor `{1,1,1}`
  during training (current `at_seam()` resets to center)? With a stored buffer, "center" is
  just an index — does the re-anchor still mean anything?
- **Q6 — ensemble coupling.** Where does κ-consensus feedback sit relative to the forward
  generative channel — does consensus replace/blend the forward 4 inputs, or ride the
  separate feedback weight block alongside them?
```
