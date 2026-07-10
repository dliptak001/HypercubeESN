# Free-run error *recovery* — is it real, and what does it mean?

**Status:** central question **ANSWERED 2026-07-09** (anchor dose-response below) —
recovery is the anchor, but only in a tuned band (interior optimum near
`INPUT_SCALING=0.01`; too little or too much → no relock). Two follow-ups remain
(re-sync metric, initial-condition generality); the observer-floor idea was killed as
ill-posed (see "Next step (a)").
**Context:** single-run, seed 13649419, `in_scale=0.01`, `feedback_scaling=0.04`,
`leak=1.0`, `EPOCHS=100`, shared-scale normalization, LR anneal knee at 75%.

## The observation

In the generative free-run the per-step (channel-RMS) error does **not** climb
monotonically. It spikes hard and then *recovers*:

```
free-run  125  ( 2.26 lt)  err 0.170140
free-run  150  ( 2.72 lt)  err 0.030254
free-run  175  ( 3.17 lt)  err 0.746380   <-- big spike
free-run  200  ( 3.62 lt)  err 0.126857   <-- recovered
free-run  225  ( 4.08 lt)  err 0.130263
free-run  300  ( 5.43 lt)  err 0.072210   <-- back near the noise floor
free-run  350  ( 6.34 lt)  err 0.044449
free-run  450  ( 8.15 lt)  err 0.753278   <-- spikes again late
free-run  475  ( 8.60 lt)  err 0.855939
free-run  500  ( 9.06 lt)  err 0.434962
```

Headline for this run: `VPT 169 steps (3.06 lt)  free-run RMSE 0.258925 (500 steps)`.

The question: **is it typical Lorenz-ESN behaviour to recover this cleanly from a
large deviation?**

## Short answer

- For a **standard autonomous ESN free-run** (reservoir driven *only* by its own
  fed-back output): **no.** Clean, sustained recovery from a 0.75 spike back to
  ~0.13 would be atypical — almost a red flag.
- For **this harness** (Janus anchored drive): **yes, and it's the mechanism we
  built.** The recovery is the anchor doing its job, not an artifact — with one
  caveat we still need to close (see "Next step").

## Why a textbook free-run does NOT recover

In the canonical setup the reservoir's only input is its own output. Once the
generated trajectory decorrelates from the true orbit — which it must, at the
Lyapunov rate under chaos — nothing couples them any more. A good model keeps the
trajectory *on* the attractor (right lobes, right statistics) but at the **wrong
phase**. Error grows to the *climatological floor* (the RMS distance between two
random points on the attractor, ~0.3–0.5 in our normalized units) and stays there.

```
err
0.5│                    ┌──────────────  climatological floor, STAYS here
   │                 ╱‾‾
0.3│          ┌─────╯
   │      ╱‾‾╯
0.0│──────╯
   └────────────────────────────────── t
        VPT      decorrelated for good
```

There is no restoring force, so no dips back down. If we saw this trace in a pure
autonomous run we'd go bug-hunting.

## Why THIS run recovers — the Janus anchor

Our free-run is a **driven** reservoir, not a free one. Two ports:

```
input port     <- REAL past block          (in_scale = 0.01)   anchored to truth
feedback port  <- model's OWN prediction    (feedback = 0.04)   self-generated
```

Every step the reservoir is still being driven by *real, on-attractor* data through
the anchored past block. A reservoir continuously driven by a real signal exhibits
**generalized synchronization**: its state is pulled back toward a physically
consistent region, which lets the readout re-lock after a deviation. The dips at
175→200 and around 350 are that re-locking.

### Key geometry: the anchor is not leaking the answer

During free-run the two cursors sit far apart on the stream:

```
  index:   0        500        10500              20500      21000
           |         |           |                  |          |
           seed    train-lb   train-center       train-ub   stream-end
                                 ^                    ^
                          past cursor            future cursor
                          ~10500 -> 10000        ~20501 -> 21000
                          (walks BACKWARD)       (walks FORWARD, scored)
```

The past cursor (real drive) and the future cursor (scored) are ~10000 indices ≈
**~200 Lyapunov times** apart — fully decorrelated on the attractor. So the anchor
supplies real *climate* (a sane on-attractor reference frame) but carries **no phase
information about the specific future being scored**. The re-lock to the true orbit
therefore has to come from the learned dynamics + feedback loop, not from the anchor
handing over the phase. That's the encouraging interpretation.

### The spikes are probably lobe-transition mistimings

The bursts (175, 450, 475) most likely correspond to the model hesitating at a wing
switch — the sensitive saddle region between the two lobes — then getting
re-stabilized by the anchor. Error concentrating at lobe switches is very
characteristic of Lorenz prediction. (Unconfirmed — would need to overlay the spike
steps against actual lobe-crossing events in the true orbit.)

## Two caveats for interpreting the numbers

1. **VPT undersells this run.** A single first-crossing at 169 steps treats the run
   as "done" at the first 0.30 spike, but the rollout keeps re-locking well past
   that. If re-synchronization is the property we care about, VPT-to-first-crossing
   isn't capturing it — a "fraction of steps under threshold" or a burst count would
   describe it better.
2. **Not comparable to published autonomous-ESN VPTs.** Those are free rollouts;
   ours is anchor-driven. Different regime — don't benchmark our ~3 lt against
   textbook autonomous numbers.

## RESULT — anchor dose-response (2026-07-09)

Swept the anchor via `INPUT_SCALING` (the past block's gain on the input port),
everything else fixed, same seed 13649419, `feedback_scaling=0.04`. The relationship
is **non-monotonic — a Goldilocks knob with an interior optimum**, NOT "more anchor =
more recovery":

| `INPUT_SCALING` | final train RMSE (ep99) | VPT | free-run RMSE | relock? |
|---|---|---|---|---|
| 0.00 (none)   | **0.000606** | 3.15 lt | 0.446120 | no — textbook saturation |
| 0.01 (gentle) | 0.001030 | 3.06 lt | **0.258925** | **YES** → 0.127, 0.072, 0.044 |
| 0.04 (strong) | 0.001797 | **2.14 lt** | 0.456741 | no — diverges early (step 118) |

Two clean trends:
- **Train RMSE rises monotonically** with anchor strength (0.0006 → 0.0010 → 0.0018).
- **Free-run RMSE is U-shaped**, minimum at 0.01. Both endpoints sit at the
  climatological floor (~0.45); only the middle re-locks.

```
free-run RMSE
 0.45 ●________________________________●     0.00 and 0.04: no relock
       \                              /
        \                            /
 0.26   \__________●______________/          0.01: sweet spot
         0.00     0.01           0.04   INPUT_SCALING
```

**Why (follows directly from the FEEDBACK=0 finding).** During TRAINING the past
block is *decorrelated* from the future target (cursors sweep in from opposite ends —
see "Next step (a)"). So `INPUT_SCALING` injects a **decorrelated distractor** into the
reservoir during training — which is exactly why train RMSE rises monotonically with
it (stronger off-phase drive → harder to fit the one-step map). The anchor is thus
double-edged:

- **Too little (0.00):** no real-manifold drive → nothing re-syncs the state after a
  divergence → textbook autonomous saturation at the climatological floor.
- **Just right (0.01):** weak enough not to corrupt the learned map or dominate the
  state, strong enough to nudge the state back onto the true manifold → relock. Since
  the anchor is ~200 lt decorrelated from the scored future it isn't leaking phase —
  it grounds the state on the true attractor instead of letting the feedback loop
  settle onto its own wrong pseudo-attractor.
- **Too much (0.04):** the decorrelated drive BOTH wrecks the base map (train RMSE up,
  VPT collapses 3.1 → 2.1 lt, so free-run crosses threshold at step 118 vs 169) AND
  pushes the state off the future's phase every step → no relock, worse everywhere.

Relock lives in a narrow band where the anchor grounds the state without overwhelming
it. **Caveat: n=1 orbit/seed so far** — the optimum near 0.01 could be partly
seed-luck; confirm with a finer sweep (0.005 / 0.01 / 0.02) over several seeds / x0s
(ties into follow-up (c)).

One more finding, now explained by the above:

- **VPT is blind to the recovery benefit.** At 0.00 vs 0.01, VPT is ~identical (3.15
  vs 3.06 lt) — the anchor does NOT delay the first divergence, it enables *recovery
  afterward*. VPT-to-first-crossing cannot see that; RMSE can (0.446 vs 0.259). (At
  0.04 VPT *does* move, but for the other reason — the base map is degraded, not
  because recovery changed.) Strong motivation for the re-sync-aware metric below.

## Next step (resume here 2026-07-10)

The central "is recovery real / what causes it" question is settled (anchor). Two
follow-ups remain.

### (a) — DEAD END: the "observer floor" is ill-posed (found 2026-07-09)

An earlier plan was to zero `FEEDBACK_SCALING` to get an "observer floor" — anchor on,
prediction never re-enters the state — and measure how much the generative loop adds.
**This does not work, and the reason is instructive.** Running `INPUT_SCALING=0.4`,
`FEEDBACK_SCALING=0` leaves training stuck at the climatological floor (train RMSE
~0.32, never descends).

Why: in the Janus geometry the two cursors start at OPPOSITE ends of the training
window and sweep toward each other (`PastCursor` resets to `ub` and decrements;
`FutureCursor` resets to `lb` and increments — JanusCursor.h). The training target is
the FUTURE sample `S[f]`, but the past block on the input port is a *decorrelated*
point on the attractor — up to ~360 lt away, coinciding with the target only at the
center crossing:

```
              k=0            k=10000(center)      k=20000
  past   :  20500 ───────────► 10500 ◄─────────── 500     (input port)
  future :    500 ───────────► 10500 ───────────► 20500   (feedback port; target = S[future])
  |f-p|  :  20000     ...        0       ...      20000    samples  (~360 lt .. 0 .. ~360 lt)
```

The predictive signal during training lives in the FEEDBACK port: with `feedback>0`
the reservoir absorbs the future stream's own one-step history `S[f-1]`, and `S[f-1] ->
S[f]` is a learnable autoregressive step (this is why the feedback-on runs trained to
~0.001). Zero the feedback and the only live channel is the decorrelated past, so the
readout can do no better than predict the mean on ~all 19,501 steps → pinned at the
climatological floor. `FEEDBACK=0` is therefore an ill-posed *training* task, NOT a
clean observer. A genuine observer floor can't come from one scalar — you'd need
feedback ON during training with its free-run contribution suppressed separately,
which is a different (train/test-mismatched) construction. **Consequence:** the anchor
is a *stabilizer during free-run*, not the training signal; the feedback port is what
makes the task learnable. Don't re-attempt the `FEEDBACK=0` observer floor.

### (b) Re-sync-aware metric. VPT-to-first-crossing demonstrably undersells this
harness (finding 1). Add a companion metric — fraction of steps under threshold, or a
burst count — so the survey stats reflect the recovery behavior the anchor buys.

### (c) Does re-lock generalize across initial conditions?

The anchor A/B so far is a
single orbit (`INITIAL_LORENZ_STATE = {0.65, 0.75, 0.1}`, seed 13649419). The recovery
may be specific to this trajectory's particular spike/lobe structure. Re-run the anchor
ON/OFF contrast across several `INITIAL_LORENZ_STATE`s (different x0 → a genuinely
different orbit and different eval tail, not just a different reservoir seed) and
confirm the anchor-driven re-lock holds broadly rather than being an artifact of this
one orbit. Watch whether spike *locations* and recovery *depth* track the orbit's lobe
transitions.
