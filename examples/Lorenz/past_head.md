# Past-prediction head — a ground-truth-free error signal for free-run

**Status:** BUILT, TESTED, then **REMOVED 2026-07-09** — §4 answered, verdict below.
The design and reasoning are kept here as the record; the code is gone (it only slowed
training while giving no usable signal).
**Related:** [recovery.md](recovery.md) (the relock investigation this feeds),
[JanusCursor.md](JanusCursor.md) (the dual-cursor geometry), the long-free-run TODO at
`Lorenz.h` `FREE_RUN_WINDOW_SIZE`.

## 0. Verdict (2026-07-09) — it DECOUPLES; removed

Built as a standalone diagnostic `Readout` (zero-delta to the future path, per §5), run
on seed 13649419. Result: **past-head error decouples from future-head error, decisively
and over long horizons.** At SR=0.90 over a 2000-step / 36-Lyapunov-time free-run, past
err stayed pinned at ~0.005 (≈ its training floor) for the ENTIRE rollout while future
err bounced around the climatological floor (0.1–0.9); past RMSE 0.006 vs future RMSE
0.44. Even the *derivative* of past err showed no response to future spikes (a ±0.0008
band, nothing at the step-175 divergence).

- **Not the hoped-for future-health proxy** (the "decouple" branch of §4 won). The past
  head's *input* — the anchored real past on the input port — never diverges, so it
  predicts `S[p]` at training accuracy regardless of what the future feedback does.
- **What it *does* prove** (kept as a finding): the shared reservoir state stays on the
  true manifold throughout free-run — it never collapses. So a future divergence here is
  an *extrapolation* limit (decorrelated anchor carries no future phase), NOT reservoir-
  state corruption. That distinguishes "reservoir healthy, future unpredictable" (what we
  saw) from "reservoir collapsed" — a real but narrow diagnostic, not worth the training
  cost of a second readout trained every step.
- **Removed** from `Lorenz.{h,cpp}` and the `ESN::LastReadoutInput()` helper it used.

Everything below is the original proposal, retained for context.

## 1. The problem it solves

In free-run we can only score the **future** head against held-out truth for as long as
the precomputed future runway lasts — `E = FREE_RUN_WINDOW_SIZE = 500` samples, ~9
Lyapunov times. After that (and in any real deployment, where there is no precomputed
future at all) we have **no error signal**. We can't answer the free-run question we
actually care about — *do we keep breaking and re-locking, or only early on?* — without
growing the stream.

But the **past cursor is always driven by real data**: it walks backward through the
anchor history (center ~10500 → 0), so ~10,500 real samples sit ahead of it — **~20×
the future runway**. If the readout also predicts the past sample, we can score that
prediction against known truth every step, indefinitely.

## 2. The two-head layout

Add a second readout head that predicts the past sample. Both heads read the **same**
reservoir state; only the future prediction is ever fed back.

```
        input port (ALWAYS real)                 feedback port (real → own pred in free-run)
               S[p] ─┐                               ┌─ S[f-1] / own prediction
                     ▼                               ▼
             ┌──────────────── reservoir: shared state x(t) ────────────────┐
             └─────────┬───────────────────────────────────────┬───────────┘
                       ▼                                        ▼
                 PAST head → Ŝ[p]                         FUTURE head → Ŝ[f]
             score vs real S[p]  ← ALWAYS available     score vs S[f]  ← runway only (500)
```

The past head is **pure diagnostic**: the input port keeps receiving the real `S[p]`,
never the prediction, so the anchor stays real and the reservoir dynamics are unchanged.

## 3. The one trap: horizon-1 alignment, or it reads ~0

`S[p]` is injected into the input port every step. If the past head predicts `S[p]` from
a state that has **already absorbed** `S[p]`, it's input reconstruction — RMSE ≈ 0, no
information. It must be **horizon-1 aligned**, exactly like the future head: predict
`S[p]` at the **pre-injection** state (which has only seen the past through `S[p+1]`),
*before* `ReservoirStep` injects it. Then it's a genuine one-step prediction of the
backward-running past sequence — symmetric to `S[f-1] → S[f]`.

Timing is free: the training/free-run loops already call `Predict()` before
`ReservoirStep()`. Add `S[p]` (= `past_future_states.past`) to the target vector.

## 4. The question that decides if it's useful

**Does past-head error TRACK future-head error?** Two competing mechanisms:

- **Tracks (useful).** The two heads share one reservoir state. When the future
  self-feedback drifts and injects a bad value, it corrupts the shared state → the past
  head's prediction of the (real) next past also degrades → both spike together, both
  heal on re-lock. Then past-head RMSE is a **deployable, ground-truth-free proxy for
  reservoir health** — computable when no future truth exists.
- **Decouples (less useful).** The always-real anchor keeps the state "past-consistent"
  even while the future has diverged → past head stays green while the future is red.

We can't tell which without building it and **plotting past-err vs future-err across a
free-run**. Given the recovery findings (the anchor demonstrably re-syncs the shared
state), the bet is *tracks*, at least during re-lock episodes — but that must be
measured, not assumed.

## 5. Minimal build — a SEPARATE diagnostic readout (zero-delta)

The reservoir is fixed and the past head never feeds back, so a past head that reads the
**same** reservoir states with its **own** parameters leaves the future head and the
entire free-run trajectory **bit-identical**. This is the whole point of a diagnostic:
it measures without disturbing. So the past head is a **second `Readout`** (the class
already exists — `TrainStep` / `PredictRaw` / etc.), not extra outputs on the existing one:

- Instantiate a second `Readout` (`num_outputs = 3`, past xyz) fed the same reservoir
  state each step — via an ESN state accessor, or by housing it inside `ESN` alongside
  the existing readout.
- Each training step: `TrainStep` it against the past target
  `past_future_states.past` — the **pre-injection** `S[p]` (§3), read at the
  same pre-`ReservoirStep` state as the future head.
- In `FreeRun`: `PredictRaw` its past output, score vs the real `S[p]`, print alongside
  the future err (so the §4 tracking question is answerable from the first run).

Because none of this touches the future readout's weights or the fed-back
`ExtractFuturePredicted(outputs[0..2])`, the future-head VPT/RMSE are **unchanged** — a
true single-delta addition to the current results.

**Do NOT just bump `num_outputs 3 → 6` on the existing readout.** The data path is
`state → Embed → [Conv+Pool]×L → Flatten → Linear → output` (Readout.h): one shared conv
trunk. Training all six outputs under one loss lets past-head gradients flow into that
shared trunk and shift the future features — coupling your diagnostic to the very thing
it's supposed to measure, for no benefit. Only choose the shared-trunk form if you
actually want the two heads to share a representation (not the case for a health signal).
