# Past-prediction head — a ground-truth-free error signal for free-run

**Status:** proposed 2026-07-09, not yet built. A concrete code change (readout
`num_outputs 3 → 6`), gated on the empirical question in §4.
**Related:** [recovery.md](recovery.md) (the relock investigation this feeds),
[JanusCursor.md](JanusCursor.md) (the dual-cursor geometry), the long-free-run TODO at
`Lorenz.h` `FREE_RUN_WINDOW_SIZE`.

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
`ReservoirStep()`. Add `S[p]` (= `std::get<1>(past_future_states)`) to the target vector.

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

## 5. Minimal build

- `readout.num_outputs: 3 → 6` (future xyz + past xyz).
- Extend `ExtractTargets` to fill `[future xyz, past xyz]`; the past target is
  `std::get<1>(past_future_states)` (the pre-injection `S[p]`).
- In `FreeRun`, score `outputs[3..5]` against the real `S[p]` and print it alongside the
  future err (so the tracking question in §4 is answerable from the first run).
- `ExtractFuturePredicted` still feeds back only `outputs[0..2]` — loop dynamics
  unchanged.

**Caveat — not a single-delta change.** The readout now splits capacity across 6 outputs
instead of 3, so future-head VPT/RMSE will shift slightly vs current runs. If the future
head must stay bit-identical, the alternative is a **separate past-only readout** (more
code, future head untouched). For a first experiment, start with the 6-output version
and eat the small shift.
