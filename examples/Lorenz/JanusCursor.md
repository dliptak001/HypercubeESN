# JanusCursor

| Item | Location |
|------|----------|
| Source | `JanusCursor.h` (header-only) |
| Result type | `JanusCursorResult` = `std::pair<int32_t, int32_t>` → `{past_index, future_index}` |
| Primary consumer | `LorenzDatastream` (public inheritance); harness protocol in [`README.md`](README.md) |

This document describes what `JanusCursor` does. It holds **no stream data** — only two integer indices and the window geometry that defines them. The owner maps those indices into its own buffer.

---

## 1. Purpose

A pair of **counter-moving cursors** over one linear index space, used to half-anchor a free-run:

- **Past cursor** — reads real history (the **anchor**); may walk below the training window into a past runway.
- **Future cursor** — tracks the prediction horizon; walking past the window’s upper edge is the generative signal (`OOB()`).

Both cursors share one window `[lb, ub]` centered at `center_index` with width parameter `span`. `Reset()` seats them at **opposite edges**; each `Step()` moves them in **opposite directions** so they sweep toward and then past each other.

```text
stream:  0 ....... lb =========== center =========== ub ....... N
                    ^future                          ^past       (after Reset)
         future ──────────────────>          <────────────── past  (each Step)
```

The class is **system-agnostic**: any array addressable by `int32_t` index can sit under the dual indices. Lorenz-63 is only the in-tree instantiation (see `README.md`).

### Past cursor = continuous drive, not only a lag feature

The past cursor’s job is **dual**:

1. **Horizon / VPT** — real history on the input port can delay first threshold-crossing.  
2. **Lock / re-lock** — continuous real past is a **drive** signal. While free-run is generative on the future port, the reservoir is still driven by on-attractor truth through the past port, so free-run error need not climb to the climatological floor and stay there: it can spike and **recover** deep into the window.

In project language that re-locking is **generalized synchronization (GS)**: a reservoir continuously driven by a real signal has its state pulled toward a physically consistent region, so the readout can re-lock after a deviation. VPT (first crossing) is often **blind** to this — two arms can share similar VPT while free-run RMSE and late re-collapse differ. Score both.

House notes (do not lose again): former `examples/Lorenz/recovery.md` (pruned; recover from git if needed — e.g. `05bd00e`, `ca86a76`, `6182a84`) documented anchor dose-response (Goldilocks `INPUT_SCALING`), CLE condition for sync, and prior art (reservoir observers / GS forecasting: Lu et al. 2017; Chaos 2019/2024). **Forward-only** ablation (`config::FORWARD_ONLY`) zeros past every step and removes this GS drive by design — see [`TODO_forward_only_ablation.md`](TODO_forward_only_ablation.md).

During free-run the two cursors sit far apart on the stream (often tens–hundreds of Lyapunov times). The anchor supplies real **climate** / manifold grounding, not a leak of the scored future’s phase — re-lock must still come from learned dynamics + feedback, not from past handing over the answer.

---

## 2. Window geometry

Constructed as:

```text
lb_ = center_index − span / 2     // integer division
ub_ = center_index + span / 2
```

| Invariant | Code |
|-----------|------|
| `span > 0` | else `std::out_of_range` |
| `center_index > span / 2` | else `std::out_of_range` (ensures `lb_ >= 0` under non-negative `center_index`) |

**Integer half-span.** `span / 2` truncates. The index gap is:

```text
ub − lb = 2 · (span / 2)
```

- Even `span`: `ub − lb == span` (symmetric window).
- Odd `span`: `ub − lb == span − 1` (one unit short of `span`).

The class does **not** know the stream length `N`. Bounds relative to a real buffer (`ub ≤ N`, eval runway past `ub`, past runway `[0, lb)`) are the **owner’s** responsibility. `LorenzDatastream` validates window vs. `stream_length` at construction.

Suggested layout when an owner places the window on a stream (as Lorenz does):

```text
 array index n:   0 ······· lb ····· center ····· ub ········· stream end
                  │         │           │           │            │
                 seed   train edge   midpoint   train edge   eval / free-run
                        (lb)        (center)      (ub)         tail

 window [lb, ub]     — training / washout region the cursors start on
 [0, lb)             — past free-run runway (anchor history past may enter)
 (ub, end]           — prediction / eval runway (future is generative here)
```

---

## 3. Nested cursors

`PastCursor` and `FutureCursor` are **private nested types**. Callers only see the public `JanusCursor` API. Their rules:

| | PastCursor | FutureCursor |
|---|------------|--------------|
| `Reset()` | `idx_ = ub_` | `idx_ = lb_` |
| `Step()` | `idx_ -= 1`; returns new index | `idx_ += 1`; returns new index |
| Nested `OOB()` | `idx_ < lb_` | `idx_ > ub_` |
| `index()` | current | current |
| `next_index()` | `idx_ − 1` (no mutate) | `idx_ + 1` (no mutate) |
| Nested `AtStartPosition()` | `idx_ == ub_` | `idx_ == lb_` |

Both nested constructors recompute the same `lb_` / `ub_` from `(span, center_index)` and apply the same validation, then call `Reset()`.

**Past below `lb` is intentional.** The past cursor may leave the window into `[0, lb)` (and below, if the owner allows). That is real history for anchoring, not a public error. Nested `PastCursor::OOB()` tracks “below `lb`” but is **not** exposed on `JanusCursor`.

**Future past `ub` is the public generative signal.** Nested `FutureCursor::OOB()` is what public `OOB()` reports.

---

## 4. Public API

```cpp
JanusCursor(int32_t span, int32_t center_index);

void Reset();
JanusCursorResult Step();           // advance both; return new {past, future}
JanusCursorResult Indices() const;  // current {past, future}
JanusCursorResult NextIndices() const; // peek: {past−1, future+1}, no mutate

bool  OOB() const;                  // future_index > ub  only
float Distance() const;             // (future − past) / span  as float
bool  AtStartPosition() const;      // past at ub only (proxy for “just Reset”)
```

### Semantics detail

| Method | Behavior |
|--------|----------|
| **Ctor** | Builds both nested cursors (each validates and `Reset()`s). Stores `span_` for `Distance()`. |
| **`Reset()`** | Past → `ub`, future → `lb`. Idempotent seating at opposite edges. |
| **`Step()`** | Calls both nested `Step()`s; returns the **post-advance** index pair. Does not throw. |
| **`Indices()`** | Current pair without moving. |
| **`NextIndices()`** | Peeks one step ahead without moving. Safe to call at any time; does not clamp. |
| **`OOB()`** | **`future_cursor_.OOB()` only** — true iff `future_index > ub`. Past position is ignored. |
| **`Distance()`** | `1.0f * (future_index − past_index) / span_` |
| **`AtStartPosition()`** | **`past_cursor_.AtStartPosition()` only** — true iff past is at `ub`. Does not check the future cursor. |

### `Distance()` order parameter

```text
Distance = (future_index − past_index) / span
```

After `Reset()` (even `span`):

```text
(lb − ub) / span = −1
```

| Phase | past | future | Distance (even span) |
|-------|------|--------|----------------------|
| Just `Reset()` | `ub` | `lb` | `−1` (fully apart) |
| Center crossing | ~`center` | ~`center` | `~0` |
| Mirror extreme | `lb` | `ub` | `+1` |

Timeline after `k` successful `Step()`s from `Reset()`:

```text
 past   = ub − k
 future = lb + k
 meet when k = (ub − lb) / 2   (= span/2 when span is even)
```

```text
 k (steps after Reset)     0              ~span/2            ~span
 past index              ub  ──────────►  center  ─────────►  lb
 future index            lb  ──────────►  center  ─────────►  ub
 |future − past|        span               ~0               span
 Distance()              −1                 ~0               +1
```

Further steps (past `< lb`, future `> ub`) push `|Distance|` **beyond** 1 — the class does not clamp. Training loops typically stop at `OOB()` (future just past `ub`).

For **odd** `span`, `|Distance|` at the extremes is `(span−1)/span`, not exactly 1, because `ub − lb = span − 1`.

---

## 5. What the class does *not* do

These are deliberate non-features (owner / harness responsibilities):

| Not in `JanusCursor` | Where it lives |
|----------------------|----------------|
| Stream storage, integration, normalization | Owner buffer (`LorenzDatastream`) |
| Stream-length / runway validation | Owner ctor checks |
| Multi-epoch presentation | Caller: `Reset()` + loop until `OOB()` |
| Reflecting / triangle-wave re-sweeps | Not implemented — each pass is one-way |
| Throwing when past underruns index 0 | Owner may throw (`LorenzDatastream::Step`) |
| Public past-below-`lb` flag | Nested only; public `OOB` is future-only |
| Port packing, teacher forcing, free-run policy | `Lorenz` harness (`README.md`) |

---

## 6. Owner contract (how a consumer should use it)

Typical loop:

```text
JanusCursor cursors(span, center);
// owner has buffer S[0 .. N]

cursors.Reset();
while (!cursors.OOB()) {
    auto [p, f] = cursors.Indices();
    // read S[p], S[f]  — both in-window while !OOB() at loop entry
    // ... train / washout ...
    cursors.Step();
}
// now future_index > ub  — generative region for the future half
// past may still be in-window or already below lb; keep reading S[past]
// as long as past >= 0 (or whatever the owner allows)
```

`Step()` always advances **both** indices, including after `OOB()` is true. Owners that must not let past fall below 0 should check `Indices().first` **before** calling `Step()` again (as `Lorenz::FreeRun` does).

`NextIndices()` is a pure peek; nothing in-tree currently depends on it for control flow, but it is part of the public surface.

---

## 7. Code map (this header only)

```text
JanusCursorResult = pair<int32_t, int32_t>   // {past, future}

JanusCursor
  span_
  PastCursor   lb_, ub_, idx_     Reset→ub  Step→−1  OOB: idx < lb
  FutureCursor lb_, ub_, idx_     Reset→lb  Step→+1  OOB: idx > ub

  Reset / Step / Indices / NextIndices
  OOB          → FutureCursor::OOB
  Distance     → (future − past) / span_
  AtStartPosition → PastCursor::AtStartPosition
```

---

## 8. Related in-tree

| Document / file | Covers |
|-----------------|--------|
| [`README.md`](README.md) | Full Lorenz experiment: stream layout, 8-channel ports, train / free-run, VPT scoring, how to build and run |
| `LorenzDatastream.{h,cpp}` | RK4 orbit, normalize to float `[-1,1]`, inherits `JanusCursor`, maps indices → samples |
| `Lorenz.{h,cpp}` | ESN ports, `Train` / `FreeRun`, multi-seed survey `main` |
| `docs/LorenzFreeRun.md` | Older A(x)-vs-tanh free-run campaign notes — **not** a description of this cursor (stale relative to the Janus harness) |

---

## 9. What this method is (and is not)

**Is:**

- Dual-cursor index motion over one forward index space
- Opposite-ends seating + counter-walk (one-way pass per `Reset`)
- Future-over-`ub` as the sole public out-of-window signal
- A geometry primitive for half-anchored free-run (past may leave the window into history)

**Is not:**

- A data source (no samples, no normalization)
