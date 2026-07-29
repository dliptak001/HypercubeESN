# Cursor

| | |
|--|--|
| Source | `Cursor.h` (header-only) |
| Role | Single forward index over a training window |

Holds **no stream data** — only an integer index and window geometry. The owner
(`LorenzDatastream`) maps the index into its buffer.

## Geometry

```text
stream:  0 ....... lb ======================= ub ....... N
                     ^ Reset                    | train end
         ──────────────────────────────────────>  each Step()
                                                  OOB when index > ub
```

| Parameter | Meaning |
|-----------|---------|
| `span` | Training-window width (`ub = start_index + span`) |
| `start_index` | First index of the training window (`lb`) |

`Reset()` seats at `lb`. Each `Step()` increments by one. `OOB()` is true once
`index > ub` — the free-run / evaluation runway.

The class does **not** know stream length `N`. Eval runway bounds are the owner's job.

## Public API

```text
Cursor(int32_t span, int32_t start_index);

void    Reset();
int32_t Step();              // advance; return new index
int32_t Index() const;
int32_t NextIndex() const;

bool    OOB() const;         // index > ub
bool    AtStartPosition() const;  // index == lb

int32_t Span() const;
int32_t LowerBound() const;
int32_t UpperBound() const;
```

## How the example uses it

In `examples/Lorenz`, train and washout run while `!OOB()` with teacher-forced
samples at the current index. Free-run continues on the eval runway with the
model's prediction fed on the **input** bank (`num_external_feedback_channels = 0`).
