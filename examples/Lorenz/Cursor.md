# Cursor

| | |
|--|--|
| Source | `Cursor.h` (header-only) |
| Role | Single forward index over a training window |

Holds **no stream data** — only an integer index and window geometry.
`LorenzDatastream` inherits Cursor and maps the index into its orbit buffer.

## Geometry

Training window is **`[0, span]` inclusive** (start always 0). The class does
**not** know stream length `N`.

```text
stream:  0 ======================= span ....... N
         ^ Reset                    | train end
         ──────────────────────────>  each Step()
                                      OOB when index > span
```

| Parameter | Meaning |
|-----------|---------|
| `span` | Last in-window train index; `OOB` when `index > span` |

`Reset()` seats at 0. Each `Step()` increments by one. Free-run / eval runway
is `index > span` up to the owner's stream end.

## Public API

```text
explicit Cursor(int32_t span);

void    Reset();
int32_t Step();              // advance; return new index
int32_t Index() const;
int32_t NextIndex() const;

bool    OOB() const;         // index > span
bool    AtStartPosition() const;  // index == 0

int32_t Span() const;
```

## How the example uses it

In `examples/Lorenz`, train and washout run while `!OOB()` with teacher-forced
samples at the current index. Free-run continues on the eval runway with the
model's prediction fed on the **input** bank (`num_external_feedback_channels = 0`).
