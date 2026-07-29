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

## Public API

```text
explicit Cursor(int32_t span);

void    Reset();             // index = 0
void    Seek(int32_t index); // seat (index >= 0); stream bounds = owner
int32_t Step();              // advance; return new index
int32_t Index() const;
int32_t NextIndex() const;

bool    OOB() const;         // index > span
bool    AtStartPosition() const;  // index == 0

int32_t Span() const;
```

## How the example uses it

- **Train:** `Reset()` then teacher-force while `!OOB()`.
- **Free-run washout:** `Seek` then teacher-force W steps — edge of train
  (`span − W + 1`) for Unseen / TrainHoldout, or start of train (`0`) for
  TrainInSample — then generative free-run with prediction on the **input** bank.
