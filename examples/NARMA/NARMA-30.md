# NARMA-30 — Results

Parent walkthrough / recurrence / protocol: [NARMA.md](NARMA.md).

**tanh-wrapped** fixed coeffs (`NARMA_TANH_WRAP=1`) for honest order-scaling.
Rough literature bands for order 30 exist but are poorly standardized — see
[NARMA.md](NARMA.md#reading-the-results).

---

## Spotlight — best-3 seeds (NRMSE 0.0570)

Top three of the 10-seed pool below (lowest NRMSE). Full pool: mean 0.0601,
std 0.0022, range 0.0570–0.0636.

| | NRMSE | vs literature |
|--|------:|---------------|
| **HypercubeESN (best seed)** | **0.0570** | **~5.3× lower error** than strong floor (0.30); **~7.0×** vs good floor (0.40); **~10.5×** vs good ceiling (0.60) |
| Best-3 mean | **0.0576** | **~5.2× lower error** than strong floor (0.30) |
| Best-3 std | 0.0006 | — |
| Best-3 range | 0.0570–0.0582 | — |
| Lit. strong floor / good band | 0.30 / 0.40–0.60 | reference only (poorly standardized) |

**Best test NRMSE 0.0570** (R² = 0.9968, res seed 73896) on **tanh-wrapped
NARMA-30**. Best-3 mean **0.0576** (std 0.0006). About **five times below**
the bottom of the rough literature **strong / large-N** band (0.30–0.50); see
caveats in [NARMA.md](NARMA.md#reading-the-results).

### Best-3 seeds

| res seed | NRMSE | R² | train (s) |
|---------:|------:|---:|----------:|
| **73896** | **0.0570** | **0.9968** | 1512 |
| 73898 | 0.0577 | 0.9967 | 1561 |
| 73897 | 0.0582 | 0.9966 | 1529 |

| Knob | Value |
|------|--------|
| Variant | tanh-wrapped (α=0.3, β=0.05, γ=1.5, δ=0.1) |
| Reservoir | DIM=10 (N=1024), M=`history_depth`=16, FSF **off** |
| Seed | best-3: 73896, 73898, 73897 (from 10-seed pool) |
| Series | warmup 300 · collect 32000 (train 25600 / test 6400) |
| Drive | sr 0.99 · leak 1 · input_scaling 0.03 |
| Readout | 600 epochs, batch 128, lr 0.0015 (cosine, floor 7.5e-06) |
| HCNN | Conv(1→16, TANH) → MaxPool → Linear(32768→1) · 32993 params |

Open-loop baseline from Release `NARMA.exe`. Train-set target mean 0.9490
(identical series across seeds). Wall time ~24–27 min/seed on the collect
machine.

---

## Multi-seed summary (M=16, 10 seeds)

Two batches, same op-point / series / M; reservoir seeds only differ.

| Metric | All 10 | Batch A (n=5) | Batch B (n=5) |
|--------|------:|-------------:|-------------:|
| Seeds | see table | 73896–73900 | 147792 … 443400 |
| mean NRMSE | **0.0601** | 0.0590 | 0.0612 |
| std | 0.0022 | 0.0023 | 0.0015 |
| min | **0.0570** | 0.0570 | 0.0597 |
| max | 0.0636 | 0.0628 | 0.0636 |
| M | 16 | 16 | 16 |
| collect | 32000 | 32000 | 32000 |

### Per-seed NRMSE

| res seed | batch | NRMSE | R² | train (s) |
|---------:|------:|------:|---:|----------:|
| **73896** | A | **0.0570** | **0.9968** | 1512 |
| 73897 | A | 0.0582 | 0.9966 | 1529 |
| 73898 | A | 0.0577 | 0.9967 | 1561 |
| 73899 | A | 0.0628 | 0.9961 | 1461 |
| 73900 | A | 0.0593 | 0.9965 | 1552 |
| 147792 | B | 0.0601 | 0.9964 | 1630 |
| 221691 | B | 0.0597 | 0.9964 | 1468 |
| 295592 | B | 0.0616 | 0.9962 | 1486 |
| 369495 | B | 0.0636 | 0.9960 | 1538 |
| 443400 | B | 0.0609 | 0.9963 | 1493 |

---

## M sweep

Single depth only for this campaign:

| M | n seeds | mean NRMSE | std | min | max |
|--:|--------:|-----------:|----:|----:|----:|
| 16 | 10 | 0.0601 | 0.0022 | 0.0570 | 0.0636 |

---

## Notes

- Same drive/readout op-point as the NARMA-50 campaign (sr 0.99, is 0.03,
  collect 32000, same seed batches); **M=16** here vs **M=32** for N50
  ([NARMA-50.md](NARMA-50.md)).
- Seed spread stays very tight after doubling the pool (std ~0.002); all ten
  seeds under 0.064 NRMSE. Batch A mean (0.0590) is slightly better than batch B
  (0.0612); combined mean is **0.0601**. Best three all come from batch A.
