# NARMA-30 — Results

Parent walkthrough / recurrence / protocol: [NARMA.md](NARMA.md).

**tanh-wrapped** fixed coeffs (`NARMA_TANH_WRAP=1`) for honest order-scaling.
Rough literature bands for order 30 exist but are poorly standardized — see
[NARMA.md](NARMA.md#reading-the-results).

---

## Spotlight — best seed NRMSE 0.0570

| | NRMSE | vs this run |
|--|------:|------------|
| **HypercubeESN (best seed)** | **0.0570** | — |
| HypercubeESN (5-seed mean) | 0.0590 | — |
| Literature “strong / large-N” band (floor) | 0.30 | **~5.3× higher error** |
| Literature “good” band (floor) | 0.40 | **~7.0× higher error** |
| Literature “good” band (ceiling) | 0.60 | **~10.5× higher error** |

**Best test NRMSE 0.0570** (R² = 0.9968, res seed 73896) on **tanh-wrapped
NARMA-30**. Five-seed mean **0.0590** (std 0.0023). About **five times below**
the bottom of the rough literature **strong / large-N** band (0.30–0.50); see
caveats in [NARMA.md](NARMA.md#reading-the-results).

| Knob | Value |
|------|--------|
| Variant | tanh-wrapped (α=0.3, β=0.05, γ=1.5, δ=0.1) |
| Reservoir | DIM=10 (N=1024), M=`history_depth`=16, FSF **off** |
| Seed | best res 73896; sweep 73896–73900 |
| Series | warmup 300 · collect 32000 (train 25600 / test 6400) |
| Drive | sr 0.99 · leak 1 · input_scaling 0.03 |
| Readout | 600 epochs, batch 128, lr 0.0015 (cosine, floor 7.5e-06) |
| HCNN | Conv(1→16, TANH) → MaxPool → Linear(32768→1) · 32993 params |

Open-loop baseline from Release `NARMA.exe`. Train-set target mean 0.9490
(identical series across seeds). Wall time ~24–26 min/seed on the collect
machine.

---

## Multi-seed summary (M=16)

| Metric | Value |
|--------|------:|
| Seeds | 73896–73900 (5) |
| mean NRMSE | **0.0590** |
| std | 0.0023 |
| min | **0.0570** |
| max | 0.0628 |
| M | 16 |
| collect | 32000 |

### Per-seed NRMSE

| res seed | NRMSE | R² | train (s) |
|---------:|------:|---:|----------:|
| **73896** | **0.0570** | **0.9968** | 1512 |
| 73897 | 0.0582 | 0.9966 | 1529 |
| 73898 | 0.0577 | 0.9967 | 1561 |
| 73899 | 0.0628 | 0.9961 | 1461 |
| 73900 | 0.0593 | 0.9965 | 1552 |

---

## M sweep

Single depth only for this campaign:

| M | mean NRMSE | std | min | max |
|--:|-----------:|----:|----:|----:|
| 16 | 0.0590 | 0.0023 | 0.0570 | 0.0628 |

---

## Notes

- Same drive/readout op-point as the NARMA-50 campaign (sr 0.99, is 0.03,
  collect 32000, seeds 73896–73900); **M=16** here vs **M=32** for N50
  ([NARMA-50.md](NARMA-50.md)).
- Seed spread is very tight (std ~0.002); all five seeds under 0.063 NRMSE.
- Supersedes an older single-seed collect-32000 figure of **0.0629** (same M
  and drive family; that run sits at the high end of this five-seed range).
- The multi-seed M-sweep table in [NARMA.md](NARMA.md#results-memory-depth-m-sweep)
  still uses **collect 8000** and an older op-point (means ~0.09–0.13 at good M).
- Compare next: NARMA-100 at the same op-point family.
