# NARMA-50 — Results

Parent walkthrough / recurrence / protocol: [NARMA.md](NARMA.md).

Order 50 is **not** a standard published RC rung. No literature NRMSE band —
internal stress test against NARMA-30 only. **tanh-wrapped** fixed coeffs
(`NARMA_TANH_WRAP=1`) for honest order-scaling.

---

## Spotlight — best seed NRMSE 0.0797

| | NRMSE | vs this run |
|--|------:|------------|
| **HypercubeESN (best seed)** | **0.0797** | — |
| HypercubeESN (5-seed mean) | 0.0882 | — |
| Literature “strong / large-N” band | — | *no published band* |
| Literature “good” band | — | *no published band* |

**Best test NRMSE 0.0797** (R² = 0.9937, res seed 73899) on **tanh-wrapped
NARMA-50**. Five-seed mean **0.0882** (std 0.0069). No comparable literature
band — internal stress rung vs NARMA-30 (**0.0570** best / **0.0590** mean in
[NARMA-30.md](NARMA-30.md)).

| Knob | Value |
|------|--------|
| Variant | tanh-wrapped (α=0.3, β=0.05, γ=1.5, δ=0.1) |
| Reservoir | DIM=10 (N=1024), M=`history_depth`=32, FSF **off** |
| Seed | best res 73899; sweep 73896–73900 |
| Series | warmup 300 · collect 32000 (train 25600 / test 6400) |
| Drive | sr 0.99 · leak 1 · input_scaling 0.03 |
| Readout | 600 epochs, batch 128, lr 0.0015 (cosine, floor 7.5e-06) |
| HCNN | Conv(1→16, TANH) → MaxPool → Linear(32768→1) · 32993 params |

Open-loop baseline from Release `NARMA.exe`. Train-set target mean 0.9946
(identical series across seeds). Wall time ~26–30 min/seed on the collect
machine.

---

## Multi-seed summary (M=32)

| Metric | Value |
|--------|------:|
| Seeds | 73896–73900 (5) |
| mean NRMSE | **0.0882** |
| std | 0.0069 |
| min | **0.0797** |
| max | 0.0956 |
| M | 32 |
| collect | 32000 |

### Per-seed NRMSE

| res seed | NRMSE | R² | train (s) |
|---------:|------:|---:|----------:|
| 73896 | 0.0948 | 0.9910 | 1796 |
| 73897 | 0.0835 | 0.9930 | 1746 |
| 73898 | 0.0956 | 0.9909 | 1797 |
| **73899** | **0.0797** | **0.9937** | 1538 |
| 73900 | 0.0877 | 0.9923 | 1546 |

---

## M sweep

Single depth only for this campaign:

| M | mean NRMSE | std | min | max |
|--:|-----------:|----:|----:|----:|
| 32 | 0.0882 | 0.0069 | 0.0797 | 0.0956 |

---

## Notes

- Same drive/readout op-point as the NARMA-30 spotlight (sr 0.99, is 0.03,
  collect 32000), except **M=32** (N30 spotlight used M=16).
- Seed spread is tight (std ~0.007); all five seeds sit under 0.10 NRMSE.
- `history_depth` is capped at 64 in the reservoir — M cannot match order 50
  one-for-one; M=32 is the MC-favored depth on this stack.
- Compare next: NARMA-100 at the same op-point.
