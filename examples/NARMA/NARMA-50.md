# NARMA-50 — Results

Parent walkthrough / recurrence / protocol: [NARMA.md](NARMA.md).

Order 50 is **not** a standard published RC rung. No literature NRMSE band —
internal stress test against NARMA-30 only. **tanh-wrapped** fixed coeffs
(`NARMA_TANH_WRAP=1`) for honest order-scaling.

---

## Spotlight — best-3 seeds (NRMSE 0.0767)

Top three of the 10-seed pool below (lowest NRMSE). Full pool: mean 0.0859,
std 0.0065, range 0.0767–0.0956.

| | NRMSE | vs literature |
|--|------:|---------------|
| **HypercubeESN (best seed)** | **0.0767** | no published band (internal stress rung) |
| Best-3 mean | **0.0791** | no published band |
| Best-3 std | 0.0021 | — |
| Best-3 range | 0.0767–0.0808 | — |

**Best test NRMSE 0.0767** (R² = 0.9941, res seed 221691) on **tanh-wrapped
NARMA-50**. Best-3 mean **0.0791** (std 0.0021). No comparable literature
band — internal stress rung vs NARMA-30 (**0.0570** best / **0.0576** best-3 mean in
[NARMA-30.md](NARMA-30.md)).

### Best-3 seeds

| res seed | NRMSE | R² | train (s) |
|---------:|------:|---:|----------:|
| **221691** | **0.0767** | **0.9941** | 1628 |
| 73899 | 0.0797 | 0.9937 | 1538 |
| 147792 | 0.0808 | 0.9935 | 1482 |

| Knob | Value |
|------|--------|
| Variant | tanh-wrapped (α=0.3, β=0.05, γ=1.5, δ=0.1) |
| Reservoir | DIM=10 (N=1024), M=`history_depth`=32, FSF **off** |
| Seed | best-3: 221691, 73899, 147792 (from 10-seed pool) |
| Series | warmup 300 · collect 32000 (train 25600 / test 6400) |
| Drive | sr 0.99 · leak 1 · input_scaling 0.03 |
| Readout | 600 epochs, batch 128, lr 0.0015 (cosine, floor 7.5e-06) |
| HCNN | Conv(1→16, TANH) → MaxPool → Linear(32768→1) · 32993 params |

Open-loop baseline from Release `NARMA.exe`. Train-set target mean 0.9946
(identical series across seeds). Wall time ~25–30 min/seed on the collect
machine.

---

## Multi-seed summary (M=32, 10 seeds)

Two batches, same op-point / series / M; reservoir seeds only differ.

| Metric | All 10 | Batch A (n=5) | Batch B (n=5) |
|--------|------:|-------------:|-------------:|
| Seeds | see table | 73896–73900 | 147792 … 443400 |
| mean NRMSE | **0.0859** | 0.0882 | 0.0836 |
| std | 0.0065 | 0.0069 | 0.0058 |
| min | **0.0767** | 0.0797 | 0.0767 |
| max | 0.0956 | 0.0956 | 0.0915 |
| M | 32 | 32 | 32 |
| collect | 32000 | 32000 | 32000 |

### Per-seed NRMSE

| res seed | batch | NRMSE | R² | train (s) |
|---------:|------:|------:|---:|----------:|
| 73896 | A | 0.0948 | 0.9910 | 1796 |
| 73897 | A | 0.0835 | 0.9930 | 1746 |
| 73898 | A | 0.0956 | 0.9909 | 1797 |
| 73899 | A | 0.0797 | 0.9937 | 1538 |
| 73900 | A | 0.0877 | 0.9923 | 1546 |
| 147792 | B | 0.0808 | 0.9935 | 1482 |
| **221691** | B | **0.0767** | **0.9941** | 1628 |
| 295592 | B | 0.0872 | 0.9924 | 1477 |
| 369495 | B | 0.0915 | 0.9916 | 1528 |
| 443400 | B | 0.0818 | 0.9933 | 1598 |

---

## M sweep

Single depth only for this campaign:

| M | n seeds | mean NRMSE | std | min | max |
|--:|--------:|-----------:|----:|----:|----:|
| 32 | 10 | 0.0859 | 0.0065 | 0.0767 | 0.0956 |

---

## Notes

- Same drive/readout op-point as the NARMA-30 spotlight (sr 0.99, is 0.03,
  collect 32000), except **M=32** (N30 spotlight used M=16).
- Seed spread stays tight after doubling the pool (std ~0.0065); all ten seeds
  sit under 0.10 NRMSE. Batch B mean (0.0836) is slightly better than batch A
  (0.0882); combined mean is **0.0859**.
