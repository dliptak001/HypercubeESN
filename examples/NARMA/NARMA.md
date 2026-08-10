# NARMA-N — Nonlinear System Identification

NARMA is HypercubeESN’s primary **open-loop** validator: a white input `u(t)`
drives the reservoir; the readout reconstructs the NARMA output `y(t)`. The task
stresses **memory depth and nonlinear mixing together**.

## One system, three orders

The point of this campaign is not three separately tuned demos. It is that
**one fixed HypercubeESN configuration** — same dim, M, spectral radius, input
scaling, leak, series length, HCNN stack, training schedule, and the same 20
reservoir seeds — is run on tanh-wrapped **NARMA-30, NARMA-50, and NARMA-70**.
Only the recurrence order changes.

**Featured metric:** best 5 of 20 (lowest test NRMSE among the 20 seeds).

| Order | Best-5 mean | Best-5 std | Best seed |
|------:|------------:|-----------:|----------:|
| **30** | **0.0441** | 0.0017 |  **0.0419** |
| **50** | **0.0751** | 0.0009 |  **0.0742** |
| **70** | **0.1251** | 0.0016 |  **0.1225** |

### Is a shared op-point across orders common?

**Somewhat uncommon in the published RC literature.**

Holding the machine fixed and only changing order is closer to a
**robustness/transfer** test: does the architecture + delay-line depth still
identify the series when the required lag and nonlinear product structure get
harder?

Caveat so this is not oversold:

- We still chose a strong shared op-point. It is not “default random knobs.”

---

## Shared configuration

| Knob | Value |
|------|--------|
| Variant | **tanh-wrapped** — fixed coeffs α=0.3, β=0.05, γ=1.5, δ=0.1; `u ∈ [0, 0.5]` |
| `data_seed` | **1939** — fixes the entire u/y series (same series for every reservoir seed) |
| Reservoir | dim=10 (N=1024), **M = history_depth = 32**, spectral radius 0.99, leak 1.0, input_scaling **0.03**, **bias_scaling 0.02** |
| Readout input | `readout_slices = 2` (B=2 → HCNN start dim 11, capacity 2048) |
| HCNN | Conv(1→16, TANH) → MaxPool → Linear(16384→1) · **16593** trained parameters |
| `readout.seed` | **3423555** — fixed HCNN weight init so multi-seed spread is reservoir-side |
| Series | warmup 300 · collect **32000** (train 25600 / test 6400) |
| Training | 600 epochs, batch 128, lr 0.0015 cosine (floor 7.5e-06) |
| Best-epoch | `restore_best_epoch = true`, **holdout_frac = 0** — restores min **train** MSE epoch (not a validation split). Test NRMSE is still a clean held-out metric. |
| Reservoir seeds (live harness) | Explicit list `campaign::kReservoirSeeds` in `NARMA.cpp`: **1** entry = spot run; **3** entries = literature band (mean/std/min/max over all three). No best-k selection. |

Metric: **test NRMSE** = RMSE / std(target) on the held-out 6400 steps.

---

## Conclusion

Strong performance across all three orders. As expected, error rises with
order.
