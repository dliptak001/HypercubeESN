# TODO — Forward-only vs Janus (next session)

**Context:** Critical assessment of Janus Cursor + port mapping + ablation design  
**Date parked:** 2026-07-27  
**Status:** design agreed; **implementation next**

---

## Confirmed facts (do not re-litigate)

1. **Port mapping (keep as-is)**  
   - **Past / reverse** cursor → **input** port (`num_inputs = 4`)  
   - **Future / forward** cursor → **external-feedback** port  
   - Free-run: real past on input; `ExtractFuturePredicted` on ext-fb  

2. **Do not reverse the ports** for this experiment — ext-fb is the teacher → self-loop path; input is the anchor.

3. **Canonical comparison we want**  
   - **Baseline (Janus):** forward + reverse (current)  
   - **Forward-only:** reverse has **no dynamical impact** — as if reverse path does not exist  

---

## Forward-only = zero reverse every step (train + free-run)

Enough for “no reverse impact” on the reservoir:

```text
past[0..3] = 0   // every ReservoirStep
future     = real future          // train / washout
future     = pack(prediction)     // free-run
ReservoirStep(past, future)
```

- Keep architecture fixed vs baseline (`num_inputs = 4`, same scales, seeds, orbits).  
- Zero the **staged past vector**; do not need to remove the input weight block.  
- Zero reverse in **warmup + train + washout + free-run** (whole trial).  
- Free-run-only zero is a *different* arm (test-time tether drop), not “forward-only model.”

---

## Implementation sketch (when coding)

- [ ] Flag or config arm: e.g. `FORWARD_ONLY` / `USE_REVERSE_PATH` in `config::` or CLI  
- [ ] Baseline: current `ExtractPast`  
- [ ] Forward-only: skip ExtractPast / memset past to 0 before every `ReservoirStep`  
- [ ] Same multi-seed / multi-orbit free-run survey as today  
- [ ] Log which arm; report VPT (lt) + free-run RMSE side by side  
- [ ] Optional later: train-On / free-run-Off arm; matched drive energy  

---

## Related reading

- `JanusCursor.md` — index geometry  
- `README.md` — lag curriculum, port roles, half-anchored free-run  
- Session notes: Janus critique (confounds lag curriculum vs free-run); ablation design (2×2 if expanding)

---

## Out of scope for first cut

- Port swap  
- Pure Pathak (no real past) as the only arm  
- Changing free-run window / VPT threshold between arms  
