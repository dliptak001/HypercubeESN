# Closed-loop - Areas of Exploration

A running list of exploratory ideas — not approved work, just captured directions
worth investigating.

## 1. Free Running

_Closed-loop / autonomous generative operation — the readout drives the reservoir
forward with no external input (e.g. Lorenz free-run)._

## 2. General

### Ensemble consensus feedback

**Maturity:** _Novel_ — output-averaging ESN ensembles are established, but coupling
the members at runtime by feeding each its deviation from the ensemble consensus
through the **feedback driver path** is not standard procedure.

A general `EnsembleESN` capability with two configurations keyed to the training mode:
an **averaging ensemble** (no feedback; works in batch or online; members trained
independently and outputs combined) and a **feedback ensemble** (online only; the
consensus coupling is always engaged across training and inference; the feedback
intensity starts low and ramps up once members are competent, to avoid early
destabilization). Output-space combination (mean/median); members differ by seed.
Feedback needs one additive `ESN` seam (`StepLiveExternalFeedback`). Task-agnostic;
demonstration examples come after the capability.

→ Finalized design note: [ensemble_esn_feedback.md](ensemble_esn_feedback.md)
(two configurations, the online feedback mechanism, competence-gated intensity ramp,
member diversity, the one required core change, the κ = 0 baseline, open questions).
