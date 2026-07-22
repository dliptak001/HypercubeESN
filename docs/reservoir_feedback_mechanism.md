# Reservoir external-feedback substrate

> Status: **landed**. External (caller-owned) drive port.
>
> With full-state linear feedback (FSF) also landed as a separate **internal** port,
> this document covers only the **external** path. See
> [full_state_linear_feedback.md](full_state_linear_feedback.md) for FSF and
> [Reservoir.md](Reservoir.md) for the full drive-port picture.

## 1. Source-agnostic by design — external drive only (this port)

The reservoir exposes an **external feedback port**: a buffer the caller stages each
step, summed into the neurons, then cleared. The reservoir does **not** invent these
values.

Full-state linear feedback is a **different** port: internal policy φ = V·x, not
staged via `InjectExternalFeedback`. Do not overload this port for FSF.

## 2. Mechanism — twin of the input port

```
 weight layout in vtx_weight_:
   [ input | external feedback (if D>0) | FSF (if enabled) | recurrent ]
              └ external_feedback_scaling/√dim     └ SR-rescaled only
```

- **Buffer** `vtx_ext_feedback_`: allocated only when `num_external_feedback_channels > 0`.
- **Weight block** drawn from the master-seed **ExternalFeedback** substream (label 3).
- **Fan-in** `external_feedback_scaling / √dim` (local variance normalization of the dim-neighbor gather; not a claim that the scaling transfers across DIM).
- **Outside SR rescale.**
- **Per-step:** stage → `Step` consumes → clear. Not in snapshots.

## 3. Injection API

```cpp
void Reservoir::InjectExternalFeedback(size_t channel, float value);
void Reservoir::InjectExternalFeedback(const float* values, size_t count); // count == D
```

Channel layout: block broadcast, `D ≤ N`, D need not divide N (benign tail).

## 4. ESN seam

```cpp
ESN::ReservoirStep(inputs, external_feedback = nullptr);
```

Non-null `external_feedback` requires `num_external_feedback_channels > 0`. This is
the only way **caller-owned** closed-loop drive enters at the ESN layer. FSF, if
enabled, still applies inside `Reservoir::Step` regardless of this argument.
