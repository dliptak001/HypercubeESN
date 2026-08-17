#!/usr/bin/env python3
"""Binary classification on a synthetic scalar signal (Python host).

Labels the sign of a sine wave (class 0 below zero, class 1 at/above) and
trains an HCNN classification head on the hypercube reservoir. Same public
``hypercube_esn`` API as regression — only ``readout_task`` and
``readout_num_outputs`` change.

Easy synthetic task (onboarding). Not a claim about industrial process ID;
see C++ SignalClassification for that style of host. Classification returns
**logits** (no softmax) — use argmax for the predicted class.

Requires a checkout of this file (examples are **not** installed by the
PyPI wheel). From the repo root, after ``pip install hypercube-esn`` (or
``pip install ./python``)::

    python python/examples/classification.py
"""

from __future__ import annotations

import numpy as np

import hypercube_esn as he


def main() -> None:
    n = 1600
    signal = np.sin(np.linspace(0, 16 * np.pi, n)).astype(np.float32)
    # Class indices as float32 (0.0 / 1.0) — multi-class layout for num_outputs=2.
    labels = np.where(signal >= 0.0, 1.0, 0.0).astype(np.float32)

    warmup = 100
    # Two output units = two class logits (SDK does not apply softmax).
    esn = he.ESN(
        dim=6,
        seed=73895,
        readout_task="classification",
        readout_num_outputs=2,
        readout_epochs=80,
        verbose=True,
    )

    # Explicit targets: one label per collected state (after warmup).
    # fit() holds out the tail via train_frac for accuracy().
    esn.fit(signal, targets=labels[warmup:], warmup=warmup, train_frac=0.7)

    acc = esn.accuracy()
    print(f"collected states: {esn.num_collected_states}")
    print(f"train / test:     {esn.train_size} / {esn.test_size}")
    print(f"held-out accuracy: {acc:.4f}  ({100.0 * acc:.1f}%)")

    # Argmax over logits from the live reservoir state after fit.
    logits = esn.predict()
    pred_class = int(np.argmax(logits))
    print(f"live logits: {logits}  → class {pred_class}")


if __name__ == "__main__":
    main()
