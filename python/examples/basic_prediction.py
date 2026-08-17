#!/usr/bin/env python3
"""Basic next-step prediction on a sine wave (Python host).

Mirrors the spirit of the C++ BasicPrediction example using only the public
``hypercube_esn`` API — no CMake, no native example binaries.

This is an onboarding demo (small dim, short train). It is **not** a paper
validator; see the C++ NARMA / MemoryCapacity / Lorenz campaigns for frozen
storefront metrics.

Requires a checkout of this file (examples are **not** installed by the
PyPI wheel). From the repo root, after ``pip install hypercube-esn`` (or
``pip install ./python``)::

    python python/examples/basic_prediction.py
"""

from __future__ import annotations

import numpy as np

import hypercube_esn as he


def main() -> None:
    # Synthetic scalar drive — float32 is the native exchange type.
    signal = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)

    # dim=7 → N = 128. Modest epochs so a laptop finishes in a few seconds.
    esn = he.ESN(dim=7, seed=73895, readout_epochs=80, verbose=True)
    # Auto next-step targets (horizon=1), default 70/30 train/test split.
    esn.fit(signal, warmup=200)

    print(f"collected states: {esn.num_collected_states}")
    print(f"train / test:     {esn.train_size} / {esn.test_size}")
    print(f"held-out R²:      {esn.r2():.6f}")
    print(f"held-out NRMSE:   {esn.nrmse():.6f}")

    # Live HCNN forward from the reservoir's *current* state (after the last
    # collected step of fit). Regression head — no softmax.
    y = esn.predict()
    print(f"live predict() shape: {y.shape}  value: {y[0]:.6f}")


if __name__ == "__main__":
    main()
