# Python examples

Runnable **Python hosts** for the `hypercube-esn` API. They use only the public
package surface — no CMake, no C++ example binaries.

| Script | What it shows |
|--------|----------------|
| [`basic_prediction.py`](basic_prediction.py) | Scalar next-step regression (`fit` → R² / NRMSE) |
| [`classification.py`](classification.py) | Binary labels + classification head (`accuracy`) |

## How to run

Examples live in the **git tree**. They are **not** installed by the PyPI wheel
(`pip install hypercube-esn` alone does not place these scripts on disk).

From the **repository root** (after a Release wheel or local build):

```bash
pip install hypercube-esn
# or, from this tree:  pip install ./python
python python/examples/basic_prediction.py
python python/examples/classification.py
```

From the `python/` directory after an editable/local install:

```bash
cd python
pip install . --no-build-isolation   # needs a C++ toolchain if no wheel cache
python examples/basic_prediction.py
```

## What these are not

- **Not** the frozen C++ storefronts (NARMA best-5, MemoryCapacity grids, Lorenz
  VPT / GS surveys). Those live under [`examples/`](../../examples/) and report
  the paper / release numbers.
- **Not** pytest. CI smoke is [`tests/test_basic.py`](../tests/test_basic.py);
  these scripts are for humans to read and run.
- **Not** hard tasks. Sine next-step and sign-of-sine classification are easy
  onboarding signals so the API is obvious; do not cite their metrics as
  storefront results.

## Going further

| Want… | See… |
|-------|------|
| Full Python API | [`docs/Python_SDK.md`](../../docs/Python_SDK.md) |
| C++ BasicPrediction / NARMA / Lorenz | [`examples/README.md`](../../examples/README.md) |
| Package install / PyPI | [`README.md`](../README.md) |
