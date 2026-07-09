---
name: pybuild
description: Rebuild and smoke-test the HypercubeESN Python bindings module
argument-hint: "[test|smoke|full]"
user-invocable: true
allowed-tools: Read, Bash, Glob, Grep
---

# pybuild — Rebuild & Smoke-Test Python Bindings

Rebuild the `hypercube-esn` Python module from `python/` and run a smoke test to verify it works.

## Argument Parsing

The user invokes this skill as `/pybuild $ARGUMENTS`.

- If `$ARGUMENTS` is `?` or `help` → display the **Usage Help** below, then stop
- If `$ARGUMENTS` is empty or `smoke` → rebuild + quick smoke test (sine prediction + all DIMs)
- If `$ARGUMENTS` is `test` or `full` → rebuild + full pytest suite (`python/tests/test_basic.py`)

### Usage Help

When the user asks for help, respond with this and do nothing else:

```
## PYBUILD — Rebuild & Smoke-Test Python Bindings

**Syntax**: `/pybuild [smoke|test|full]`

**Examples**:
  /pybuild          — Rebuild and run quick smoke test
  /pybuild smoke    — Same as above
  /pybuild test     — Rebuild and run full pytest suite
  /pybuild full     — Same as test
```

---

## Step 1: Rebuild the Python Module

Run the following PowerShell command via bash heredoc. This is the ONLY build method that works on this system — do NOT deviate from it.

```bash
powershell.exe -File - <<'PSEOF'
$env:PATH = "C:\Program Files\JetBrains\CLion 2026.1\bin\mingw\bin;C:\Program Files\JetBrains\CLion 2026.1\bin\ninja\win\x64;" + $env:PATH
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_MAKE_PROGRAM = "C:\Program Files\JetBrains\CLion 2026.1\bin\ninja\win\x64\ninja.exe"
$env:CC = "C:\Program Files\JetBrains\CLion 2026.1\bin\mingw\bin\gcc.exe"
$env:CXX = "C:\Program Files\JetBrains\CLion 2026.1\bin\mingw\bin\g++.exe"
Set-Location "C:\CLion\HypercubeESN\python"
& "C:\Program Files\Python310\python.exe" -m pip install . --no-build-isolation --force-reinstall --no-deps 2>&1
PSEOF
```

### Build rules

- **MUST use PowerShell** — bash cannot properly pass env vars to pip's subprocess
- **MUST use `--no-build-isolation`** — pip's isolated build env does not inherit PATH, so cmake/gcc/ninja are invisible
- **MUST set all 5 env vars** (`PATH`, `CMAKE_GENERATOR`, `CMAKE_MAKE_PROGRAM`, `CC`, `CXX`) with full paths to CLion-bundled tools
- **MUST use `--force-reinstall --no-deps`** to replace the existing install without re-downloading numpy

### If build fails

1. Read the error output carefully
2. If it's a C++ compilation error — the issue is in `python/bindings.cpp` or the core sources
3. If it's a CMake error about missing programs — check that the env vars are set correctly
4. If it's a linker error about undefined symbols — check `python/CMakeLists.txt` static linking section
5. Report the error with diagnosis to the user

---

## Step 2: Smoke Test or Full Test

### Smoke test (default)

Run from a temp `.py` file located OUTSIDE the `python/` tree (so the source `hypercube_esn/` package dir doesn't shadow the installed one). Do NOT use inline `python -c "…"` or a PowerShell here-string (`@'…'@`) nested inside the bash heredoc — both silently return no output through this harness.

**Step 2a — create the script with the Write tool** (not `Set-Content`) at `C:\Users\DavidLiptak\pybuild_smoke.py`:

```python
import hypercube_esn as he
import numpy as np

print(f"hypercube_esn {he.__version__}")
signal = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)
esn = he.ESN(reservoir_hypercube_dimension=6, seed=42)
esn.fit(signal, warmup=200, train_frac=0.7)
r2 = esn.r2()
nrmse = esn.nrmse()
print(f"Sine next-step: R2={r2:.6f}, NRMSE={nrmse:.6f}")
assert r2 > 0.99, f"R2 too low: {r2}"
for d in range(5, 13):
    e = he.ESN(reservoir_hypercube_dimension=d, seed=1)
    print(f"  DIM {d}: N={e.reservoir_neuron_count} OK")
print("Smoke test PASSED")
```

**Step 2b — run it** with a simple PowerShell line (this form reliably returns output):

```bash
powershell.exe -File - <<'PSEOF'
$env:PATH = "C:\Program Files\JetBrains\CLion 2026.1\bin\mingw\bin;" + $env:PATH
Set-Location C:\Users\DavidLiptak
& "C:\Program Files\Python310\python.exe" C:\Users\DavidLiptak\pybuild_smoke.py 2>&1
Remove-Item C:\Users\DavidLiptak\pybuild_smoke.py -Force
PSEOF
```

The `ESN.fit(...)` high-level call does warmup + run + train with an automatic train/test split; the no-argument `r2()` / `nrmse()` then evaluate the held-out test portion. The reservoir prints a one-line `[Reservoir ...]` diagnostic per construction (to stderr, after the script's own stdout) — expected, not an error.

### Full test (`test` or `full` argument)

```bash
powershell.exe -File - <<'PSEOF'
Set-Location C:\Users\DavidLiptak
& "C:\Program Files\Python310\python.exe" -m pytest C:\CLion\HypercubeESN\python\tests\test_basic.py -v --import-mode=importlib 2>&1
PSEOF
```

### Test rules

- **MUST use `--import-mode=importlib`** — default pytest import mode resolves the test file path back into the `python/` source tree, causing it to import the source `hypercube_esn/__init__.py` (which lacks `_core.pyd`) instead of the installed package
- `C:\Users\DavidLiptak` is a safe working directory for this purpose

---

## Step 3: Report

Structure the output as:

```
## Python Build Report

**Build**: [Succeeded | Failed] (wheel size)
**Test**: [N/N passed | FAILED] (duration)

### Results
[Key metrics from smoke test or pytest output]

### Errors
[If any — with diagnosis]
```

If the smoke test R² drops below 0.99 or any DIM fails to construct, flag it as a regression.
