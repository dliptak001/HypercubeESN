# Basic Prediction — Sine Wave Forecasting

## What this example demonstrates

This is the simplest possible end-to-end reservoir computing demo.
A sine wave is fed into the reservoir, and the HCNN readout learns to
predict the next value without ever seeing the input directly.

If you're new to reservoir computing, start here.

## Conceptual background

In a traditional neural network, you train all the weights. In reservoir
computing, the recurrent network (the "reservoir") has fixed, random weights.
Only the output layer (the "readout") is trained.

This works because the reservoir transforms the 1-dimensional input into an
N-dimensional state vector that encodes the input's recent history. The readout
finds the right mapping from those N dimensions to the target.

A sine wave is the easiest test case: the dynamics are perfectly periodic,
so the reservoir state at any point encodes enough history to predict the
next value with near-zero error.

## The pipeline

```
Input signal ──> Reservoir ──> Readout ──> Prediction
  sin(0.1t)      256 neurons     (trained)      sin(0.1(t+1))
                  (fixed)
```

**Step by step:**

1. **Generate signal** — A sine wave, `sin(0.1t)`. Amplitude stays in [-1, +1],
   which is the reservoir's native input range.

2. **Warmup** — Drive the reservoir for 200 steps without recording. This lets
   the internal state settle into a trajectory that reflects the input history.

3. **Collect** — Drive for 2000 more steps, recording the N-dimensional state
   at each step. This is the training + test data.

4. **Train** — The HCNN readout takes the full 256-vertex raw state as input.
   Convolutional kernels discover features on the hypercube topology. Trained
   with 1500 epochs of Adam with a cosine learning-rate schedule.

5. **Evaluate** — Measure R² and NRMSE on the held-out 30% test set.

## What to expect

DIM=8, 256 neurons, `history_depth=16` and `spectral_radius=0.99` (realized
~0.99), `input_scaling=0.09`, `leak_rate=1.0` (full replacement — the struct
default, no override). Sine prediction is trivially easy:

- **HCNN** (all 256 vertices, 1 Conv+Pool pair, 1500 epochs, batch 64,
  `lr_max=0.0015` cosine, ~13 s): R² = 1.000000 (effectively perfect),
  NRMSE ≈ 0.00003 — well into the sub-0.1% range.

The value of this example is in demonstrating the pipeline, not
stressing the readout. Harder tasks (StreamingAnomaly,
SignalClassification) are where the architecture shows its capacity.

## Things to try

- **Leak rate.** The example uses the struct default (`leak_rate = 1.0`,
  full replacement). Set `cfg.reservoir.leak_rate` in the source to try
  0.5 (moderate) or 0.2 (slow leaky integrator). At leak < 1.0, neurons
  blend old state with new activation:
  ```
  state[v] = (1 - leak) * old_state[v] + leak * tanh(s)
  ```
  where `s` is the pre-activation drive for vertex `v` (the input term
  `input_scaling * W_in·u` plus the recurrent term `W·state`).
  This smooths the state trajectory for periodic signals, but the effect
  is modest here because sine prediction is already near-perfect. The
  leak rate matters much more on irregular signals — see StreamingAnomaly
  and SignalClassification.

- **HCNN epochs / learning rate.** The default is 1500 epochs.
  Raising `cfg.readout.lr_max` above ~0.005 is risky — weights can
  diverge into denormals and the CPU falls off fast-math paths.

- **HCNN layer count.** The readout uses one Conv+Pool pair
  (`num_layers = 1`, the struct default — not overridden in the source).
  Set `cfg.readout.num_layers = 0` for auto-sizing (`min(DIM-2, 2)` pairs)
  or increase to see how depth affects fit on a trivial signal.

- **Change the signal.** Replace `sin(0.1t)` with a more complex waveform
  (sum of two sines, or a chirp). HCNN's advantage grows as the target
  becomes more nonlinear in the reservoir state.

- **Increase the horizon.** Change `horizon = 1` to 5 or 10. Multi-step
  prediction is harder because the reservoir must encode more history.

## Build and run

cmake/g++ ship with CLion and are not on `PATH`. Build the Release tree with
the bundled toolchain (see [Building and Running](../README.md#building-and-running-c)
in the project README), then run the `BasicPrediction` target:

```
cmake-build-release\BasicPrediction.exe
```
