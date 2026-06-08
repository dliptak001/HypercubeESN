"""HypercubeESN: reservoir computing on Boolean hypercube graphs.

This package provides Python bindings for the HypercubeESN C++ library.
The reservoir topology is a Boolean hypercube graph with N = 2^dim neurons.

Quick start::

    import numpy as np
    import hypercube_esn as he

    signal = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)
    esn = he.ESN(dim=7)
    esn.fit(signal, warmup=200)
    print(f"R² = {esn.r2()}")
"""

from __future__ import annotations

import pathlib
import pickle
import numpy as np

from ._core import _ESN

__version__ = "1.0.1"
__all__ = ["ESN"]

# Valid hypercube dimensions (matches the C++ Reservoir::Create [5, 16] check).
_DIM_MIN = 5
_DIM_MAX = 16


def _to_float32(arr):
    """Ensure array is C-contiguous float32."""
    return np.ascontiguousarray(arr, dtype=np.float32)


class ESN:
    """Echo State Network on a Boolean hypercube reservoir.

    The reservoir has N = 2^dim neurons arranged on the vertices of a
    dim-dimensional Boolean hypercube. Connectivity is computed via XOR
    addressing with zero storage overhead.

    Parameters
    ----------
    dim : int
        Hypercube dimension (5-16). Determines the number of neurons: N = 2^dim.
    seed : int
        RNG seed for weight initialization. Default: 73895 (a surveyed default
        seed; the realization a seed yields measurably affects performance).
    spectral_radius : float
        Target spectral radius for the recurrent weight matrix. Default: 0.99.
        DIM-invariant: the same value works at every dim.
    input_scaling : float
        Input drive coefficient. Input weights carry a 1/√dim fan-in
        normalization, so a given value yields the same tanh drive at any dim.
        Default: 0.5. (The legacy 0.02 was a normalization artifact and no
        longer applies.)
    leak_rate : float
        Leaky integrator coefficient. 1.0 = full replacement (default),
        < 1.0 adds temporal smoothing.
    num_inputs : int
        Number of input channels. Channel k drives the contiguous vertex block
        [k * N/num_inputs, (k+1) * N/num_inputs). Default: 1.
    history_depth : int
        Delay-line depth M: how many past output slices the readout sees,
        in [1, 64]. Deeper lines extend short-range temporal memory. Default: 16.
    output_fraction : float
        Fraction of N vertices used as readout features, in (0.0, 1.0]. Default: 1.0.
        Reduce for large dim to control readout cost via sub-hypercube selection.
    readout_num_outputs : int
        Output dimensionality. Targets for regression, classes for classification.
        Default: 1.
    readout_task : str
        "regression" (default) or "classification".
    readout_num_layers : int
        Conv+Pool pairs in the HCNN readout. 0 = auto: min(dim-2, 2). Default: 0.
    readout_conv_channels : int
        Base channel count (doubles per layer). Default: 16.
    readout_epochs : int
        Number of training epochs. Default: 200.
    readout_batch_size : int
        Mini-batch size. Default: 32.
    readout_lr_max : float
        Peak learning rate (cosine schedule). Keep <= 0.005 to avoid NaN. Default: 0.005.
    readout_lr_min_frac : float
        LR floor as a fraction of lr_max. Default: 0.1.
    readout_lr_decay_epochs : int
        Cosine decay horizon. 0 = use ``readout_epochs``. Default: 0.
    readout_weight_decay : float
        L2 regularization on CNN weights. Default: 0.0.
    readout_seed : int
        CNN weight initialization seed. Default: 42.
    readout_verbose : bool
        Print per-epoch LR / loss to stdout. Default: False.

    Examples
    --------
    Simple (single-input next-step prediction):

    >>> import numpy as np
    >>> import hypercube_esn as he
    >>> signal = np.sin(np.linspace(0, 20*np.pi, 2000)).astype(np.float32)
    >>> esn = he.ESN(dim=6)
    >>> esn.fit(signal, warmup=200)
    ESN(dim=6, ...)
    >>> esn.r2()
    0.999...

    Explicit (multi-input, custom targets):

    >>> esn = he.ESN(dim=7, num_inputs=2)
    >>> esn.warmup(inputs[:200])
    >>> esn.run(inputs[200:])
    >>> esn.train(targets[:1400])
    >>> esn.r2(targets, start=1400)
    0.99...
    """

    def __init__(
        self,
        dim: int,
        *,
        seed: int = 73895,
        spectral_radius: float = 0.99,
        input_scaling: float = 0.5,
        leak_rate: float = 1.0,
        num_inputs: int = 1,
        history_depth: int = 16,
        output_fraction: float = 1.0,
        readout_num_outputs: int = 1,
        readout_task: str = "regression",
        readout_num_layers: int = 0,
        readout_conv_channels: int = 16,
        readout_epochs: int = 200,
        readout_batch_size: int = 32,
        readout_lr_max: float = 0.005,
        readout_lr_min_frac: float = 0.1,
        readout_lr_decay_epochs: int = 0,
        readout_weight_decay: float = 0.0,
        readout_seed: int = 42,
        readout_verbose: bool = False,
    ):
        if not (_DIM_MIN <= dim <= _DIM_MAX):
            raise ValueError(f"dim must be {_DIM_MIN}-{_DIM_MAX}, got {dim}")
        self._readout_kwargs = {
            "readout_num_outputs": readout_num_outputs,
            "readout_task": readout_task,
            "readout_num_layers": readout_num_layers,
            "readout_conv_channels": readout_conv_channels,
            "readout_epochs": readout_epochs,
            "readout_batch_size": readout_batch_size,
            "readout_lr_max": readout_lr_max,
            "readout_lr_min_frac": readout_lr_min_frac,
            "readout_lr_decay_epochs": readout_lr_decay_epochs,
            "readout_weight_decay": readout_weight_decay,
            "readout_seed": readout_seed,
            "readout_verbose": readout_verbose,
        }
        self._impl = _ESN(
            dim=dim,
            seed=seed,
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            leak_rate=leak_rate,
            num_inputs=num_inputs,
            history_depth=history_depth,
            output_fraction=output_fraction,
            **self._readout_kwargs,
        )
        self._targets: np.ndarray | None = None
        self._train_size: int | None = None

    def warmup(self, inputs: np.ndarray) -> None:
        """Drive the reservoir without recording states (wash out transient).

        Parameters
        ----------
        inputs : ndarray
            Input signal. Shape ``(num_steps,)`` for single-input or
            ``(num_steps, num_inputs)`` for multi-input. Converted to float32.
        """
        self._impl.warmup(_to_float32(inputs))

    def run(self, inputs: np.ndarray) -> None:
        """Drive the reservoir and record states for training/evaluation.

        Parameters
        ----------
        inputs : ndarray
            Input signal. Same shape convention as ``warmup()``.

        Notes
        -----
        Multiple ``run()`` calls accumulate states. Use ``clear_states()``
        to reset between independent sequences.
        """
        self._impl.run(_to_float32(inputs))

    def clear_states(self) -> None:
        """Clear collected states and cached features.

        The reservoir's live internal state is preserved. The trained
        readout is also preserved. Stored targets from ``fit()`` are cleared.
        """
        self._impl.clear_states()
        self._targets = None
        self._train_size = None

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray | None = None,
        *,
        warmup: int = 200,
        train_size: int | None = None,
        train_frac: float | None = None,
        horizon: int = 1,
    ) -> "ESN":
        """High-level pipeline: warmup, run, train with automatic train/test split.

        Two modes:

        **Auto-target** (``targets=None``, single-input only):
            Generates next-step prediction targets from the input signal,
            shifted by ``horizon`` steps.

        **Explicit-target** (any ``num_inputs``):
            Uses the provided ``targets`` array directly. ``targets[i]`` is
            the target for the i-th collected state. ``horizon`` is ignored.

        After ``fit()``, call ``r2()``, ``nrmse()``, or ``accuracy()`` with
        no arguments to evaluate the held-out test portion.

        Parameters
        ----------
        inputs : ndarray
            Input signal. Shape ``(total_steps,)`` for single-input or
            ``(total_steps, num_inputs)`` for multi-input.
        targets : ndarray, optional
            Explicit target values, one per collected state. Required for
            multi-input ESN. If omitted, auto-generates next-step targets
            from the input signal.
        warmup : int
            Number of initial timesteps for transient washout. Default: 200.
        train_size : int, optional
            Number of training samples. Mutually exclusive with ``train_frac``.
        train_frac : float, optional
            Fraction of collected states used for training. Used only when
            ``train_size`` is not provided. Default: 0.7 if neither is given.
        horizon : int
            Prediction horizon for auto-target mode. Target at step t is the
            input at step t + horizon. Ignored when ``targets`` is provided.
            Default: 1.

        Returns
        -------
        ESN
            Self, for method chaining.

        Examples
        --------
        Single-input next-step prediction:

        >>> esn = he.ESN(dim=7)
        >>> esn.fit(signal, warmup=200)
        >>> print(esn.r2())

        Multi-input with explicit targets:

        >>> esn = he.ESN(dim=7, num_inputs=3)
        >>> esn.fit(inputs, targets=channel_0[201:], warmup=200)
        >>> print(esn.r2())
        """
        inputs = _to_float32(inputs)
        self.clear_states()

        if targets is None:
            # Auto-target mode: next-step prediction on single-input signal
            if self.num_inputs != 1:
                raise ValueError(
                    "targets must be provided for multi-input ESN "
                    f"(num_inputs={self.num_inputs}). Auto-target (next-step "
                    "prediction) is only available for single-input."
                )
            if inputs.ndim != 1:
                raise ValueError(
                    f"inputs must be 1D for auto-target mode, got shape {inputs.shape}"
                )
            if horizon < 1:
                raise ValueError(f"horizon must be >= 1, got {horizon}")
            if warmup + horizon >= len(inputs):
                raise ValueError(
                    f"warmup ({warmup}) + horizon ({horizon}) >= len(inputs) "
                    f"({len(inputs)}). Not enough data to collect any states."
                )
            self.warmup(inputs[:warmup])
            self.run(inputs[warmup:-horizon])
            self._targets = _to_float32(inputs[warmup + horizon:])
        else:
            # Explicit-target mode: works for any num_inputs
            targets = _to_float32(targets)
            if warmup >= len(inputs) if inputs.ndim == 1 else warmup >= inputs.shape[0]:
                raise ValueError(
                    f"warmup ({warmup}) >= number of input steps. "
                    "Not enough data to collect any states."
                )
            self.warmup(inputs[:warmup])
            self.run(inputs[warmup:])
            if len(targets) != self.num_collected:
                raise ValueError(
                    f"targets length ({len(targets)}) must equal num_collected "
                    f"({self.num_collected}). Provide one target per collected state."
                )
            self._targets = targets

        # Determine train_size
        if train_size is not None:
            if train_frac is not None:
                raise ValueError("Specify train_size or train_frac, not both")
        else:
            if train_frac is None:
                train_frac = 0.7
            train_size = int(self.num_collected * train_frac)

        if train_size <= 0 or train_size > self.num_collected:
            raise ValueError(
                f"train_size ({train_size}) must be in [1, num_collected={self.num_collected}]"
            )

        self._train_size = train_size
        self.train(self._targets[:train_size])
        return self

    def train(
        self,
        targets: np.ndarray,
    ) -> None:
        """Train the HCNN readout on collected states.

        Uses the readout configuration supplied at ESN construction time
        (the ``readout_*`` keyword arguments to the constructor).

        Parameters
        ----------
        targets : ndarray
            For regression: shape ``(train_size,)`` or
            ``(train_size * num_outputs,)`` for multi-output.
            For classification: shape ``(train_size,)`` with class indices
            as floats (0.0, 1.0, ...).
        """
        self._impl.train(_to_float32(targets))

    def predict_raw(self, timestep: int) -> float:
        """Return the raw continuous prediction for a collected timestep.

        Parameters
        ----------
        timestep : int
            Index into collected states, in [0, num_collected).

        Returns
        -------
        float
            Continuous prediction value.
        """
        return self._impl.predict_raw(timestep)

    def predict_live_raw(self) -> float:
        """Predict from the reservoir's current live state (single output).

        For autoregressive / closed-loop inference loops: drive the reservoir
        one step (``run``/``warmup``), then read the prediction here without
        touching the collected-state buffer.

        Returns
        -------
        float
            Continuous prediction value. Use :meth:`predict_live_raw_multi`
            when the readout has more than one output.
        """
        return self._impl.predict_live_raw()

    def predict_live_raw_multi(self) -> np.ndarray:
        """Multi-output predict from the reservoir's current live state.

        The closed-loop feedback primitive for multi-output readouts: after
        driving the reservoir one step, returns every channel's prediction so
        they can be fed back together.

        Returns
        -------
        ndarray
            1D array of shape ``(num_outputs,)`` with float32 predictions.
        """
        return self._impl.predict_live_raw_multi()

    def predictions(self) -> np.ndarray:
        """Return predictions for all collected timesteps.

        Returns
        -------
        ndarray
            1D array of shape ``(num_collected,)`` with float32 predictions.
        """
        return self._impl.predictions()

    def r2(
        self,
        targets: np.ndarray | None = None,
        start: int | None = None,
        count: int | None = None,
    ) -> float:
        """Compute R-squared on a slice of collected states.

        Can be called with no arguments after ``fit()`` to evaluate the
        held-out test portion, or with explicit arguments for full control.

        Parameters
        ----------
        targets : ndarray, optional
            Target values, index-aligned with collected states (``targets[i]``
            is the target for collected state ``i``). If omitted, uses targets
            stored by ``fit()``.

            **Important:** Do not slice targets before passing. Use ``start``
            and ``count`` to select the evaluation window.
        start : int, optional
            First timestep index. Default: 0, or ``train_size`` after ``fit()``.
        count : int, optional
            Number of timesteps to evaluate. Default: all remaining from ``start``.

        Returns
        -------
        float
            R² value. 1.0 = perfect, 0.0 = predicts the mean. Can be negative.
        """
        targets, start, count = self._resolve_eval_args(targets, start, count)
        return self._impl.r2(targets, start, count)

    def nrmse(
        self,
        targets: np.ndarray | None = None,
        start: int | None = None,
        count: int | None = None,
    ) -> float:
        """Compute Normalized RMSE on a slice of collected states.

        Can be called with no arguments after ``fit()`` to evaluate the
        held-out test portion, or with explicit arguments for full control.

        Parameters
        ----------
        targets : ndarray, optional
            Same convention as ``r2()``.
        start : int, optional
            First timestep index. Default: 0, or ``train_size`` after ``fit()``.
        count : int, optional
            Number of timesteps to evaluate. Default: all remaining from ``start``.

        Returns
        -------
        float
            NRMSE value. 0.0 = perfect, 1.0 = as bad as predicting the mean.
        """
        targets, start, count = self._resolve_eval_args(targets, start, count)
        return self._impl.nrmse(targets, start, count)

    def accuracy(
        self,
        labels: np.ndarray | None = None,
        start: int | None = None,
        count: int | None = None,
    ) -> float:
        """Compute classification accuracy on a slice of collected states.

        Can be called with no arguments after ``fit()`` to evaluate the
        held-out test portion, or with explicit arguments for full control.

        Parameters
        ----------
        labels : ndarray, optional
            Class labels. For multi-class (num_outputs > 1): class indices
            (0.0, 1.0, 2.0, ...). For binary (num_outputs == 1): values
            in {-1.0, +1.0}. Same alignment convention as ``r2()``.
            If omitted, uses targets stored by ``fit()``.
        start : int, optional
            First timestep index. Default: 0, or ``train_size`` after ``fit()``.
        count : int, optional
            Number of timesteps to evaluate. Default: all remaining from ``start``.

        Returns
        -------
        float
            Fraction correct in [0.0, 1.0].
        """
        labels, start, count = self._resolve_eval_args(labels, start, count)
        return self._impl.accuracy(labels, start, count)

    def _resolve_eval_args(
        self,
        targets: np.ndarray | None,
        start: int | None,
        count: int | None,
    ) -> tuple[np.ndarray, int, int]:
        """Resolve optional targets/start/count for r2/nrmse/accuracy."""
        if targets is None:
            # Use stored targets from fit()
            if self._targets is None:
                raise ValueError(
                    "No targets available. Either call fit() first, or pass "
                    "targets explicitly."
                )
            targets = self._targets
            if start is None:
                start = self._train_size
            if count is None:
                count = self.num_collected - start
        else:
            targets = _to_float32(targets)
            if start is None:
                start = 0
            if count is None:
                count = self.num_collected - start
            if len(targets) < start + count:
                raise ValueError(
                    f"targets too short ({len(targets)}) for start={start}, "
                    f"count={count} (need >= {start + count}). targets must be "
                    f"index-aligned with collected states — do not slice the "
                    f"array before passing. Use the start parameter instead."
                )
        return targets, start, count

    def selected_states(self) -> np.ndarray:
        """Return stride-selected states.

        Returns
        -------
        ndarray
            Array of shape ``(num_collected, num_output_verts)``.
        """
        return self._impl.selected_states()

    @property
    def dim(self) -> int:
        """Hypercube dimension."""
        return self._impl.dim

    @property
    def N(self) -> int:
        """Number of neurons (2^dim)."""
        return self._impl.N

    @property
    def num_collected(self) -> int:
        """Number of timesteps recorded by ``run()``."""
        return self._impl.num_collected

    @property
    def num_outputs(self) -> int:
        """Number of readout outputs (after training)."""
        return self._impl.num_outputs

    @property
    def num_inputs(self) -> int:
        """Number of input channels."""
        return self._impl.num_inputs

    @property
    def output_fraction(self) -> float:
        """Fraction of vertices used as readout features."""
        return self._impl.output_fraction

    @property
    def num_output_verts(self) -> int:
        """Number of selected output vertices."""
        return self._impl.num_output_verts

    @property
    def history_depth(self) -> int:
        """Delay-line depth M (past output slices the readout sees)."""
        return self._impl.history_depth

    @property
    def seed(self) -> int:
        """RNG seed used to initialize reservoir weights."""
        return self._impl.seed

    @property
    def spectral_radius(self) -> float:
        """Target spectral radius of the recurrent weight matrix."""
        return self._impl.spectral_radius

    @property
    def leak_rate(self) -> float:
        """Leaky integrator coefficient."""
        return self._impl.leak_rate

    @property
    def input_scaling(self) -> float:
        """Input drive coefficient."""
        return self._impl.input_scaling

    @property
    def train_size(self) -> int | None:
        """Number of training samples from ``fit()``, or None."""
        return self._train_size

    @property
    def test_size(self) -> int | None:
        """Number of test samples from ``fit()``, or None."""
        if self._train_size is None:
            return None
        return self.num_collected - self._train_size

    def __repr__(self) -> str:
        parts = [
            f"ESN(dim={self.dim}, N={self.N}",
            f"collected={self.num_collected}",
        ]
        if self._train_size is not None:
            parts.append(f"train={self._train_size}, test={self.test_size}")
        return ", ".join(parts) + ")"

    # ── Persistence ──

    _PERSISTENCE_VERSION = 1

    def __getstate__(self) -> dict:
        """Serialize ESN state for pickling.

        Persists the constructor config (reservoir + readout) and the
        trained readout state. Collected states, cached features, and
        fit() targets are NOT saved.
        """
        return {
            "_version": self._PERSISTENCE_VERSION,
            "dim": self.dim,
            "seed": self.seed,
            "spectral_radius": self.spectral_radius,
            "input_scaling": self.input_scaling,
            "leak_rate": self.leak_rate,
            "num_inputs": self.num_inputs,
            "history_depth": self.history_depth,
            "output_fraction": self.output_fraction,
            "readout_kwargs": dict(self._readout_kwargs),
            "readout_state": self._impl._get_readout_state(),
        }

    def __setstate__(self, state: dict) -> None:
        """Restore ESN from pickled state."""
        version = state.get("_version", 0)
        if version > self._PERSISTENCE_VERSION:
            raise ValueError(
                f"Model was saved with persistence version {version}, "
                f"but this version only supports up to "
                f"{self._PERSISTENCE_VERSION}. Upgrade hypercube-esn."
            )
        readout_kwargs = state.get("readout_kwargs", {})
        self.__init__(
            dim=state["dim"],
            seed=state["seed"],
            spectral_radius=state["spectral_radius"],
            input_scaling=state["input_scaling"],
            leak_rate=state["leak_rate"],
            num_inputs=state["num_inputs"],
            history_depth=state["history_depth"],
            output_fraction=state["output_fraction"],
            **readout_kwargs,
        )
        self._impl._set_readout_state(state["readout_state"])

    def save(self, path) -> None:
        """Save the trained ESN to a file.

        Saves the reservoir configuration and trained readout weights.
        Collected states and fit() targets are NOT saved — the file is
        compact (typically < 1 MB).

        The file is a standard Python pickle.

        Parameters
        ----------
        path : str or Path
            File path to write.

        Examples
        --------
        >>> esn.fit(signal, warmup=200)
        >>> esn.save("model.pkl")
        >>> loaded = he.ESN.load("model.pkl")
        """
        with open(pathlib.Path(path), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path) -> "ESN":
        """Load a saved ESN from a file.

        Parameters
        ----------
        path : str or Path
            File path to read.

        Returns
        -------
        ESN
            The restored ESN with its trained readout intact.

        Notes
        -----
        The restored ESN has zero collected states. To make predictions
        on new data, call ``warmup()`` and ``run()`` first.

        Examples
        --------
        >>> esn = he.ESN.load("model.pkl")
        >>> esn.warmup(new_signal[:200])
        >>> esn.run(new_signal[200:])
        >>> preds = esn.predictions()
        """
        with open(pathlib.Path(path), "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected ESN, got {type(obj).__name__}")
        return obj
