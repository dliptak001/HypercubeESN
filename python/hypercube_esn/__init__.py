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

from ._core import _ESN, _EnsembleESN

__version__ = "1.4.0"
__all__ = ["ESN", "EnsembleESN"]

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
    history_floor : float
        Depth taper K in [0.1, 1.0]: recurrent weights are linearly scaled from
        just below 1.0 at the most-recent history slice down to K at the deepest,
        so older states influence the next state less. Applied before the
        spectral-radius rescale (which normalizes overall magnitude). 1.0 = no
        taper (default); has no effect when history_depth == 1.
    verbose : bool
        Print the one-line reservoir construction banner. Default: True.
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
        Peak learning rate (cosine schedule). Keep <= 0.005 to avoid NaN. Default: 0.0015.
    readout_lr_min_frac : float
        LR floor as a fraction of lr_max. Default: 0.01.
    readout_lr_decay_epochs : int
        Cosine decay horizon. 0 = use ``readout_epochs``. Default: 0.
    readout_weight_decay : float
        L2 regularization on CNN weights. Default: 0.0.
    readout_momentum : float
        Heavy-ball SGD momentum. 0 = plain SGD; 0.9 is typical for CNN
        training. Default: 0.0.
    readout_activation : str
        Per-Conv-layer activation: "tanh" (default), "relu", "leaky_relu",
        or "none".
    readout_seed : int
        CNN weight initialization seed. Default: 42.

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
        history_floor: float = 1.0,
        verbose: bool = True,
        readout_num_outputs: int = 1,
        readout_task: str = "regression",
        readout_num_layers: int = 0,
        readout_conv_channels: int = 16,
        readout_epochs: int = 200,
        readout_batch_size: int = 32,
        readout_lr_max: float = 0.0015,
        readout_lr_min_frac: float = 0.01,
        readout_lr_decay_epochs: int = 0,
        readout_weight_decay: float = 0.0,
        readout_momentum: float = 0.0,
        readout_activation: str = "tanh",
        readout_seed: int = 42,
    ):
        if not (_DIM_MIN <= dim <= _DIM_MAX):
            raise ValueError(f"dim must be {_DIM_MIN}-{_DIM_MAX}, got {dim}")
        self._verbose = verbose
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
            "readout_momentum": readout_momentum,
            "readout_activation": readout_activation,
            "readout_seed": readout_seed,
        }
        self._impl = _ESN(
            dim=dim,
            seed=seed,
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            leak_rate=leak_rate,
            num_inputs=num_inputs,
            history_depth=history_depth,
            history_floor=history_floor,
            verbose=verbose,
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

    def reset_reservoir_only(self) -> None:
        """Zero the reservoir's live internal state only.

        Collected states and the trained readout are preserved. Useful for
        episodic tasks where each sequence should start from a clean reservoir
        without discarding previously collected training data.
        """
        self._impl.reset_reservoir_only()

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

        Raises
        ------
        ValueError
            If the readout has more than one output. Scalar prediction
            requires ``num_outputs == 1``; use :meth:`predict_raw_multi` for
            multi-output readouts.
        """
        return self._impl.predict_raw(timestep)

    def predict_raw_multi(self, timestep: int) -> np.ndarray:
        """Multi-output prediction for a single collected timestep.

        Parameters
        ----------
        timestep : int
            Index into collected states, in [0, num_collected).

        Returns
        -------
        ndarray
            1D array of shape ``(num_outputs,)`` with float32 predictions.
            Works for any ``num_outputs`` (including 1).
        """
        return self._impl.predict_raw_multi(timestep)

    def predict_live_raw(self) -> float:
        """Predict from the reservoir's current live state (single output).

        For autoregressive / closed-loop inference loops: drive the reservoir
        one step (``run``/``warmup``), then read the prediction here without
        touching the collected-state buffer.

        Returns
        -------
        float
            Continuous prediction value.

        Raises
        ------
        ValueError
            If the readout has more than one output. Use
            :meth:`predict_live_raw_multi` for multi-output readouts.
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

    def predict_from_state(self, state: np.ndarray) -> np.ndarray:
        """Run the readout on a caller-supplied reservoir state.

        Parameters
        ----------
        state : ndarray
            A reservoir state of shape ``(num_output_verts,)``,
            e.g. one returned by :meth:`copy_live_state`. Converted to float32.

        Returns
        -------
        ndarray
            1D array of shape ``(num_outputs,)`` with float32 predictions.
        """
        return self._impl.predict_from_state(_to_float32(state))

    def predictions(self) -> np.ndarray:
        """Return predictions for all collected timesteps.

        Returns
        -------
        ndarray
            1D array of shape ``(num_collected,)`` with float32 predictions.

        Raises
        ------
        ValueError
            If the readout has more than one output (this returns one scalar
            per timestep). Use :meth:`predictions_multi` to retrieve all
            channels, or :meth:`r2` / :meth:`nrmse` to evaluate them.
        """
        return self._impl.predictions()

    def predictions_multi(self) -> np.ndarray:
        """Return multi-output predictions for all collected timesteps.

        Returns
        -------
        ndarray
            2D array of shape ``(num_collected, num_outputs)`` with float32
            predictions. Works for any ``num_outputs`` (including 1).
        """
        return self._impl.predictions_multi()

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

    def collected_states(self) -> np.ndarray:
        """Return all collected states.

        Returns
        -------
        ndarray
            Array of shape ``(num_collected, num_output_verts)``.
        """
        return self._impl.collected_states()

    # ── Streaming / online training ──

    def init_online(self, warmup_inputs: np.ndarray) -> None:
        """Initialize the readout for online (streaming) training.

        Runs ``warmup_inputs`` through the reservoir to reach a representative
        state, then builds the readout CNN. Call before any ``train_live_*``
        method. Uses the readout configuration supplied at construction.

        Parameters
        ----------
        warmup_inputs : ndarray
            Warmup signal. Same shape convention as :meth:`warmup`.
        """
        self._impl.init_online(_to_float32(warmup_inputs))

    def copy_live_state(self) -> np.ndarray:
        """Copy the current reservoir state.

        Returns
        -------
        ndarray
            1D array of shape ``(num_output_verts,)``. Accumulate these across
            steps to build the ``states`` batch for ``train_live_batch*``.
        """
        return self._impl.copy_live_state()

    def train_live_step(
        self, target_class: float, lr: float, weight_decay: float = 0.0
    ) -> None:
        """Single-sample online classification step on the live reservoir state.

        Parameters
        ----------
        target_class : float
            Class index as a float (0.0, 1.0, ...).
        lr : float
            Learning rate for this step.
        weight_decay : float, optional
            L2 weight decay. Default 0.0.
        """
        self._impl.train_live_step(target_class, lr, weight_decay)

    def train_live_batch(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        lr: float,
        weight_decay: float = 0.0,
    ) -> None:
        """Mini-batch online classification step on pre-accumulated states.

        Parameters
        ----------
        states : ndarray
            Shape ``(count, num_output_verts)`` of states from
            :meth:`copy_live_state`. Converted to float32.
        targets : ndarray
            Shape ``(count,)`` of integer class indices.
        lr : float
            Learning rate.
        weight_decay : float, optional
            L2 weight decay. Default 0.0.
        """
        targets = np.ascontiguousarray(targets, dtype=np.int32)
        self._impl.train_live_batch(_to_float32(states), targets, lr, weight_decay)

    def train_live_step_regression(
        self, target: np.ndarray, lr: float, weight_decay: float = 0.0
    ) -> None:
        """Single-sample online regression step on the live reservoir state.

        Parameters
        ----------
        target : ndarray
            Shape ``(num_outputs,)`` target values. Converted to float32.
        lr : float
            Learning rate.
        weight_decay : float, optional
            L2 weight decay. Default 0.0.
        """
        self._impl.train_live_step_regression(_to_float32(target), lr, weight_decay)

    def train_live_batch_regression(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        lr: float,
        weight_decay: float = 0.0,
    ) -> None:
        """Mini-batch online regression step on pre-accumulated states.

        Parameters
        ----------
        states : ndarray
            Shape ``(count, num_output_verts)`` of states from
            :meth:`copy_live_state`. Converted to float32.
        targets : ndarray
            Shape ``(count, num_outputs)`` target values. Converted to float32.
        lr : float
            Learning rate.
        weight_decay : float, optional
            L2 weight decay. Default 0.0.
        """
        self._impl.train_live_batch_regression(
            _to_float32(states), _to_float32(targets), lr, weight_decay
        )

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
    def num_output_verts(self) -> int:
        """Number of readout-input vertices (all N reservoir vertices)."""
        return self._impl.num_output_verts

    @property
    def history_depth(self) -> int:
        """Delay-line depth M (past output slices the readout sees)."""
        return self._impl.history_depth

    @property
    def history_floor(self) -> float:
        """Depth taper K: deepest-history recurrent weight scale (1.0 = no taper)."""
        return self._impl.history_floor

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
    def verbose(self) -> bool:
        """Whether the reservoir prints its construction banner."""
        return self._verbose

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

    # Bumped to 2 in 1.1.0: the serialized config gained the `verbose` field and
    # new readout_kwargs keys (readout_momentum, readout_activation,
    # readout_verbose_train_acc). Bumping ensures an older library rejects a
    # newer pickle with the friendly "Upgrade hypercube-esn" message in
    # __setstate__ rather than crashing on unexpected __init__ kwargs.
    # Bumped to 3: the serialized config gained the `history_floor` depth-taper
    # field.
    # Bumped to 4: the serialized config gained the `noise_scaling` /
    # `noise_seed` training-noise fields.
    # Bumped to 5: the training-noise feature was removed; `noise_scaling` /
    # `noise_seed` are no longer written. Any keys present in an older (v4)
    # pickle are ignored on load.
    # Bumped to 6: the readout verbose feature was removed; `readout_verbose` /
    # `readout_verbose_train_acc` are no longer written, and any present in an
    # older (v5) pickle's readout_kwargs are stripped on load.
    _PERSISTENCE_VERSION = 6

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
            "history_floor": self.history_floor,
            "verbose": self._verbose,
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
        readout_kwargs = dict(state.get("readout_kwargs", {}))
        # v6 removed the readout verbose feature; drop the keys if an older
        # (v5) pickle still carries them so __init__ doesn't reject them.
        readout_kwargs.pop("readout_verbose", None)
        readout_kwargs.pop("readout_verbose_train_acc", None)
        self.__init__(
            dim=state["dim"],
            seed=state["seed"],
            spectral_radius=state["spectral_radius"],
            input_scaling=state["input_scaling"],
            leak_rate=state["leak_rate"],
            num_inputs=state["num_inputs"],
            history_depth=state["history_depth"],
            history_floor=state.get("history_floor", 1.0),
            verbose=state.get("verbose", True),
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


class EnsembleESN:
    """Consensus feedback coupling of M ESN members.

    M members share one base configuration and differ only by a derived
    reservoir seed, so they share the hypercube topology but carry different
    random weight realizations (decorrelating their errors). Every online step,
    the ensemble reads each member's output, forms the consensus ``c`` (mean or
    median), and injects each member's scaled deviation ``phi_i = kappa*(y_i-c)``
    on its feedback channels before stepping — a single-step closed loop.

    The class is **online-only and feedback-only**: there is no batch path and no
    feedback-less mode. The coupling intensity ``kappa`` rises on a
    competence-gated ramp the class owns — held at ``kappa_start`` until the
    smoothed consensus error crosses ``gate_threshold``, then ramped to
    ``kappa_target``.

    "One D, three roles": the output dimension, the readout output count, and the
    per-member feedback-channel count are the same number — you pass it once as
    ``num_outputs``.

    Parameters
    ----------
    dim : int
        Hypercube dimension (5-16); each member has N = 2^dim neurons.
    num_members : int
        M, the member count. Default 3 (the smallest M for which a median is
        meaningful). M >= 1; ``combine="median"`` requires M >= 3.
    combine : str
        Consensus statistic: ``"mean"`` (default) or ``"median"`` (per channel).
    ensemble_seed : int
        Master seed; per-member seeds are derived from it. Equal ensemble seeds
        reproduce the whole ensemble exactly. Default: 73895.
    num_outputs : int
        Output dimension D = readout outputs = feedback channels. Default: 1.
    spectral_radius, input_scaling, leak_rate, num_inputs, history_depth, \
    history_floor, feedback_scaling, bias_scaling, lorentz_gamma, \
    lorentz_inv_sigma2 :
        Shared reservoir parameters (see :class:`ESN`). Identical across members.
    readout_num_layers, readout_conv_channels, readout_activation, readout_seed :
        Shared readout (HCNN) architecture. The readout trains online via
        regression; the batch cosine-schedule fields do not apply.
    lr, weight_decay : float
        Shared online learning rate / L2, passed to every member's readout each
        training step. Held constant through the kappa ramp. Defaults: 0.01, 0.0.
    washout : int
        W — initial steps with the readout update suppressed (transient washout).
        Default: 100.
    resequence_washout : int
        Short washout re-imposed at each :meth:`begin_sequence`. Default: 16.
    kappa_start, kappa_target : float
        Start and target coupling intensity. Defaults: 0.0, 0.5.
    kappa_ramp_rate : float
        Per-step linear increment of kappa once the gate opens. <= 0 snaps to
        ``kappa_target`` immediately. Default: 0.0 (snap).
    gate_threshold : float
        The ramp opens once the smoothed consensus |error| falls below this.
        **Default 0.0 means the gate never fires** (kappa stays at
        ``kappa_start`` — the kappa=0 baseline); set a positive value for a
        coupled run.
    gate_err_ema_alpha : float
        EMA factor for the running consensus-error estimate, in (0, 1].
        Default: 0.05.

    Examples
    --------
    >>> import numpy as np, hypercube_esn as he
    >>> ens = he.EnsembleESN(dim=6, num_members=3, num_outputs=1,
    ...                      gate_threshold=0.2, kappa_target=0.2)
    >>> sig = np.sin(0.1 * np.arange(4000)).astype(np.float32)
    >>> c = np.zeros(1, np.float32)
    >>> for t in range(len(sig) - 1):
    ...     c = ens.step(sig[t], sig[t + 1])     # input, next-step target
    >>> ens.kappa, ens.gate_open
    """

    _DIM_MIN = 5
    _DIM_MAX = 16

    def __init__(
        self,
        dim: int,
        *,
        num_members: int = 3,
        combine: str = "mean",
        ensemble_seed: int = 73895,
        num_outputs: int = 1,
        spectral_radius: float = 0.99,
        input_scaling: float = 0.5,
        leak_rate: float = 1.0,
        num_inputs: int = 1,
        history_depth: int = 16,
        history_floor: float = 1.0,
        feedback_scaling: float = 0.5,
        bias_scaling: float = 0.02,
        lorentz_gamma: float = 1.1,
        lorentz_inv_sigma2: float = 250.0,
        readout_num_layers: int = 0,
        readout_conv_channels: int = 16,
        readout_activation: str = "tanh",
        readout_seed: int = 42,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        washout: int = 100,
        resequence_washout: int = 16,
        kappa_start: float = 0.0,
        kappa_target: float = 0.5,
        kappa_ramp_rate: float = 0.0,
        gate_threshold: float = 0.0,
        gate_err_ema_alpha: float = 0.05,
    ):
        if not (self._DIM_MIN <= dim <= self._DIM_MAX):
            raise ValueError(f"dim must be {self._DIM_MIN}-{self._DIM_MAX}, got {dim}")
        # Full constructor config, retained for pickling: reconstructing with the
        # same config re-derives every member's reservoir seed identically, so a
        # restored ensemble's members match the saved ones before their trained
        # readout weights are loaded back (see __setstate__).
        self._config = {
            "dim": dim,
            "num_members": num_members,
            "combine": combine,
            "ensemble_seed": ensemble_seed,
            "num_outputs": num_outputs,
            "spectral_radius": spectral_radius,
            "input_scaling": input_scaling,
            "leak_rate": leak_rate,
            "num_inputs": num_inputs,
            "history_depth": history_depth,
            "history_floor": history_floor,
            "feedback_scaling": feedback_scaling,
            "bias_scaling": bias_scaling,
            "lorentz_gamma": lorentz_gamma,
            "lorentz_inv_sigma2": lorentz_inv_sigma2,
            "readout_num_layers": readout_num_layers,
            "readout_conv_channels": readout_conv_channels,
            "readout_activation": readout_activation,
            "readout_seed": readout_seed,
            "lr": lr,
            "weight_decay": weight_decay,
            "washout": washout,
            "resequence_washout": resequence_washout,
            "kappa_start": kappa_start,
            "kappa_target": kappa_target,
            "kappa_ramp_rate": kappa_ramp_rate,
            "gate_threshold": gate_threshold,
            "gate_err_ema_alpha": gate_err_ema_alpha,
        }
        self._impl = _EnsembleESN(
            dim=dim,
            ensemble_seed=ensemble_seed,
            num_members=num_members,
            combine=combine,
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            leak_rate=leak_rate,
            num_inputs=num_inputs,
            history_depth=history_depth,
            history_floor=history_floor,
            num_outputs=num_outputs,
            feedback_scaling=feedback_scaling,
            bias_scaling=bias_scaling,
            lorentz_gamma=lorentz_gamma,
            lorentz_inv_sigma2=lorentz_inv_sigma2,
            readout_num_layers=readout_num_layers,
            readout_conv_channels=readout_conv_channels,
            readout_activation=readout_activation,
            readout_seed=readout_seed,
            lr=lr,
            weight_decay=weight_decay,
            washout=washout,
            resequence_washout=resequence_washout,
            kappa_start=kappa_start,
            kappa_target=kappa_target,
            kappa_ramp_rate=kappa_ramp_rate,
            gate_threshold=gate_threshold,
            gate_err_ema_alpha=gate_err_ema_alpha,
        )

    def step(
        self, input: np.ndarray, target: np.ndarray | None = None
    ) -> np.ndarray:
        """One lockstep online step; returns the consensus output.

        Parameters
        ----------
        input : ndarray or float
            Task input for this step, shape ``(num_inputs,)`` (a scalar is
            accepted when ``num_inputs == 1``). Converted to float32.
        target : ndarray or float, optional
            Regression target, shape ``(num_outputs,)``. When given (and past the
            washout), each member's readout takes an online update toward it.
            Pass ``None`` for inference: no update is taken and kappa holds.

        Returns
        -------
        ndarray
            The consensus ``c`` for this step, shape ``(num_outputs,)`` float32 —
            the ensemble's output.
        """
        t = None if target is None else _to_float32(target)
        return self._impl.step(_to_float32(input), t)

    def begin_sequence(self) -> None:
        """Start a fresh, independent sequence.

        Resets every member's reservoir state together (trained readouts
        preserved) and re-imposes the short ``resequence_washout``. The kappa
        schedule, the competence already achieved, and the step counter are not
        rewound.
        """
        self._impl.begin_sequence()

    def member_output(self, i: int) -> np.ndarray:
        """Member ``i``'s last output (shape ``(num_outputs,)``).

        The value used to form the most recent consensus, not a fresh
        re-evaluation.
        """
        return self._impl.member_output(i)

    def all_member_outputs(self) -> np.ndarray:
        """All members' last outputs, shape ``(num_members, num_outputs)``."""
        return self._impl.all_member_outputs()

    @property
    def kappa(self) -> float:
        """Current coupling intensity (the operating point of the ramp)."""
        return self._impl.kappa

    @property
    def gate_open(self) -> bool:
        """Whether the competence-gated kappa ramp has triggered."""
        return self._impl.gate_open

    @property
    def current_step(self) -> int:
        """Monotone step counter t."""
        return self._impl.current_step

    @property
    def num_members(self) -> int:
        """Member count M."""
        return self._impl.num_members

    @property
    def num_outputs(self) -> int:
        """Output dimension D (= feedback channels)."""
        return self._impl.num_outputs

    @property
    def num_inputs(self) -> int:
        """Task input width."""
        return self._impl.num_inputs

    def __repr__(self) -> str:
        return (
            f"EnsembleESN(M={self.num_members}, D={self.num_outputs}, "
            f"step={self.current_step}, kappa={self.kappa:.4g}, "
            f"gate_open={self.gate_open})"
        )

    # ── Persistence ──

    # v1: initial EnsembleESN persistence — constructor config + per-member
    # readout weights + schedule/competence state (kappa, gate, consensus_err,
    # err_init, step). Reservoir live state is NOT saved (a restored ensemble has
    # cold reservoirs and re-washes out), matching ESN's persistence contract.
    _PERSISTENCE_VERSION = 1

    def __getstate__(self) -> dict:
        """Serialize the trained ensemble for pickling.

        Persists the constructor config and the trained state (every member's
        readout weights plus the kappa-ramp / competence state). The reservoirs'
        live dynamical state is NOT saved — a restored ensemble has cold
        reservoirs and re-washes out before training resumes (drive it through a
        warmup, or call :meth:`begin_sequence`, before trusting outputs).
        """
        return {
            "_version": self._PERSISTENCE_VERSION,
            "config": dict(self._config),
            "state": self._impl._get_state(),
        }

    def __setstate__(self, state: dict) -> None:
        """Restore an EnsembleESN from pickled state."""
        version = state.get("_version", 0)
        if version > self._PERSISTENCE_VERSION:
            raise ValueError(
                f"EnsembleESN was saved with persistence version {version}, "
                f"but this version only supports up to "
                f"{self._PERSISTENCE_VERSION}. Upgrade hypercube-esn."
            )
        self.__init__(**state["config"])
        self._impl._set_state(state["state"])

    def save(self, path) -> None:
        """Save the trained ensemble to a file (a standard Python pickle).

        Saves the configuration, every member's trained readout weights, and the
        kappa-ramp / competence state. Reservoir live state is NOT saved, so the
        file is compact. Restore with :meth:`load`.
        """
        with open(pathlib.Path(path), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path) -> "EnsembleESN":
        """Load a saved EnsembleESN from a file.

        The restored ensemble has cold reservoirs: drive it through a warmup
        (or :meth:`begin_sequence`) before trusting its outputs.
        """
        with open(pathlib.Path(path), "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected EnsembleESN, got {type(obj).__name__}")
        return obj
