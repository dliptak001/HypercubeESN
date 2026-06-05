"""Essential smoke tests for the hypercube_esn wheel.

Kept deliberately lean so the cibuildwheel test step stays well under ~15s on
every Python × platform combo. The goal here is to prove the compiled
``_core`` extension loads and the end-to-end pipeline produces sane numbers on
the target platform — not to exhaustively cover the Python wrapper's argument
logic. Two small shared training runs (one regression, one classification) are
reused across tests; everything else is construction-only.
"""

import pickle

import numpy as np
import pytest

from hypercube_esn import ESN


# ── Shared fixtures (each trained once per module, cheaply) ──

@pytest.fixture(scope="module")
def sine():
    return np.sin(np.linspace(0, 16 * np.pi, 800)).astype(np.float32)


@pytest.fixture(scope="module")
def fitted(sine):
    """A trained regression ESN, reused by pipeline + persistence tests."""
    esn = ESN(dim=5, readout_epochs=80)
    esn.fit(sine, warmup=100)
    return esn, sine


@pytest.fixture(scope="module")
def classifier(sine):
    """A trained 2-class classifier, reused by classification tests."""
    labels = np.where(sine >= 0, 1.0, 0.0).astype(np.float32)
    esn = ESN(dim=5, readout_num_outputs=2, readout_task="classification",
              readout_epochs=60, readout_batch_size=32)
    esn.warmup(sine[:100])
    esn.run(sine[100:])
    esn.train(labels[100:600])
    return esn, sine, labels


# ── Construction & metadata (no training — catches load/ABI failures) ──

class TestConstruction:

    @pytest.mark.parametrize("dim", [5, 8])
    def test_construct(self, dim):
        esn = ESN(dim=dim)
        assert esn.dim == dim
        assert esn.N == 2 ** dim
        assert esn.num_collected == 0

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="dim must be 5-16"):
            ESN(dim=4)
        with pytest.raises(ValueError, match="dim must be 5-16"):
            ESN(dim=17)

    def test_defaults(self):
        esn = ESN(dim=5)
        assert esn.num_inputs == 1
        assert esn.output_fraction == pytest.approx(1.0)
        assert esn.history_depth == 16
        assert esn.seed == 73895


# ── Core regression pipeline (shared trained ESN) ──

class TestPipeline:

    def test_fit_r2_nrmse(self, fitted):
        esn, _ = fitted
        assert esn.r2() > 0.80, f"R² too low: {esn.r2()}"
        assert esn.nrmse() < 0.6, f"NRMSE too high: {esn.nrmse()}"

    def test_predictions_array(self, fitted):
        esn, _ = fitted
        preds = esn.predictions()
        assert preds.shape == (esn.num_collected,)
        assert preds.dtype == np.float32

    def test_train_test_split(self, fitted):
        esn, _ = fitted
        assert esn.train_size == int(esn.num_collected * 0.7)
        assert esn.test_size == esn.num_collected - esn.train_size

    def test_selected_states_shape(self, fitted):
        esn, _ = fitted
        states = esn.selected_states()
        assert states.shape == (esn.num_collected, esn.num_output_verts)
        assert states.dtype == np.float32


# ── Multi-input (contiguous-block channels) ──

class TestMultiInput:

    def test_run(self):
        esn = ESN(dim=5, num_inputs=4)  # 4 | 32
        rng = np.random.default_rng(0)
        inputs = (rng.standard_normal((300, 4)) * 0.1).astype(np.float32)
        esn.warmup(inputs[:100])
        esn.run(inputs[100:])
        assert esn.num_collected == 200

    def test_warmup_divisibility(self):
        # num_inputs must divide N = 2^dim; a 10-element drive is not
        # divisible by 4, so warmup must reject it.
        esn = ESN(dim=5, num_inputs=4)
        with pytest.raises(Exception, match="divisible"):
            esn.warmup(np.ones(10, dtype=np.float32))


# ── Classification head (shared trained classifier) ──

class TestClassification:

    def test_accuracy(self, classifier):
        esn, _, labels = classifier
        acc = esn.accuracy(labels[100:], 500, 200)
        assert acc > 0.7, f"Accuracy too low: {acc}"


# ── Persistence (reuses shared trained models) ──

class TestPersistence:

    def test_pickle_roundtrip(self, fitted, sine):
        esn, _ = fitted
        r2_before = esn.r2()
        loaded = pickle.loads(pickle.dumps(esn))
        assert loaded.num_collected == 0  # states are not persisted
        loaded.warmup(sine[:100])
        loaded.run(sine[100:-1])
        r2_after = loaded.r2(sine[101:], start=esn.train_size)
        assert abs(r2_before - r2_after) < 1e-5

    def test_save_load(self, fitted, sine, tmp_path):
        esn, _ = fitted
        path = tmp_path / "model.pkl"
        esn.save(path)
        loaded = ESN.load(path)
        assert loaded.dim == esn.dim
        loaded.warmup(sine[:100])
        loaded.run(sine[100:-1])
        assert abs(esn.r2() - loaded.r2(sine[101:], start=esn.train_size)) < 1e-5

    def test_preserves_config(self):
        esn = ESN(dim=8, seed=123, spectral_radius=0.85, input_scaling=0.05,
                  leak_rate=0.7, history_depth=8, num_inputs=2, output_fraction=0.5)
        loaded = pickle.loads(pickle.dumps(esn))
        assert loaded.dim == 8
        assert loaded.seed == 123
        assert loaded.history_depth == 8
        assert loaded.num_inputs == 2
        assert loaded.output_fraction == pytest.approx(0.5)

    def test_classification_roundtrip(self, classifier):
        esn, sine, labels = classifier
        acc_before = esn.accuracy(labels[100:], 500, 200)
        loaded = pickle.loads(pickle.dumps(esn))
        assert loaded.num_outputs == 2
        loaded.warmup(sine[:100])
        loaded.run(sine[100:])
        acc_after = loaded.accuracy(labels[100:], 500, 200)
        assert abs(acc_before - acc_after) < 1e-5

    def test_load_wrong_type_raises(self, tmp_path):
        path = tmp_path / "not_esn.pkl"
        with open(path, "wb") as f:
            pickle.dump({"not": "an ESN"}, f)
        with pytest.raises(TypeError, match="Expected ESN"):
            ESN.load(path)
