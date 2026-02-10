import pickle
from contextlib import contextmanager

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spflow.exceptions import InvalidParameterError, InvalidTypeError, OptionalDependencyError
from spflow.interfaces import sklearn as sklearn_interface
from spflow.interfaces.sklearn import (
    SPFlowDensityEstimator,
    _as_2d_numpy,
    _reduce_log_likelihood,
    _require_sklearn,
    _torch_dtype_from_str,
)
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.normal import Normal
from spflow.modules.products.product import Product


def _independent_normals_model(n_features: int) -> Product:
    leaves = [Normal(scope=Scope([i]), out_channels=1) for i in range(n_features)]
    return Product(leaves)


def _randn(*shape: int) -> np.ndarray:
    return np.random.standard_normal(shape).astype(np.float32)


def test_fit_and_score_samples_shapes():
    X = _randn(128, 3)

    est = SPFlowDensityEstimator(structure_learner="learn_spn", dtype="float32", min_instances_slice=20)
    est.fit(X)

    scores = est.score_samples(X[:7])
    assert scores.shape == (7,)
    assert np.issubdtype(scores.dtype, np.floating)


def test_score_samples_matches_direct_log_likelihood():
    X = _randn(64, 4)
    x_tensor = torch.tensor(X, dtype=torch.float32)

    model = _independent_normals_model(n_features=X.shape[1])
    model.maximum_likelihood_estimation(x_tensor)

    est = SPFlowDensityEstimator(model=model, fit_params=False, dtype="float32")
    est.fit(X)

    direct_ll = model.log_likelihood(x_tensor).sum(dim=1).squeeze(-1).squeeze(-1)  # (B,)
    np.testing.assert_allclose(est.score_samples(X), direct_ll.detach().cpu().numpy(), rtol=1e-6, atol=1e-6)


def test_sample_shape_and_dtype():
    X = _randn(128, 2)
    est = SPFlowDensityEstimator(structure_learner="learn_spn", dtype="float32", min_instances_slice=20)
    est.fit(X)

    samples = est.sample(11, random_state=0)
    assert samples.shape == (11, 2)
    assert samples.dtype == np.float32


def test_pipeline_compatibility_smoke():
    X = _randn(200, 2)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "spn",
                SPFlowDensityEstimator(
                    structure_learner="learn_spn", dtype="float32", min_instances_slice=30
                ),
            ),
        ]
    )
    pipe.fit(X)
    scores = pipe["spn"].score_samples(pipe["scaler"].transform(X[:5]))
    assert scores.shape == (5,)


def test_clone_and_pickle_roundtrip():
    X = _randn(64, 3)

    est = SPFlowDensityEstimator(structure_learner="learn_spn", dtype="float32", min_instances_slice=20)
    cloned = clone(est)
    assert isinstance(cloned, SPFlowDensityEstimator)
    assert cloned.get_params()["structure_learner"] == "learn_spn"

    est.fit(X)
    dumped = pickle.dumps(est)
    loaded = pickle.loads(dumped)
    np.testing.assert_allclose(loaded.score_samples(X[:8]), est.score_samples(X[:8]), rtol=0.0, atol=0.0)


def test_require_sklearn_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(sklearn_interface, "_SKLEARN_AVAILABLE", False)
    with pytest.raises(OptionalDependencyError):
        _require_sklearn()


def test_as_2d_numpy_handles_1d_and_rejects_nd():
    one_d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = _as_2d_numpy(one_d)
    assert out.shape == (3, 1)

    with pytest.raises(InvalidParameterError):
        _as_2d_numpy(np.zeros((2, 2, 2), dtype=np.float32))


def test_torch_dtype_from_str_variants():
    assert _torch_dtype_from_str(None) is None
    assert _torch_dtype_from_str("float64") is torch.float64
    with pytest.raises(InvalidParameterError):
        _torch_dtype_from_str("float16")


def test_reduce_log_likelihood_reductions_and_errors():
    ll2 = torch.randn(3, 4)
    out2 = _reduce_log_likelihood(ll2, channel_agg="first", repetition_agg="first")
    assert out2.shape == (3,)

    ll3 = torch.randn(2, 5, 2)
    out3 = _reduce_log_likelihood(ll3, channel_agg="logsumexp", repetition_agg="first")
    assert out3.shape == (2,)

    ll4 = torch.randn(2, 3, 2, 2)
    out4 = _reduce_log_likelihood(ll4, channel_agg="logmeanexp", repetition_agg="logsumexp")
    assert out4.shape == (2,)

    empty = torch.empty((0, 4, 1, 1))
    out_empty = _reduce_log_likelihood(empty, channel_agg="first", repetition_agg="first")
    assert out_empty.shape == (0,)

    with pytest.raises(InvalidParameterError):
        _reduce_log_likelihood(torch.randn(2), channel_agg="first", repetition_agg="first")

    with pytest.raises(InvalidParameterError):
        _reduce_log_likelihood(torch.randn(2, 3, 2, 2), channel_agg="bad", repetition_agg="first")


def test_fit_prometheus_forwards_kwargs(monkeypatch):
    X = _randn(20, 2)
    captured: dict[str, object] = {}

    def _fake_learn_prometheus(x_tensor, *, leaf_modules, **kwargs):
        captured["shape"] = tuple(x_tensor.shape)
        captured["kwargs"] = kwargs
        assert leaf_modules is not None
        return _independent_normals_model(n_features=x_tensor.shape[1])

    monkeypatch.setattr(sklearn_interface, "learn_prometheus", _fake_learn_prometheus)

    est = SPFlowDensityEstimator(
        structure_learner="prometheus",
        structure_learner_kwargs={"out_channels": 3, "min_instances_slice": 7},
        dtype="float32",
    )
    est.fit(X)
    assert captured["shape"] == (20, 2)
    assert captured["kwargs"] == {"out_channels": 3, "min_instances_slice": 7, "min_features_slice": 2}


def test_fit_rejects_unknown_leaf_and_structure_learner():
    X = _randn(8, 2)

    with pytest.raises(InvalidParameterError):
        SPFlowDensityEstimator(leaf="bad").fit(X)  # type: ignore[arg-type]

    with pytest.raises(InvalidParameterError):
        SPFlowDensityEstimator(structure_learner="bad").fit(X)  # type: ignore[arg-type]


def test_fit_rejects_non_module_model():
    class _NonModule:
        device = "cpu"

    X = _randn(6, 2)
    with pytest.raises(InvalidTypeError):
        SPFlowDensityEstimator(model=_NonModule()).fit(X)


def test_fit_calls_mle_when_fit_params_enabled(monkeypatch):
    X = _randn(16, 2)
    model = _independent_normals_model(n_features=2)
    called: dict[str, object] = {}

    def _fake_mle(x_tensor):
        called["shape"] = tuple(x_tensor.shape)

    monkeypatch.setattr(model, "maximum_likelihood_estimation", _fake_mle)

    est = SPFlowDensityEstimator(model=model, fit_params=True, dtype="float32")
    est.fit(X)
    assert est.model_ is model
    assert called["shape"] == (16, 2)


def test_fit_uses_explicit_device_override():
    X = _randn(10, 2)
    est = SPFlowDensityEstimator(structure_learner="learn_spn", device="cpu", dtype="float32")
    est.fit(X)
    assert est.n_features_in_ == 2


def test_sample_validates_arguments():
    X = _randn(24, 2)
    est = SPFlowDensityEstimator(structure_learner="learn_spn", dtype="float32", min_instances_slice=12)
    est.fit(X)

    with pytest.raises(InvalidParameterError):
        est.sample(0)

    with pytest.raises(InvalidTypeError):
        est.sample(1, random_state="bad")  # type: ignore[arg-type]


def test_sample_uses_cuda_device_index_for_fork_rng(monkeypatch):
    class _DummyModel:
        def sample(self, num_samples: int):
            return torch.zeros((num_samples, 2), dtype=torch.float32)

    captured: dict[str, object] = {}

    @contextmanager
    def _fake_fork_rng(*, devices):
        captured["devices"] = devices
        yield

    est = SPFlowDensityEstimator(model=None)
    est.model_ = _DummyModel()

    monkeypatch.setattr(est, "_device", lambda: torch.device("cuda:0"))
    monkeypatch.setattr(torch.random, "fork_rng", _fake_fork_rng)

    out = est.sample(3, random_state=1)
    assert out.shape == (3, 2)
    assert captured["devices"] == [0]
