import pickle

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spflow.interfaces.sklearn import SPFlowDensityEstimator
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.normal import Normal
from spflow.modules.products.product import Product


def _independent_normals_model(n_features: int) -> Product:
    leaves = [Normal(scope=Scope([i]), out_channels=1) for i in range(n_features)]
    return Product(leaves)


def test_fit_and_score_samples_shapes():
    X = np.random.randn(128, 3).astype(np.float32)

    est = SPFlowDensityEstimator(structure_learner="learn_spn", dtype="float32", min_instances_slice=20)
    est.fit(X)

    scores = est.score_samples(X[:7])
    assert scores.shape == (7,)
    assert np.issubdtype(scores.dtype, np.floating)


def test_score_samples_matches_direct_log_likelihood():
    X = np.random.randn(64, 4).astype(np.float32)
    x_tensor = torch.tensor(X, dtype=torch.float32)

    model = _independent_normals_model(n_features=X.shape[1])
    model.maximum_likelihood_estimation(x_tensor)

    est = SPFlowDensityEstimator(model=model, fit_params=False, dtype="float32")
    est.fit(X)

    direct_ll = model.log_likelihood(x_tensor).sum(dim=1).squeeze(-1).squeeze(-1)  # (B,)
    np.testing.assert_allclose(est.score_samples(X), direct_ll.detach().cpu().numpy(), rtol=1e-6, atol=1e-6)


def test_sample_shape_and_dtype():
    X = np.random.randn(128, 2).astype(np.float32)
    est = SPFlowDensityEstimator(structure_learner="learn_spn", dtype="float32", min_instances_slice=20)
    est.fit(X)

    samples = est.sample(11, random_state=0)
    assert samples.shape == (11, 2)
    assert samples.dtype == np.float32


def test_pipeline_compatibility_smoke():
    X = np.random.randn(200, 2).astype(np.float32)

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
    X = np.random.randn(64, 3).astype(np.float32)

    est = SPFlowDensityEstimator(structure_learner="learn_spn", dtype="float32", min_instances_slice=20)
    cloned = clone(est)
    assert isinstance(cloned, SPFlowDensityEstimator)
    assert cloned.get_params()["structure_learner"] == "learn_spn"

    est.fit(X)
    dumped = pickle.dumps(est)
    loaded = pickle.loads(dumped)
    np.testing.assert_allclose(loaded.score_samples(X[:8]), est.score_samples(X[:8]), rtol=0.0, atol=0.0)
