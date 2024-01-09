import unittest
import numpy as np
import scipy
from spflow.meta.data import Scope
from spflow import sample, maximum_likelihood_estimation
from spflow.modules.node.leaf.categorical import Categorical
from spflow import tensor as T
from utils import compare_spflow_with_scipy_dist
from pytest import raises
from tests.fixtures import backend_auto


def make_leaf(probs=[0.1, 0.2, 0.7]):
    return Categorical(scope=Scope([0]), probs=probs)


def make_data():
    return T.tensor([0, 1, 2]).reshape((-1, 1))


def test_sample():
    probs = [0.3, 0.7]
    leaf = make_leaf(probs=probs)
    samples = sample(leaf, num_samples=10000)
    p_0 = (samples == 0).sum() / samples.shape[0]
    p_1 = (samples == 1).sum() / samples.shape[0]
    assert np.isclose(p_0, probs[0], atol=1e-1)
    assert np.isclose(p_1, probs[1], atol=1e-1)


def test_log_likelihood():
    probs = [0.1, 0.2, 0.7]
    leaf = make_leaf(probs=probs)
    data = make_data()

    from scipy.stats import rv_discrete

    # Create a custom discrete distribution
    values = np.arange(len(probs))  # assuming data indices match with probability indices
    custom_dist = rv_discrete(name="custom", values=(values, probs))
    compare_spflow_with_scipy_dist(leaf, custom_dist.logpmf, data)


def test_maximum_likelihood_estimation():
    leaf = make_leaf(probs=[0.1, 0.2, 0.7])
    data = T.reshape(T.tensor([0, 1, 1, 2, 2, 2]), (-1, 1))
    maximum_likelihood_estimation(leaf, data)
    assert np.isclose(leaf.probs[0].item(), 1 / 6, atol=1e-2)
    assert np.isclose(leaf.probs[1].item(), 2 / 6, atol=1e-2)
    assert np.isclose(leaf.probs[2].item(), 3 / 6, atol=1e-2)


def test_constructor():
    # Check that parameters are set correcT.
    probs = [0.1, 0.2, 0.7]
    leaf = make_leaf(probs=probs)
    assert np.isclose(T.tensor(leaf.probs, requires_grad=False, copy=True), probs).all()

    # Check invalid parameters
    with raises(ValueError):
        leaf = make_leaf(probs=[0.1, 0.2, 0.7, 0.1])
        leaf = make_leaf(probs=[0.1, 0.2, 0.7, -0.1])
        leaf = make_leaf(probs=[0.1, 0.2, 0.7, 1.1])


def test_requires_grad():
    if T.get_backend() != T.Backend.PYTORCH:
        return
    leaf = make_leaf()
    assert leaf.probs.requires_grad


if __name__ == "__main__":
    unittest.main()
