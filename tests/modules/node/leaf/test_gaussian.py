import unittest
import numpy as np
import scipy
from spflow.meta.data import Scope
from spflow import sample, maximum_likelihood_estimation
from spflow import tensor as T
from spflow.modules.node.leaf.gaussian import Gaussian
from utils import compare_spflow_with_scipy_dist
from pytest import raises
from tests.fixtures import backend_auto


def make_leaf(mean=0.0, std=1.0):
    return Gaussian(scope=Scope([0]), mean=mean, std=std)


def make_data(n=2, p=0.5):
    return T.reshape(T.tensor([0.4, 0.3, -0.1]), (-1, 1))


def test_sample():
    mean = 0.7
    std = 0.3
    leaf = make_leaf(mean=mean, std=std)
    samples = sample(leaf, num_samples=500)
    assert np.isclose(samples.mean().item(), mean, atol=1e-1)
    assert np.isclose(samples.std().item(), std, atol=1e-1)


def test_log_likelihood():
    mean = 0.7
    std = 0.3
    leaf = make_leaf(mean=mean, std=std)
    data = make_data()
    scipy_dist = scipy.stats.norm(leaf.mean.item(), leaf.std.item())
    compare_spflow_with_scipy_dist(leaf, scipy_dist.logpdf, data)


def test_maximum_likelihood_estimation():
    mean = 0.7
    std = 0.3
    leaf = make_leaf(mean=mean, std=std)
    data = make_data()
    maximum_likelihood_estimation(leaf, data, bias_correction=False)
    assert np.isclose(leaf.mean.item(), data.mean(), atol=1e-2)
    assert np.isclose(leaf.std.item(), np.std(T.tolist(data)), atol=1e-2)


def test_constructor():
    # Check that parameters are set correctly
    mean = 7.0
    std = 3.0
    leaf = make_leaf(mean=mean, std=std)
    assert leaf.mean == mean
    assert leaf.std == std

    # Check invalid parameters
    with raises(ValueError):
        leaf = make_leaf(std=-0.5)
        leaf = make_leaf(std=0.0)


def test_requires_grad():
    if T.get_backend() != T.Backend.PYTORCH:
        return
    leaf = make_leaf()
    assert leaf.mean.requires_grad
    assert leaf.std.requires_grad


if __name__ == "__main__":
    unittest.main()
