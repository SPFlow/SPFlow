import unittest
from tests.fixtures import set_seed
import numpy as np
import scipy
from spflow.meta.data import Scope
from spflow import sample, maximum_likelihood_estimation
from spflow.modules.node.leaf.uniform import Uniform
from pytest import raises
import torch


def make_leaf(low=0.5, high=1.8):
    return Uniform(scope=Scope([0]), low=low, high=high)


def make_data():
    return torch.rand(5).reshape((-1, 1))


def test_sample():
    low, high = 0.5, 1.8
    leaf = make_leaf(low=low, high=high)
    samples = sample(leaf, num_samples=10000)
    assert torch.all(samples >= low)
    assert torch.all(samples <= high)


def test_maximum_likelihood_estimation():
    low, high = 0.1, 1.7
    leaf = make_leaf(low, high)
    data = make_data() * (high - low) + low
    maximum_likelihood_estimation(leaf, data)

    # MLE does nothing for Uniform distributions since the bounds are fixed and there is no learnable parameter
    assert np.allclose(leaf.low.detach(), low, atol=1e-5)
    assert np.allclose(leaf.high.detach(), high, atol=1e-5)


def test_constructor():
    # Check that parameters are set correcT.
    low, high = 0.5, 1.8
    leaf = make_leaf(low=low, high=high)
    assert np.isclose(leaf.low.detach(), low)
    assert np.isclose(leaf.high.detach(), high)

    # Check invalid parameters
    with raises(ValueError):
        leaf = make_leaf(low=1.8, high=0.5)
        leaf = make_leaf(low=0.5, high=0.5)
        leaf = make_leaf(low=np.NINF, high=0.5)
        leaf = make_leaf(low=0.5, high=np.INF)


def test_requires_grad():
    leaf = make_leaf()
    assert not leaf.low.requires_grad
    assert not leaf.high.requires_grad


if __name__ == "__main__":
    unittest.main()
