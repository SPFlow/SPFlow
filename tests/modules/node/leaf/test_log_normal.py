import unittest

from tests.modules.node.leaf.utils import evaluate_log_likelihood
from tests.fixtures import set_seed
import numpy as np
import pytest
import scipy
import torch
from pytest import raises

from spflow import maximum_likelihood_estimation, sample
from spflow.meta.data import Scope
from spflow.modules.node.leaf.log_normal_old import LogNormal


def make_leaf(mean=0.0, std=1.0):
    return LogNormal(scope=Scope([0]), mean=mean, std=std)


def make_data(mean=0.0, std=1.0, n_samples=5):
    return torch.distributions.LogNormal(loc=mean, scale=std).sample((n_samples, 1))


def test_sample():
    mean = 0.7
    std = 0.3
    leaf = make_leaf(mean=mean, std=std)
    samples = sample(leaf, num_samples=500)

    # TODO: what can we test here?


def test_log_likelihood():
    evaluate_log_likelihood(make_leaf(), make_data())


@pytest.mark.parametrize("bias_correction", [True, False])
def test_maximum_likelihood_estimation(bias_correction):
    mean = 0.7
    std = 0.3
    leaf = make_leaf(mean=mean, std=std)
    data = make_data(mean=mean, std=std, n_samples=10000)
    maximum_likelihood_estimation(leaf, data, bias_correction=bias_correction)
    assert np.isclose(leaf.mean.detach(), mean, atol=1e-2)
    assert np.isclose(leaf.std.detach(), std, atol=1e-2)


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
    leaf = make_leaf()
    assert leaf.mean.requires_grad
    assert leaf.log_std.requires_grad


if __name__ == "__main__":
    unittest.main()
