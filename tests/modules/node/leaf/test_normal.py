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
from spflow.modules.node.leaf.normal import Normal


def make_leaf(mean=None, std=None):
    if mean is None:
        mean = torch.randn(1)
    if std is None:
        std = torch.rand(1) + 1e-8
    return Normal(scope=Scope(list(range(mean.shape[0]))), mean=mean, std=std)


def make_data(mean=None, std=None, n_samples=5):
    if mean is None:
        mean = torch.randn(1)
    if std is None:
        std = torch.rand(1) + 1e-8
    return torch.distributions.Normal(loc=mean, scale=std).sample((n_samples,))


def test_log_likelihood():
    evaluate_log_likelihood(make_leaf(), make_data())


def test_sample():
    mean = torch.randn(1)
    std = torch.rand(1)
    leaf = make_leaf(mean=mean, std=std)
    samples = sample(leaf, num_samples=500)
    assert torch.isclose(samples.mean(0), mean, atol=1e-1)
    assert torch.isclose(samples.std(0), std, atol=1e-1)


@pytest.mark.parametrize("bias_correction", [True, False])
def test_maximum_likelihood_estimation(bias_correction):
    mean = torch.randn(1)
    std = torch.rand(1)
    leaf = make_leaf(mean=mean, std=std)
    mean = torch.randn(1)
    std = torch.rand(1)
    data = make_data(mean, std, n_samples=500)
    maximum_likelihood_estimation(leaf, data, bias_correction=bias_correction)
    assert torch.isclose(leaf.distribution.mean, data.mean(0), atol=1e-2).all()
    assert torch.isclose(leaf.distribution.std, data.std(0, unbiased=bias_correction), atol=1e-2).all()


def test_constructor():
    # Check that parameters are set correctly
    mean = torch.randn(1)
    std = torch.rand(1)
    leaf = make_leaf(mean=mean, std=std)
    assert torch.isclose(leaf.distribution.mean, mean).all()
    assert torch.isclose(leaf.distribution.std, std).all()

    # Check invalid parameters
    with raises(ValueError):
        leaf = make_leaf(mean=torch.rand(1), std=-1.0 * torch.rand(1))
        leaf = make_leaf(mean=torch.rand(1), std=0.0 * torch.rand(1))


if __name__ == "__main__":
    unittest.main()
