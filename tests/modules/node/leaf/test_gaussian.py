import unittest

from tests.fixtures import set_seed
import numpy as np
import pytest
import scipy
import torch
from pytest import raises

from spflow import maximum_likelihood_estimation, sample
from spflow.meta.data import Scope
from spflow.modules.node.leaf.gaussian import Gaussian


def make_leaf(mean=0.0, std=1.0):
    return Gaussian(scope=Scope([0]), mean=mean, std=std)


def make_data(n=2, p=0.5):
    return torch.tensor([0.4, 0.3, -0.1]).view(-1, 1)


def test_sample():
    mean = 0.7
    std = 0.3
    leaf = make_leaf(mean=mean, std=std)
    samples = sample(leaf, num_samples=500)
    assert torch.isclose(samples.mean(), torch.tensor(mean), atol=1e-1)
    assert torch.isclose(samples.std(), torch.tensor(std), atol=1e-1)


@pytest.mark.parametrize("bias_correction", [True, False])
def test_maximum_likelihood_estimation(bias_correction):
    mean = 0.7
    std = 0.3
    leaf = make_leaf(mean=mean, std=std)
    data = make_data()
    maximum_likelihood_estimation(leaf, data, bias_correction=bias_correction)
    assert torch.isclose(leaf.mean, data.mean(), atol=1e-2)
    assert torch.isclose(leaf.std, data.std(correction=bias_correction), atol=1e-2)


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
    assert leaf.std.requires_grad


if __name__ == "__main__":
    unittest.main()
