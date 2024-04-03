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
from spflow.modules.node.leaf.exponential_old import Exponential


def make_leaf(rate=1.0):
    return Exponential(scope=Scope([0]), rate=rate)


def make_data(rate=1.0, n_samples=5):
    return torch.distributions.Exponential(rate=rate).sample((n_samples, 1))


def test_sample():
    leaf = make_leaf()
    samples = sample(leaf, num_samples=500)
    assert torch.all(samples >= 0)


def test_log_likelihood():
    evaluate_log_likelihood(make_leaf(), make_data())


@pytest.mark.parametrize("bias_correction", [True, False])
def test_maximum_likelihood_estimation(bias_correction):
    leaf = make_leaf()
    rate = 0.5
    data = make_data(rate=rate, n_samples=500)
    maximum_likelihood_estimation(leaf, data, bias_correction=bias_correction)
    assert np.isclose(leaf.rate.detach(), rate, atol=1e-1)


def test_constructor():
    # Check that parameters are set correctly
    rate = 17.0
    leaf = make_leaf(rate=rate)
    assert leaf.rate == rate

    # Check invalid parameters
    with raises(ValueError):
        leaf = make_leaf(rate=-0.1)
        leaf = make_leaf(rate=torch.nan)


def test_requires_grad():
    leaf = make_leaf()
    assert leaf.rate.requires_grad


if __name__ == "__main__":
    unittest.main()
