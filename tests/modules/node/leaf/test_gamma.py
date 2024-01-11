import unittest

import numpy as np
import pytest
import scipy
import torch
from pytest import raises

from spflow import maximum_likelihood_estimation, sample
from spflow.meta.data import Scope
from spflow.modules.node.leaf.gamma import Gamma

from tests.fixtures import set_seed


def make_leaf(alpha=1.0, beta=1.0):
    return Gamma(scope=Scope([0]), alpha=alpha, beta=beta)


def make_data(alpha=1.0, beta=1.0, n_samples=5):
    return torch.distributions.Gamma(alpha, beta).sample((n_samples, 1))


def test_sample():
    leaf = make_leaf(alpha=2.0, beta=3.0)
    samples = sample(leaf, num_samples=500)

    # TODO: what can we check here?


@pytest.mark.parametrize("bias_correction", [True, False])
def test_maximum_likelihood_estimation(bias_correction):
    alpha, beta = 0.8, 1.2
    leaf = make_leaf()
    data = make_data(alpha=alpha, beta=beta, n_samples=10000)
    maximum_likelihood_estimation(leaf, data, bias_correction=bias_correction)
    assert np.isclose(leaf.alpha.detach(), alpha, atol=1e-1)
    assert np.isclose(leaf.beta.detach(), beta, atol=1e-1)


def test_constructor():
    # Check that parameters are set correctly
    alpha, beta = 2.0, 3.0
    leaf = make_leaf(alpha=alpha, beta=beta)
    assert leaf.alpha == alpha
    assert leaf.beta == beta

    # Check invalid parameters
    with raises(ValueError):
        leaf = make_leaf(alpha=-0.5)
        leaf = make_leaf(alpha=0.0)
        leaf = make_leaf(beta=-0.5)
        leaf = make_leaf(beta=0.0)


def test_requires_grad():
    leaf = make_leaf()
    assert leaf.alpha.requires_grad
    assert leaf.beta.requires_grad


if __name__ == "__main__":
    unittest.main()
