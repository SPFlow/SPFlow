import unittest

from tests.modules.node.leaf.utils import evaluate_log_likelihood
import numpy as np
import pytest
import scipy
import torch
from pytest import raises

from spflow import maximum_likelihood_estimation, sample
from spflow.meta.data import Scope
from spflow.modules.node.leaf.geometric import Geometric
from tests.fixtures import set_seed


def make_leaf(p=0.5):
    return Geometric(scope=Scope([0]), p=p)


def make_data(p=0.5, n_samples=5):
    return torch.distributions.Geometric(probs=torch.tensor(p)).sample((n_samples, 1))


def test_sample():
    p = 0.7
    leaf = make_leaf(p=p)
    samples = sample(leaf, num_samples=100)
    assert torch.all(samples >= 0)


@pytest.mark.parametrize("bias_correction", [True, False])
def test_maximum_likelihood_estimation(bias_correction):
    leaf = make_leaf(p=0.5)
    p = 0.3
    data = make_data(p=0.3, n_samples=10000)
    maximum_likelihood_estimation(leaf, data, bias_correction=bias_correction)
    assert np.isclose(leaf.p.item(), p, atol=1e-2)


def test_log_likelihood():
    evaluate_log_likelihood(make_leaf(), make_data())


def test_constructor():
    # Check that parameters are set correct.
    p = 0.5
    leaf = make_leaf(p=p)
    assert leaf.p == p

    # Check invalid parameters
    with raises(ValueError):
        leaf = make_leaf(p=-0.5)
        leaf = make_leaf(p=1.5)


def test_requires_grad():
    leaf = make_leaf()
    assert leaf.p.requires_grad


if __name__ == "__main__":
    unittest.main()
