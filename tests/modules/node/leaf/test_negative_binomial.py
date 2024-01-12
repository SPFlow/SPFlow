import unittest
from tests.modules.node.leaf.utils import evaluate_log_likelihood
import numpy as np
import scipy
from spflow.meta.data import Scope
from spflow import sample, maximum_likelihood_estimation
from spflow.modules.node.leaf.negative_binomial import NegativeBinomial
from pytest import raises
import torch
from tests.fixtures import set_seed


def make_leaf(n=2, p=0.5):
    return NegativeBinomial(scope=Scope([0]), n=n, p=p)


def make_data(n=2, p=0.5, n_samples=5):
    return torch.distributions.NegativeBinomial(n, p).sample((n_samples, 1))


def test_log_likelihood():
    evaluate_log_likelihood(make_leaf(), make_data())


def test_sample():
    p = 0.7
    leaf = make_leaf(p=p)
    samples = sample(leaf, num_samples=500)


def test_maximum_likelihood_estimation():
    n, p = 2, 0.3
    leaf = make_leaf(n=n, p=0.5)
    data = make_data(n=n, p=p, n_samples=5000)
    maximum_likelihood_estimation(leaf, data)
    assert np.isclose(leaf.p.item(), p, atol=1e-2)


def test_constructor():
    # Check that parameters are set correcT.
    n = 4
    p = 0.5
    leaf = make_leaf(n=n, p=p)
    assert leaf.n == n
    assert leaf.p == p

    # Check invalid parameters
    with raises(ValueError):
        leaf = make_leaf(n=-1)
        leaf = make_leaf(p=-0.5)
        leaf = make_leaf(p=1.5)


def test_requires_grad():
    leaf = make_leaf()
    assert not leaf.n.requires_grad
    assert leaf.p.requires_grad


if __name__ == "__main__":
    unittest.main()
