import unittest
from tests.modules.node.leaf.utils import evaluate_log_likelihood
import numpy as np
import scipy
from spflow.meta.data import Scope
from spflow import sample, maximum_likelihood_estimation
from spflow.modules.node.leaf.binomial import Binomial
from pytest import raises
import torch
from tests.fixtures import set_seed


def make_leaf(n=2, p=0.5):
    return Binomial(scope=Scope([0]), n=n, p=p)


def make_data():
    return torch.tensor([0, 1, 2]).reshape((-1, 1))


def test_log_likelihood():
    evaluate_log_likelihood(make_leaf(), make_data())


def test_sample():
    p = 0.7
    leaf = make_leaf(p=p)
    samples = sample(leaf, num_samples=500)
    assert np.isclose(samples.mean() / leaf.n, p, atol=1e-1)


def test_maximum_likelihood_estimation():
    leaf = make_leaf(n=4, p=0.5)
    data = torch.tensor([0, 1, 2, 2, 3, 3, 3, 4]).view(-1, 1)
    maximum_likelihood_estimation(leaf, data)
    assert np.isclose(leaf.p.item(), data.sum() / (leaf.n * data.shape[0]), atol=1e-2)


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
