import unittest
import numpy as np
import scipy
from spflow.meta.data import Scope
from spflow import sample, maximum_likelihood_estimation
from spflow.modules.node.leaf import Bernoulli
from pytest import raises
import torch
from tests.fixtures import set_seed


def make_leaf(p=0.5):
    return Bernoulli(scope=Scope([0]), p=p)


def make_data():
    return torch.tensor([0, 1, 0, 1, 1, 0]).reshape((-1, 1))


def test_sample():
    p = 0.7
    leaf = make_leaf(p=p)
    samples = sample(leaf, num_samples=500)
    assert np.isclose(samples.mean(), p, atol=1e-1)


def test_maximum_likelihood_estimation():
    leaf = make_leaf(p=0.5)
    data = torch.reshape(torch.tensor([0, 1, 1, 1, 0, 0, 0, 1]), (-1, 1))
    maximum_likelihood_estimation(leaf, data)
    assert np.isclose(leaf.p.item(), torch.sum(data) / (data.shape[0]), atol=1e-2)


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
