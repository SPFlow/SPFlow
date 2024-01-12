import unittest
import numpy as np
import scipy
from spflow.meta.data import Scope
from spflow import log_likelihood, sample, maximum_likelihood_estimation
from spflow.modules.node.leaf.categorical import Categorical
from pytest import raises
import torch
from tests.fixtures import set_seed
from tests.modules.node.leaf.utils import evaluate_log_likelihood


def make_leaf(probs=[0.1, 0.2, 0.7]):
    return Categorical(scope=Scope([0]), probs=probs)


def make_data(probs=[0.1, 0.2, 0.7], n_samples=5):
    return torch.distributions.Categorical(probs=torch.tensor(probs)).sample((n_samples, 1))


def test_sample():
    probs = [0.3, 0.7]
    leaf = make_leaf(probs=probs)
    samples = sample(leaf, num_samples=10000)
    p_0 = (samples == 0).sum() / samples.shape[0]
    p_1 = (samples == 1).sum() / samples.shape[0]
    assert np.isclose(p_0, probs[0], atol=1e-1)
    assert np.isclose(p_1, probs[1], atol=1e-1)


def test_log_likelihood():
    evaluate_log_likelihood(make_leaf(), make_data())


def test_maximum_likelihood_estimation():
    leaf = make_leaf(probs=[0.1, 0.2, 0.7])
    data = make_data(probs=[0.1, 0.3, 0.6], n_samples=10000)
    maximum_likelihood_estimation(leaf, data)
    assert np.isclose(leaf.probs[0].item(), 0.1, atol=1e-2)
    assert np.isclose(leaf.probs[1].item(), 0.3, atol=1e-2)
    assert np.isclose(leaf.probs[2].item(), 0.6, atol=1e-2)


def test_constructor():
    # Check that parameters are set correcT.
    probs = [0.1, 0.2, 0.7]
    leaf = make_leaf(probs=probs)
    assert np.isclose(leaf.probs.detach(), probs).all()

    # Check invalid parameters
    with raises(ValueError):
        leaf = make_leaf(probs=[0.1, 0.2, 0.7, 0.1])
        leaf = make_leaf(probs=[0.1, 0.2, 0.7, -0.1])
        leaf = make_leaf(probs=[0.1, 0.2, 0.7, 1.1])


def test_requires_grad():
    leaf = make_leaf()
    assert leaf.probs.requires_grad


if __name__ == "__main__":
    unittest.main()
