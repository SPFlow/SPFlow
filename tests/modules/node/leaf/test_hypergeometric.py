import unittest
from spflow import log_likelihood

from tests.modules.node.leaf.utils import evaluate_log_likelihood
import numpy as np
import pytest
import scipy
import torch
from pytest import raises

from spflow import maximum_likelihood_estimation, sample
from spflow.meta.data import Scope
from spflow.modules.node.leaf.hypergeometric_old import Hypergeometric
from tests.fixtures import set_seed
import scipy


def make_leaf(N=10, M=5, n=3):
    return Hypergeometric(scope=Scope([0]), N=N, M=M, n=n)


def make_data(N=10, M=5, n=3, n_samples=5):
    return torch.from_numpy(np.random.hypergeometric(N, M, n, size=(n_samples, 1)))


def test_sample():
    leaf = make_leaf()
    samples = sample(leaf, num_samples=100)
    assert torch.all(samples >= 0)


@pytest.mark.parametrize("bias_correction", [True, False])
def test_maximum_likelihood_estimation(bias_correction):
    N, M, n = 10, 5, 3
    leaf = make_leaf()
    data = make_data()
    maximum_likelihood_estimation(leaf, data, bias_correction=bias_correction)

    # MLE should be a no-op for this distribution
    assert np.isclose(leaf.N.item(), N, atol=1e-2)
    assert np.isclose(leaf.M.item(), M, atol=1e-2)
    assert np.isclose(leaf.n.item(), n, atol=1e-2)


def test_log_likelihood():
    N, M, n = 10, 5, 3
    leaf = make_leaf(N=N, M=M, n=n)
    data = make_data(N=N, M=M, n=n)
    evaluate_log_likelihood(leaf, data)

    # Create scipy reference distribution
    scipy_dist = scipy.stats.hypergeom(N, M, n)

    # Check that the log likelihood is correct
    spflow_lls = log_likelihood(leaf, data)
    scipy_lls = scipy_dist.logpmf(data)
    assert np.allclose(spflow_lls, scipy_lls, atol=1e-5)


def test_constructor():
    # Check that parameters are set correct.
    N, M, n = 10, 5, 3
    leaf = make_leaf(N=N, M=M, n=n)
    assert leaf.N == N
    assert leaf.M == M
    assert leaf.n == n

    # Check invalid parameters
    with raises(ValueError):
        leaf = make_leaf(N=0.1)
        leaf = make_leaf(N=-1)
        leaf = make_leaf(M=0.1)
        leaf = make_leaf(M=-1)
        leaf = make_leaf(n=0.1)
        leaf = make_leaf(n=-1)
        leaf = make_leaf(M=5, N=3)  # M > N
        leaf = make_leaf(n=5, N=3)  # n > N


def test_requires_grad():
    leaf = make_leaf()
    assert not leaf.N.requires_grad
    assert not leaf.M.requires_grad
    assert not leaf.n.requires_grad


if __name__ == "__main__":
    unittest.main()
