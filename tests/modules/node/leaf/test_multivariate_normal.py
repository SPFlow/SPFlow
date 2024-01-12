#!/usr/bin/env python3

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
from spflow.modules.node.leaf.multivariate_normal import MultivariateNormal


def make_leaf(mean=None, cov=None):
    if mean is None:
        mean = torch.randn(5)
        ndim = 5
    else:
        ndim = mean.shape[0]
    if cov is None:
        cov = torch.cov(torch.randn(ndim, 100))
    return MultivariateNormal(scope=Scope(list(range(ndim))), mean=mean, cov=cov)


def make_data(mean=None, cov=None, n_samples=5):
    if mean is None:
        mean = torch.randn(5)
        ndim = 5
    else:
        ndim = mean.shape[0]
    if cov is None:
        cov = torch.cov(torch.randn(5, 100))
    return torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov).sample((n_samples,))


def test_log_likelihood():
    evaluate_log_likelihood(make_leaf(), make_data())


def test_sample():
    leaf = make_leaf()
    samples = sample(leaf, num_samples=500)
    # TODO: what should we test?


@pytest.mark.parametrize("bias_correction", [True, False])
def test_maximum_likelihood_estimation(bias_correction):
    mean = torch.randn(5)
    cov = torch.cov(torch.randn(5, 100))
    leaf = make_leaf(mean=mean, cov=cov)
    data = make_data(mean=mean, cov=cov, n_samples=10000)
    maximum_likelihood_estimation(leaf, data, bias_correction=bias_correction)
    assert torch.isclose(leaf.mean, data.mean(0), atol=1e-2).all()
    assert torch.isclose(leaf.cov, data.T.cov(correction=int(bias_correction)), atol=1e-2).all()


def test_constructor():
    # Check that parameters are set correctly
    mean = torch.randn(5)
    cov = torch.cov(torch.randn(5, 100))
    leaf = make_leaf(mean=mean, cov=cov)
    assert torch.isclose(leaf.mean, mean, atol=1e-8).all()
    assert torch.isclose(leaf.cov, cov, atol=1e-8).all()

    # Check invalid parameters
    with raises(ValueError):
        leaf = make_leaf(cov=torch.randn(5, 5))
        leaf = make_leaf(mean=torch.rand(5), cov=torch.randn(6, 6))
        leaf = make_leaf(cov=torch.rand(5, 6))


def test_requires_grad():
    leaf = make_leaf()
    assert leaf.mean.requires_grad
    assert leaf.tril_diag_aux.requires_grad
    assert leaf.tril_nondiag.requires_grad


if __name__ == "__main__":
    unittest.main()
