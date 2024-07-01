from tests.fixtures import auto_set_test_seed
import unittest
from itertools import product

from typing import Union
from spflow.meta.dispatch import init_default_sampling_context
from tests.utils.leaves import evaluate_log_likelihood, evaluate_samples
import pytest
import torch
from pytest import raises

from spflow import maximum_likelihood_estimation, marginalize
from spflow.meta.data import Scope
from spflow.modules.leaf import NegativeBinomial as NegativeBinomial

# Constants
OUT_CHANNELS = 2
OUT_FEATURES = 3
TOTAL_SCOPES = 5
TOTAL_TRIALS = 10


def make_params(n=None, p=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create parameters for a NegativeBinomial distribution.

    If n and p are not provided, they are randomly initialized, according to the module type.

    Args:
        n: total trails.
        p: success probability of each trial.
    """
    if n is not None and p is not None:
        assert p.shape == (OUT_FEATURES, OUT_CHANNELS)
        return n, p
    else:
        return torch.tensor(TOTAL_TRIALS), torch.rand(OUT_FEATURES, OUT_CHANNELS) + 1e-8


def make_leaf(n=None, p=None) -> NegativeBinomial:
    """
    Create a NegativeBinomial leaf node.

    Args:
        n: total trails.
        p: success probability of each trial.
    """
    n = n if n is not None else torch.tensor(TOTAL_TRIALS)
    p = p if p is not None else torch.rand(OUT_FEATURES, OUT_CHANNELS) + 1e-8
    scope = Scope(list(range(1, OUT_FEATURES + 1)))
    return NegativeBinomial(scope=scope, n=n, p=p)


def make_data(n=None, p=None, n_samples=5) -> torch.Tensor:
    """
    Generate data from a negative_binomial distribution.

    Args:
        n: total trails.
        p: success probability of each trial.
        n_samples: Number of samples to generate.
    """
    n = n if n is not None else torch.tensor(TOTAL_TRIALS)
    p = p if p is not None else torch.rand(TOTAL_SCOPES) + 1e-8

    return torch.distributions.NegativeBinomial(total_count=n, probs=p).sample((n_samples,))


def test_log_likelihood():
    """Test the log likelihood of a NegativeBinomial distribution."""
    evaluate_log_likelihood(make_leaf(), make_data())


@pytest.mark.parametrize("is_mpe", [False, True])
def test_sample(is_mpe: bool):
    """Test sampling from a negative_binomial distribution."""
    n, p = make_params()
    leaf = make_leaf(n=n, p=p)

    n_samples = 10
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, n=n_samples)

    for i in range(OUT_CHANNELS):
        data = torch.full((n_samples, TOTAL_SCOPES), torch.nan)
        sampling_ctx.output_ids = torch.full((n_samples, OUT_FEATURES), fill_value=i)
        evaluate_samples(leaf, data, is_mpe=is_mpe, sampling_ctx=sampling_ctx)


@pytest.mark.parametrize("bias_correction", [True, False])
def test_maximum_likelihood_estimation(bias_correction: bool):
    """Test maximum likelihood estimation of a negative_binomial distribution.

    Args:
        bias_correction: Whether to use bias correction in the estimation.
    """
    leaf = make_leaf()
    p = torch.rand(TOTAL_SCOPES)
    data = make_data(n=TOTAL_TRIALS, p=p, n_samples=1000)
    maximum_likelihood_estimation(leaf, data, bias_correction=bias_correction)
    assert torch.isclose(p[leaf.scope.query].unsqueeze(1), leaf.distribution.p, atol=1e-1).all()


def test_constructor():
    """Test the constructor of a negative_binomial distribution."""
    # Check that parameters are set correctly
    n, p = make_params()
    leaf = make_leaf(n=n, p=p)
    assert torch.isclose(leaf.distribution.p, p).all()
    assert torch.isclose(leaf.distribution.n, n).all()

    # Check invalid parameters
    with raises(ValueError):
        make_leaf(n=n, p=-1.0 * p)  # negative p
        make_leaf(n=n, p=0.0 * p)  # zero p
        make_leaf(n=torch.nan, p=p)  # nan n
        make_leaf(n=n, p=torch.full(p.shape, torch.nan))  # nan p
        make_leaf(n=n, p=p.unsqueeze(0))  # wrong p shape
        make_leaf(n=n.unsqueeze(0), p=p)  # wrong n shape
        make_leaf(n=None, p=p)  # missing n
        make_leaf(n=n, p=None)  # missing p


def test_requires_grad():
    """Test whether the mean and std of a negative_binomial distribution require gradients."""
    leaf = make_leaf()
    assert leaf.distribution.p.requires_grad


def test_marginalize():
    """Test marginalization of a negative_binomial distribution."""
    leaf = make_leaf()
    scope_og = leaf.scope.copy()
    marg_rvs = [1, 2]
    leaf_marg = marginalize(leaf, marg_rvs)
    num_leaves = OUT_CHANNELS

    assert leaf_marg.distribution.p.shape == (OUT_FEATURES - len(marg_rvs), num_leaves)
    assert leaf_marg.distribution.n.shape == (OUT_FEATURES - len(marg_rvs), num_leaves)

    # TODO: ensure, that the correct scopes were marginalized
    assert leaf_marg.scope.query == [q for q in scope_og.query if q not in marg_rvs]


if __name__ == "__main__":
    unittest.main()
