import unittest
from itertools import product

from typing import Union
from spflow.meta.dispatch import init_default_sampling_context
from tests.modules.node.leaf.utils import evaluate_log_likelihood, evaluate_samples
from tests.fixtures import set_seed
import pytest
import torch
from pytest import raises

from spflow import maximum_likelihood_estimation, sample, marginalize
from spflow.meta.data import Scope
from spflow.modules.layer.leaf.binomial import Binomial as BinomialLayer
from spflow.modules.node.leaf.binomial import Binomial as BinomialNode
from spflow.modules.layer.vectorized.leaf.binomial import Binomial as VectorizedBinomial

# Constants
NUM_LEAVES = 2
NUM_SCOPES = 3
TOTAL_SCOPES = 5
TOTAL_TRIALS = 10


def make_params(module_type: str, n=None, p=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create parameters for a binomial distribution.

    If n and p are not provided, they are randomly initialized, according to the module type.

    Args:
        module_type: Type of the module, can be "node" or "layer".
        n: total trails.
        p: success probability of each trial.
    """
    if module_type == "node":
        if n is not None and p is not None:
            assert n.shape == (1,)
            assert p.shape == (1,)
            return n, p
        else:
            return torch.tensor(TOTAL_TRIALS), torch.rand(1) + 1e-8
    elif module_type == "layer":
        if n is not None and p is not None:
            assert p.shape == (NUM_SCOPES, NUM_LEAVES)
            return n, p
        else:
            return torch.tensor(TOTAL_TRIALS), torch.rand(NUM_SCOPES, NUM_LEAVES) + 1e-8
    elif module_type == "vector":
        if n is not None and p is not None:
            assert p.shape == (NUM_SCOPES, 1)
            return n, p
        else:
            return torch.tensor(TOTAL_TRIALS), torch.rand(NUM_SCOPES, 1) + 1e-8
    else:
        raise ValueError(f"Invalid module_type: {module_type}")


def make_leaf(module_type: str, n=None, p=None) -> Union[BinomialNode, BinomialLayer]:
    """
    Create a Binomial leaf node.

    Args:
        module_type: Type of the module.
        n: total trails.
        p: success probability of each trial.
    """
    if module_type == "node":
        n = n if n is not None else torch.tensor(TOTAL_TRIALS)
        p = p if p is not None else torch.rand(1) + 1e-8
        scope = Scope([1])
        return BinomialNode(scope=scope, n=n, p=p)
    elif module_type == "layer":
        n = n if n is not None else torch.tensor(TOTAL_TRIALS)
        p = p if p is not None else torch.rand(NUM_SCOPES, NUM_LEAVES) + 1e-8
        scope = Scope(list(range(1, NUM_SCOPES + 1)))
        return BinomialLayer(scope=scope, n=n, p=p)
    elif module_type == "vector":
        n = n if n is not None else torch.tensor(TOTAL_TRIALS)
        p = p if p is not None else torch.rand(NUM_SCOPES, 1) + 1e-8
        scope = Scope(list(range(1, NUM_SCOPES + 1)))
        return VectorizedBinomial(scope=scope, n=n, p=p)
    else:
        raise ValueError(f"Invalid module_type: {module_type}")


def make_data(n=None, p=None, n_samples=5) -> torch.Tensor:
    """
    Generate data from a binomial distribution.

    Args:
        module_type: Type of the module.
        n: total trails.
        p: success probability of each trial.
        n_samples: Number of samples to generate.
    """
    n = n if n is not None else torch.tensor(TOTAL_TRIALS)
    p = p if p is not None else torch.rand(TOTAL_SCOPES) + 1e-8

    return torch.distributions.Binomial(total_count=n, probs=p).sample((n_samples,))


@pytest.mark.parametrize("module_type", ["node", "vector", "layer"])
def test_log_likelihood(module_type: str):
    """Test the log likelihood of a binomial distribution."""
    evaluate_log_likelihood(make_leaf(module_type), make_data())


@pytest.mark.parametrize("module_type,is_mpe", product(["node", "vector", "layer"], [True, False]))
def test_sample(module_type: str, is_mpe: bool):
    """Test sampling from a binomial distribution."""
    n, p = make_params(module_type)
    leaf = make_leaf(module_type, n=n, p=p)

    n_samples = 5000
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, n=n_samples)

    for i in range(NUM_LEAVES):
        data = torch.full((n_samples, TOTAL_SCOPES), torch.nan)
        sampling_ctx.output_ids = torch.full((n_samples, NUM_SCOPES), fill_value=i)
        evaluate_samples(leaf, data, is_mpe=is_mpe, sampling_ctx=sampling_ctx)


@pytest.mark.parametrize("bias_correction, module_type", product([True, False], ["node", "vector", "layer"])) #product([True, False], ["node", "layer"])
def test_maximum_likelihood_estimation(bias_correction: bool, module_type: str):
    """Test maximum likelihood estimation of a binomial distribution.

    Args:
        bias_correction: Whether to use bias correction in the estimation.
    """
    leaf = make_leaf(module_type)
    p = torch.rand(TOTAL_SCOPES)
    data = make_data(n=TOTAL_TRIALS, p=p, n_samples=1000)
    maximum_likelihood_estimation(leaf, data, bias_correction=bias_correction)
    assert torch.isclose(p[leaf.scope.query].unsqueeze(1), leaf.distribution.p, atol=1e-2).all()



@pytest.mark.parametrize("module_type", ["node", "vector", "layer"])
def test_constructor(module_type: str):
    """Test the constructor of a binomial distribution."""
    # Check that parameters are set correctly
    n, p = make_params(module_type)
    leaf = make_leaf(module_type=module_type, n=n, p=p)
    assert torch.isclose(leaf.distribution.p, p).all()
    assert torch.isclose(leaf.distribution.n, n).all()

    # Check invalid parameters
    with raises(ValueError):
        make_leaf(module_type=module_type, n=n, p=-1.0 * p)  # negative p
        make_leaf(module_type=module_type, n=n, p=0.0 * p)  # zero p
        make_leaf(module_type=module_type, n=torch.nan, p=p)  # nan n
        make_leaf(module_type=module_type, n=n, p=torch.full(p.shape, torch.nan))  # nan p
        make_leaf(module_type=module_type, n=n, p=p.unsqueeze(0))  # wrong p shape
        make_leaf(module_type=module_type, n=n.unsqueeze(0), p=p)  # wrong n shape
        make_leaf(module_type=module_type, n=None, p=p)  # missing n
        make_leaf(module_type=module_type, n=n, p=None)  # missing p


@pytest.mark.parametrize("module_type", ["node", "vector", "layer"])
def test_requires_grad(module_type: str):
    """Test whether the mean and std of a binomial distribution require gradients."""
    leaf = make_leaf(module_type)
    assert leaf.distribution.p.requires_grad



@pytest.mark.parametrize("module_type", ["node", "vector", "layer"])
def test_marginalize(module_type: str):
    """Test marginalization of a binomial distribution."""
    leaf = make_leaf(module_type)
    scope_og = leaf.scope.copy()
    marg_rvs = [1, 2]
    leaf_marg = marginalize(leaf, marg_rvs)
    num_leaves = NUM_LEAVES if module_type == "layer" else 1

    if module_type == "node":
        assert leaf_marg == None
    else:
        assert leaf_marg.distribution.p.shape == (NUM_SCOPES - len(marg_rvs), num_leaves)
        assert leaf_marg.distribution.n.shape == (NUM_SCOPES - len(marg_rvs), num_leaves)

        # TODO: ensure, that the correct scopes were marginalized
        assert leaf_marg.scope.query == [q for q in scope_og.query if q not in marg_rvs]


if __name__ == "__main__":
    unittest.main()