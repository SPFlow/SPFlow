import unittest
from itertools import product

from typing import Union
from spflow.meta.dispatch import init_default_sampling_context
from tests.modules.node.leaf.utils import evaluate_log_likelihood, evaluate_samples
import pytest
import torch
from pytest import raises

from spflow import maximum_likelihood_estimation, sample, marginalize
from spflow.meta.data import Scope
from spflow.modules.layer.leaf.exponential import Exponential as ExponentialLayer
from spflow.modules.node.leaf.exponential import Exponential as ExponentialNode
from spflow.modules.layer.vectorized.leaf.exponential import Exponential as VectorizedExponential


# Constants
NUM_LEAVES = 2
NUM_SCOPES = 3
TOTAL_SCOPES = 5


def make_params(module_type: str, rate=None) -> torch.Tensor:
    """
    Create parameters for a exponential distribution.

    If rate is not provided, it is randomly initialized, according to the module type.

    Args:
        module_type: Type of the module, can be "node" or "layer".
        rate: PyTorch tensor representing the rate parameters of the Exponential distributions
    """
    if module_type == "node":
        if rate is not None:
            assert rate.shape == (1,)
            return rate
        else:
            return torch.rand(1) + 1e-8
    elif module_type == "layer":
        if rate is not None:
            assert rate.shape == (NUM_SCOPES, NUM_LEAVES)
            return rate
        else:
            return torch.rand(NUM_SCOPES, NUM_LEAVES) + 1e-8
    elif module_type == "vector":
        if rate is not None:
            assert rate.shape == (NUM_SCOPES, 1)
            return rate
        else:
            return torch.rand(NUM_SCOPES, 1) + 1e-8
    else:
        raise ValueError(f"Invalid module_type: {module_type}")


def make_leaf(module_type: str, rate=None) -> Union[ExponentialNode, ExponentialLayer]:
    """
    Create a Normal leaf node.

    Args:
        module_type: Type of the module.
        rate: PyTorch tensor representing the rate parameters of the Exponential distributions
    """
    if module_type == "node":
        rate = rate if rate is not None else torch.rand(1) + 1e-8
        scope = Scope([1])
        return ExponentialNode(scope=scope, rate=rate)
    elif module_type == "layer":
        rate = rate if rate is not None else torch.rand(NUM_SCOPES, NUM_LEAVES) + 1e-8
        scope = Scope(list(range(1, NUM_SCOPES + 1)))
        return ExponentialLayer(scope=scope, rate=rate)
    elif module_type == "vector":
        rate = rate if rate is not None else torch.rand(NUM_SCOPES, 1) + 1e-8
        scope = Scope(list(range(1, NUM_SCOPES + 1)))
        return VectorizedExponential(scope=scope, rate=rate)
    else:
        raise ValueError(f"Invalid module_type: {module_type}")


def make_data(rate=None, n_samples=5) -> torch.Tensor:
    """
    Generate data from a exponential distribution.

    Args:
        module_type: Type of the module.
        rate: PyTorch tensor representing the rate parameters of the Exponential distributions
        n_samples: Number of samples to generate.
    """
    rate = rate if rate is not None else torch.rand(TOTAL_SCOPES) + 1e-8

    return torch.distributions.Exponential(rate=rate).sample((n_samples,))


@pytest.mark.parametrize("module_type", ["node", "vector", "layer"])
def test_log_likelihood(module_type: str):
    """Test the log likelihood of a exponential distribution."""
    evaluate_log_likelihood(make_leaf(module_type), make_data())


@pytest.mark.parametrize("module_type,is_mpe", product(["node", "vector", "layer"], [True, False]))
def test_sample(module_type: str, is_mpe: bool):
    """Test sampling from a exponential distribution."""
    rate = make_params(module_type)
    leaf = make_leaf(module_type, rate=rate)

    n_samples = 5000
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, n=n_samples)

    for i in range(NUM_LEAVES):
        data = torch.full((n_samples, TOTAL_SCOPES), torch.nan)
        sampling_ctx.output_ids = torch.full((n_samples, NUM_SCOPES), fill_value=i)
        evaluate_samples(leaf, data, is_mpe=is_mpe, sampling_ctx=sampling_ctx)


@pytest.mark.parametrize("bias_correction, module_type", product([True, False], ["node", "vector", "layer"])) # product([True, False], ["node", "layer"])
def test_maximum_likelihood_estimation(bias_correction: bool, module_type: str):
    """Test maximum likelihood estimation of a exponential distribution.

    Args:
        bias_correction: Whether to use bias correction in the estimation.
    """
    leaf = make_leaf(module_type)
    rate = torch.rand(TOTAL_SCOPES)
    data = make_data(rate, n_samples=1000)
    maximum_likelihood_estimation(leaf, data, bias_correction=bias_correction)
    assert torch.isclose(rate[leaf.scope.query].unsqueeze(1), leaf.distribution.rate, atol=1e-1).all()


@pytest.mark.parametrize("module_type", ["node", "vector", "layer"])
def test_constructor(module_type: str):
    """Test the constructor of a exponential distribution."""
    # Check that parameters are set correctly
    rate = make_params(module_type)
    leaf = make_leaf(module_type=module_type, rate=rate)
    assert torch.isclose(leaf.distribution.rate, rate).all()

    # Check invalid parameters
    with raises(ValueError):
        make_leaf(module_type=module_type, rate=-1.0 * rate)  # negative rate
        make_leaf(module_type=module_type, rate=0.0 * rate)  # zero rate
        make_leaf(module_type=module_type, rate=torch.full(rate.shape, torch.nan))  # nan rate
        make_leaf(module_type=module_type, rate=rate.unsqueeze(0))  # wrong rate shape
        make_leaf(module_type=module_type, rate=None)  # missing std


@pytest.mark.parametrize("module_type", ["node", "vector", "layer"])
def test_requires_grad(module_type: str):
    """Test whether the rate of a exponential distribution require gradients."""
    leaf = make_leaf(module_type)
    assert leaf.distribution.rate.requires_grad


@pytest.mark.parametrize("module_type", ["node", "vector", "layer"])
def test_marginalize(module_type: str):
    """Test marginalization of a exponential distribution."""
    leaf = make_leaf(module_type)
    scope_og = leaf.scope.copy()
    marg_rvs = [1, 2]
    leaf_marg = marginalize(leaf, marg_rvs)
    num_leaves = NUM_LEAVES if module_type == "layer" else 1

    if module_type == "node":
        assert leaf_marg == None
    else:
        assert leaf_marg.distribution.rate.shape == (NUM_SCOPES - len(marg_rvs), num_leaves)

        # TODO: ensure, that the correct scopes were marginalized
        assert leaf_marg.scope.query == [q for q in scope_og.query if q not in marg_rvs]


if __name__ == "__main__":
    unittest.main()