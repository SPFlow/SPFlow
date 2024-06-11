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
from spflow.modules.layer.leaf.log_normal import LogNormal as LogNormalLayer
from spflow.modules.node.leaf.log_normal import LogNormal as LogNormalNode
from spflow.modules.layer.vectorized.leaf.log_normal import LogNormal as VectorizedLogNormalLayer

# Constants
NUM_LEAVES = 2
NUM_SCOPES = 3
TOTAL_SCOPES = 5


def make_params(module_type: str, mean=None, std=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create parameters for a LogNormal distribution.

    If mean and std are not provided, they are randomly initialized, according to the module type.

    Args:
        module_type: Type of the module, can be "node" or "layer".
        mean: Mean of the distribution.
        std: Standard deviation of the distribution.
    """
    if module_type == "node":
        if mean is not None and std is not None:
            assert mean.shape == (1,)
            assert std.shape == (1,)
            return mean, std
        else:
            return torch.randn(1), torch.rand(1) + 1e-8
    elif module_type == "vector":
        if mean is not None and std is not None:
            assert mean.shape == (NUM_SCOPES, 1)
            assert std.shape == (NUM_SCOPES, 1)
            return mean, std
        else:
            return torch.randn(NUM_SCOPES, 1), torch.rand(NUM_SCOPES, 1) + 1e-8
    elif module_type == "layer":
        if mean is not None and std is not None:
            assert mean.shape == (NUM_SCOPES, NUM_LEAVES)
            assert std.shape == (NUM_SCOPES, NUM_LEAVES)
            return mean, std
        else:
            return torch.randn(NUM_SCOPES, NUM_LEAVES), torch.rand(NUM_SCOPES, NUM_LEAVES) + 1e-8
    else:
        raise ValueError(f"Invalid module_type: {module_type}")


def make_leaf(module_type: str, mean=None, std=None) -> Union[LogNormalNode, LogNormalLayer]:
    """
    Create a LogNormal leaf node.

    Args:
        module_type: Type of the module.
        mean: Mean of the distribution.
        std: Standard deviation of the distribution.
    """
    if module_type == "node":
        mean = mean if mean is not None else torch.randn(1)
        std = std if std is not None else torch.rand(1) + 1e-8
        scope = Scope([1])
        return LogNormalNode(scope=scope, mean=mean, std=std)
    elif module_type == "vector":
        mean = mean if mean is not None else torch.randn(NUM_SCOPES, 1)
        std = std if std is not None else torch.rand(NUM_SCOPES, 1) + 1e-8
        scope = Scope(list(range(1, NUM_SCOPES + 1)))
        return VectorizedLogNormalLayer(scope=scope, mean=mean, std=std)
    elif module_type == "layer":
        mean = mean if mean is not None else torch.randn(NUM_SCOPES, NUM_LEAVES)
        std = std if std is not None else torch.rand(NUM_SCOPES, NUM_LEAVES) + 1e-8
        scope = Scope(list(range(1, NUM_SCOPES + 1)))
        return LogNormalLayer(scope=scope, mean=mean, std=std)
    else:
        raise ValueError(f"Invalid module_type: {module_type}")


def make_data(mean=None, std=None, n_samples=5) -> torch.Tensor:
    """
    Generate data from a LogNormal distribution.

    Args:
        module_type: Type of the module.
        mean: Mean of the distribution.
        std: Standard deviation of the distribution.
        n_samples: Number of samples to generate.
    """
    mean = mean if mean is not None else torch.randn(TOTAL_SCOPES)
    std = std if std is not None else torch.rand(TOTAL_SCOPES) + 1e-8

    return torch.distributions.LogNormal(loc=mean, scale=std).sample((n_samples,))


@pytest.mark.parametrize("module_type", ["node", "vector", "layer"])
def test_log_likelihood(module_type: str):
    """Test the log likelihood of a LogNormal distribution."""
    evaluate_log_likelihood(make_leaf(module_type), make_data())


@pytest.mark.parametrize("module_type,is_mpe", product(["node", "vector", "layer"], [True, False]))
def test_sample(module_type: str, is_mpe: bool):
    """Test sampling from a LogNormal distribution."""
    mean, std = make_params(module_type)
    leaf = make_leaf(module_type, mean=mean, std=std)

    n_samples = 5000
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, n=n_samples)

    for i in range(NUM_LEAVES):
        data = torch.full((n_samples, TOTAL_SCOPES), torch.nan)
        sampling_ctx.output_ids = torch.full((n_samples, NUM_SCOPES), fill_value=i)
        evaluate_samples(leaf, data, is_mpe=is_mpe, sampling_ctx=sampling_ctx)


@pytest.mark.parametrize("bias_correction, module_type", product([True, False], ["node", "vector", "layer"]))
def test_maximum_likelihood_estimation(bias_correction: bool, module_type: str):
    """Test maximum likelihood estimation of a log_normal distribution.

    Args:
        bias_correction: Whether to use bias correction in the estimation.
    """
    leaf = make_leaf(module_type)
    mean = torch.randn(TOTAL_SCOPES)
    std = torch.rand(TOTAL_SCOPES)
    data = make_data(mean, std, n_samples=1000)
    maximum_likelihood_estimation(leaf, data, bias_correction=bias_correction)
    assert torch.isclose(leaf.distribution.mean, mean[leaf.scope.query].unsqueeze(1), atol=1e-1).all()
    assert torch.isclose(leaf.distribution.std, std[leaf.scope.query].unsqueeze(1), atol=1e-1).all()


@pytest.mark.parametrize("module_type", ["node", "vector", "layer"])
def test_constructor(module_type: str):
    """Test the constructor of a log_normal distribution."""
    # Check that parameters are set correctly
    mean, std = make_params(module_type)
    leaf = make_leaf(module_type=module_type, mean=mean, std=std)
    assert torch.isclose(leaf.distribution.mean, mean).all()
    assert torch.isclose(leaf.distribution.std, std).all()

    # Check invalid parameters
    with raises(ValueError):
        make_leaf(module_type=module_type, mean=mean, std=-1.0 * std)  # negative std
        make_leaf(module_type=module_type, mean=std, std=0.0 * std)  # zero std
        make_leaf(module_type=module_type, mean=torch.full(mean.shape, torch.nan), std=std)  # nan mean
        make_leaf(module_type=module_type, mean=mean, std=torch.full(std.shape, torch.nan))  # nan std
        make_leaf(module_type=module_type, mean=mean, std=std.unsqueeze(0))  # wrong std shape
        make_leaf(module_type=module_type, mean=mean.unsqueeze(0), std=std)  # wrong mean shape
        make_leaf(module_type=module_type, mean=None, std=std)  # missing mean
        make_leaf(module_type=module_type, mean=mean, std=None)  # missing std


@pytest.mark.parametrize("module_type", ["node", "vector", "layer"])
def test_requires_grad(module_type: str):
    """Test whether the mean and std of a log_normal distribution require gradients."""
    leaf = make_leaf(module_type)
    assert leaf.distribution.mean.requires_grad
    assert leaf.distribution.std.requires_grad


@pytest.mark.parametrize("module_type", ["node", "vector", "layer"])
def test_marginalize(module_type: str):
    """Test marginalization of a log_normal distribution."""
    leaf = make_leaf(module_type)
    scope_og = leaf.scope.copy()
    marg_rvs = [1, 2]
    leaf_marg = marginalize(leaf, marg_rvs)
    num_leaves = NUM_LEAVES if module_type == "layer" else 1

    if module_type == "node":
        assert leaf_marg == None
    else:
        assert leaf_marg.distribution.mean.shape == (NUM_SCOPES - len(marg_rvs), num_leaves)
        assert leaf_marg.distribution.std.shape == (NUM_SCOPES - len(marg_rvs), num_leaves)

        # TODO: ensure, that the correct scopes were marginalized
        assert leaf_marg.scope.query == [q for q in scope_og.query if q not in marg_rvs]


if __name__ == "__main__":
    unittest.main()
