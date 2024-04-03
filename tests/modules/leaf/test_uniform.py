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
from spflow.modules.layer.leaf.uniform import Uniform as UniformLayer
from spflow.modules.node.leaf.uniform import Uniform as UniformNode

# Constants
NUM_LEAVES = 2
NUM_SCOPES = 3
TOTAL_SCOPES = 5
START_TENSOR = torch.tensor([0.0]).reshape(1,)
END_TENSOR = torch.tensor([5.0]).reshape(1,)


def make_params(module_type: str, start=None, end=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create parameters for a uniform distribution.

    If start and end are not provided, they are randomly initialized, according to the module type.

    Args:
        module_type: Type of the module, can be "node" or "layer".
        start:  PyTorch tensor containing the start of the intervals (including).
        end: PyTorch tensor containing the end of the intervals (including). Must be larger than 'start'.
    """
    if module_type == "node":
        if start is not None and end is not None:
            assert start.shape == (1,)
            assert end.shape == (1,)
            return start, end
        else:
            # Generate a float tensor of a random number between 0 and 5
            start = 5 * torch.rand(1)

            # Generate a float tensor of a random number between the first tensor and 10
            # Ensure that the second random number is greater than the first
            end_min = start + 1  # Ensure the second number is greater than the first
            end_max = 10 * torch.ones(1)  # Upper bound is 10
            end = end_min + (end_max - end_min) * torch.rand(1)
            return start, end
    elif module_type == "layer":
        if start is not None and end is not None:
            if start.shape != (NUM_SCOPES, NUM_LEAVES):
                start = start.repeat(NUM_SCOPES, NUM_LEAVES)
            if end.shape != (NUM_SCOPES, NUM_LEAVES):
                end = end.repeat(NUM_SCOPES, NUM_LEAVES)
            return start, end
        else:
            # Generate a float tensor of random numbers between 0 and 5
            start = 5 * torch.rand((NUM_SCOPES, NUM_LEAVES))

            # Generate a float tensor of random numbers between the first tensor and 10
            # Ensure that the second random number is greater than the first
            end_min = start + 1  # Ensure the second number is greater than the first
            end_max = 10 * torch.ones((NUM_SCOPES, NUM_LEAVES))  # Upper bound is 10
            end = end_min + (end_max - end_min) * torch.rand((NUM_SCOPES, NUM_LEAVES))
            return start, end
    else:
        raise ValueError(f"Invalid module_type: {module_type}")


def make_leaf(module_type: str, start=None, end=None) -> Union[UniformNode, UniformLayer]:
    """
    Create a Uniform leaf node.

    Args:
        module_type: Type of the module.
        start: PyTorch tensor containing the start of the intervals (including).
        end: PyTorch tensor containing the end of the intervals (including). Must be larger than 'start'.
    """
    if module_type == "node":
        start, end = make_params(module_type, start=start, end=end)
        scope = Scope([1])
        return UniformNode(scope=scope, start=start, end=end)
    elif module_type == "layer":
        start, end = make_params(module_type, start=start, end=end)
        scope = Scope(list(range(1, NUM_SCOPES + 1)))
        return UniformLayer(scope=scope, start=start, end=end)
    else:
        raise ValueError(f"Invalid module_type: {module_type}")


def make_data(start=None, end=None, n_samples=5) -> torch.Tensor:
    """
    Generate data from a uniform distribution.

    Args:
        module_type: Type of the module.
        start: PyTorch tensor containing the start of the intervals (including).
        end: PyTorch tensor containing the end of the intervals (including). Must be larger than 'start'.
        n_samples: Number of samples to generate.
    """
    if start and end is None:
        # Generate a float tensor of a random number between 0 and 5
        start = 5 * torch.rand(5)

        # Generate a float tensor of a random number between the first tensor and 10
        # Ensure that the second random number is greater than the first
        end_min = start + 1  # Ensure the second number is greater than the first
        end_max = 10 * torch.ones(TOTAL_SCOPES)  # Upper bound is 10
        end = end_min + (end_max - end_min) * torch.rand(1)
    else:
        start = start.repeat(TOTAL_SCOPES)
        end = end.repeat(TOTAL_SCOPES)

    return torch.distributions.Uniform(low=start, high=end).sample((n_samples,))


@pytest.mark.parametrize("module_type", ["node", "layer"])
def test_log_likelihood(module_type: str):
    """Test the log likelihood of a uniform distribution."""
    evaluate_log_likelihood(make_leaf(module_type, start=START_TENSOR, end=END_TENSOR), make_data(start= START_TENSOR, end=END_TENSOR))


@pytest.mark.parametrize("module_type,is_mpe", product(["node","layer"], [False]))
def test_sample(module_type: str, is_mpe: bool):
    """Test sampling from a uniform distribution."""

    # mpe is always False for uniform distribution as mode is not defined

    start, end = make_params(module_type)
    leaf = make_leaf(module_type, start=start, end=end)

    n_samples = 5000
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, n=n_samples)

    for i in range(NUM_LEAVES):
        data = torch.full((n_samples, TOTAL_SCOPES), torch.nan)
        sampling_ctx.output_ids = torch.full((n_samples, NUM_SCOPES), fill_value=i)
        evaluate_samples(leaf, data, is_mpe=is_mpe, sampling_ctx=sampling_ctx)


@pytest.mark.parametrize("bias_correction, module_type", product([True, False], ["node", "layer"]))
def test_maximum_likelihood_estimation(bias_correction: bool, module_type: str):
    """Test maximum likelihood estimation of a uniform distribution.

    Args:
        bias_correction: Whether to use bias correction in the estimation.
    """
    leaf = make_leaf(module_type)
    data = make_data(START_TENSOR, END_TENSOR, n_samples=1000)
    # test if mle runs without errors
    maximum_likelihood_estimation(leaf, data, bias_correction=bias_correction)



@pytest.mark.parametrize("module_type", ["node", "layer"])
def test_constructor(module_type: str):
    """Test the constructor of a uniform distribution."""
    # Check that parameters are set correctly
    start, end = make_params(module_type)
    leaf = make_leaf(module_type=module_type, start=start, end=end)
    assert torch.isclose(leaf.distribution.start, start).all()
    assert torch.isclose(leaf.distribution.end, end).all()

    # Check invalid parameters
    with raises(ValueError):

        make_leaf(module_type=module_type, start=torch.full(start.shape, torch.nan), end=end)  # nan start
        make_leaf(module_type=module_type, start=start, end=torch.full(end.shape, torch.nan))  # nan end
        make_leaf(module_type=module_type, start=start, end=end.unsqueeze(0))  # wrong end shape
        make_leaf(module_type=module_type, start=start.unsqueeze(0), end=end)  # wrong start shape
        make_leaf(module_type=module_type, start=None, end=end)  # missing mean
        make_leaf(module_type=module_type, start=start, end=None)  # missing std

@pytest.mark.parametrize("module_type", ["node", "layer"])
def test_marginalize(module_type: str):
    """Test marginalization of a uniform distribution."""
    leaf = make_leaf(module_type)
    scope_og = leaf.scope.copy()
    marg_rvs = [1, 2]
    leaf_marg = marginalize(leaf, marg_rvs)

    if module_type == "node":
        assert leaf_marg == None
    else:
        assert leaf_marg.distribution.start.shape == (NUM_SCOPES - len(marg_rvs), NUM_LEAVES)
        assert leaf_marg.distribution.end.shape == (NUM_SCOPES - len(marg_rvs), NUM_LEAVES)

        # TODO: ensure, that the correct scopes were marginalized
        assert leaf_marg.scope.query == [q for q in scope_og.query if q not in marg_rvs]


if __name__ == "__main__":
    unittest.main()
