import unittest
from itertools import product
from spflow import InvalidParameterCombinationError
from tests.fixtures import auto_set_test_seed, auto_set_test_device
import pytest
import torch
from spflow.meta import Scope
from spflow.modules.leaf import Uniform

out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_params(out_features: int, out_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create parameters for a uniform distribution.

    Args:
        out_features: Number of features.
        out_channels: Number of channels.

    Returns:
        start: Lower bound of the uniform distribution.
        end: Upper bound of the uniform distribution.
    """
    start = torch.rand(out_features, out_channels)
    end = start + torch.rand(out_features, out_channels)
    return start, end


def make_leaf(start, end) -> Uniform:
    """
    Create a Uniform leaf node.
    Args:
        start: Lower bound of the uniform distribution.
        end: Upper bound of the uniform distribution.
    """
    scope = Scope(list(range(start.shape[0])))
    return Uniform(scope=scope, start=start, end=end)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_uniform_constructor_missing_start(out_features: int, out_channels: int):
    """Test the constructor of a Uniform distribution with missing start."""
    start, end = make_params(out_features, out_channels)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        Uniform(scope=scope, start=None, end=end)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_uniform_constructor_missing_end(out_features: int, out_channels: int):
    """Test the constructor of a Uniform distribution with missing end."""
    start, end = make_params(out_features, out_channels)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        Uniform(scope=scope, start=start, end=None)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_uniform_constructor_start_equals_end(out_features: int, out_channels: int):
    """Test the constructor of a Uniform distribution with start equal to end."""
    start, _ = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(start=start, end=start)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_uniform_constructor_start_greater_than_end(out_features: int, out_channels: int):
    """Test the constructor of a Uniform distribution with start greater than end."""
    start, end = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(start=end, end=start)


def test_uniform_support_half_open_interval():
    start = torch.tensor([[0.0]])
    end = torch.tensor([[1.0]])
    leaf = Uniform(scope=Scope([0]), start=start, end=end, support_outside=False)

    inside = torch.tensor([[[0.5]]])
    boundary = torch.tensor([[[1.0]]])
    below = torch.tensor([[[-0.1]]])

    assert torch.all(leaf.check_support(inside))
    assert not torch.all(leaf.check_support(boundary))
    assert not torch.all(leaf.check_support(below))


def test_uniform_support_outside_flag_allows_values():
    start = torch.tensor([[0.0]])
    end = torch.tensor([[1.0]])
    leaf = Uniform(scope=Scope([0]), start=start, end=end, support_outside=True)

    outside = torch.tensor([[[2.0]]])
    assert torch.all(leaf.check_support(outside))
