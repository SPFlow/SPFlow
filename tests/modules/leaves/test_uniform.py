from itertools import product

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.leaves import Uniform

out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_params(out_features: int, out_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create parameters for a uniform distribution.

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
    """Create a Uniform leaves node.

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
