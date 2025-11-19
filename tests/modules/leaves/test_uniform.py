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
        low: Lower bound of the uniform distribution.
        high: Upper bound of the uniform distribution.
    """
    low = torch.rand(out_features, out_channels)
    high = low + torch.rand(out_features, out_channels)
    return low, high


def make_leaf(low, high) -> Uniform:
    """Create a Uniform leaves node.

    Args:
        low: Lower bound of the uniform distribution.
        high: Upper bound of the uniform distribution.
    """
    scope = Scope(list(range(low.shape[0])))
    return Uniform(scope=scope, low=low, high=high)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_uniform_constructor_missing_low(out_features: int, out_channels: int):
    """Test the constructor of a Uniform distribution with missing low bound."""
    low, high = make_params(out_features, out_channels)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        Uniform(scope=scope, low=None, high=high)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_uniform_constructor_missing_high(out_features: int, out_channels: int):
    """Test the constructor of a Uniform distribution with missing high bound."""
    low, high = make_params(out_features, out_channels)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        Uniform(scope=scope, low=low, high=None)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_uniform_constructor_equal_bounds(out_features: int, out_channels: int):
    """Test the constructor of a Uniform distribution with identical bounds."""
    low, _ = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(low=low, high=low).distribution


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_uniform_constructor_low_greater_than_high(out_features: int, out_channels: int):
    """Test the constructor of a Uniform distribution with low greater than high."""
    low, high = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(low=high, high=low).distribution
