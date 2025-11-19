from itertools import product

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.leaves import Normal

out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_params(out_features: int, out_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create parameters for a normal distribution.

    Args:
        out_features: Number of features.
        out_channels: Number of channels.

    Returns:
        loc: loc of the normal distribution.
        scale: Standard deviation of the normal distribution.
    """
    return torch.randn(out_features, out_channels), torch.rand(out_features, out_channels)


def make_leaf(loc, scale) -> Normal:
    """Create a Normal leaves node.

    Args:
        loc: loc of the normal distribution.
        scale: Standard deviation of the normal distribution.
    """
    scope = Scope(list(range(loc.shape[0])))
    return Normal(scope=scope, loc=loc, scale=scale)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_negative_scale(out_features: int, out_channels: int):
    """Test the constructor of a Normal distribution with negative scale."""
    loc, scale = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(loc=loc, scale=-1.0 * scale).distribution


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_zero_scale(out_features: int, out_channels: int):
    """Test the constructor of a Normal distribution with zero scale."""
    loc, scale = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(loc=loc, scale=0.0 * scale).distribution


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_missing_loc(out_features: int, out_channels: int):
    """Test the constructor of a Normal distribution with missing loc."""
    loc, scale = make_params(out_features, out_channels)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        Normal(scope=scope, loc=None, scale=scale)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_missing_scale(out_features: int, out_channels: int):
    """Test the constructor of a Normal distribution with missing scale."""
    loc, scale = make_params(out_features, out_channels)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        Normal(scope=scope, loc=loc, scale=None)
