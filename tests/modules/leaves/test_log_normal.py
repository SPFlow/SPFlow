from itertools import product

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.leaves import LogNormal

out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_params(out_features: int, out_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create parameters for a normal distribution.

    Args:
        out_features: Number of features.
        out_channels: Number of channels.

    Returns:
        loc: Mean of the normal distribution.
        scale: Standard deviation of the normal distribution.
    """
    return torch.randn(out_features, out_channels), torch.rand(out_features, out_channels)


def make_leaf(loc, scale) -> LogNormal:
    """Create a LogNormal leaves node.

    Args:
        loc: Mean of the normal distribution.
        scale: Standard deviation of the normal distribution.
    """
    scope = Scope(list(range(loc.shape[0])))
    return LogNormal(scope=scope, loc=loc, scale=scale)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_valid_params(out_features: int, out_channels: int):
    """Test the constructor of a LogNormal distribution with valid parameters."""
    loc, scale = make_params(out_features, out_channels)
    leaf = make_leaf(loc=loc, scale=scale)
    assert torch.isclose(leaf.loc, loc).all()
    assert torch.isclose(leaf.scale, scale).all()


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_negative_scale(out_features: int, out_channels: int):
    """PyTorch validation triggers for negative scale."""
    loc, scale = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(loc=loc, scale=-1.0 * scale).distribution


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_zero_scale(out_features: int, out_channels: int):
    """PyTorch validation triggers for zero scale."""
    loc, scale = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(loc=loc, scale=0.0 * scale).distribution


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_nan_loc(out_features: int, out_channels: int):
    """PyTorch validation triggers for NaN loc."""
    loc, scale = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(loc=loc * torch.nan, scale=scale).distribution


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_nan_scale(out_features: int, out_channels: int):
    """PyTorch validation triggers for NaN scale."""
    loc, scale = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(loc=loc, scale=scale * torch.nan).distribution


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_missing_loc(out_features: int, out_channels: int):
    """Test the constructor of a LogNormal distribution with missing mean."""
    loc, scale = make_params(out_features, out_channels)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        LogNormal(scope=scope, loc=None, scale=scale)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_missing_scale(out_features: int, out_channels: int):
    """Test the constructor of a LogNormal distribution with missing std."""
    loc, scale = make_params(out_features, out_channels)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        LogNormal(scope=scope, loc=loc, scale=None)
