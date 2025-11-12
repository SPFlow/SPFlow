from itertools import product

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.leaf import LogNormal

out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_params(out_features: int, out_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create parameters for a normal distribution.

    Args:
        out_features: Number of features.
        out_channels: Number of channels.

    Returns:
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
    """
    return torch.randn(out_features, out_channels), torch.rand(out_features, out_channels)


def make_leaf(mean, std) -> LogNormal:
    """
    Create a LogNormal leaf node.
    Args:
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
    """
    scope = Scope(list(range(mean.shape[0])))
    return LogNormal(scope=scope, mean=mean, std=std)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_valid_params(out_features: int, out_channels: int):
    """Test the constructor of a LogNormal distribution with valid parameters."""
    mean, std = make_params(out_features, out_channels)
    leaf = make_leaf(mean=mean, std=std)
    assert torch.isclose(leaf.mean, mean).all()
    assert torch.isclose(leaf.std, std).all()


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_negative_std(out_features: int, out_channels: int):
    """Test the constructor of a LogNormal distribution with negative std."""
    mean, std = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(mean=mean, std=-1.0 * std)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_zero_std(out_features: int, out_channels: int):
    """Test the constructor of a LogNormal distribution with zero std."""
    mean, std = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(mean=mean, std=0.0 * std)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_nan_mean(out_features: int, out_channels: int):
    """Test the constructor of a LogNormal distribution with NaN mean."""
    mean, std = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(mean=mean * torch.nan, std=std)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_nan_std(out_features: int, out_channels: int):
    """Test the constructor of a LogNormal distribution with NaN std."""
    mean, std = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(mean=mean, std=std * torch.nan)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_missing_mean(out_features: int, out_channels: int):
    """Test the constructor of a LogNormal distribution with missing mean."""
    mean, std = make_params(out_features, out_channels)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        LogNormal(scope=scope, mean=None, std=std)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_missing_std(out_features: int, out_channels: int):
    """Test the constructor of a LogNormal distribution with missing std."""
    mean, std = make_params(out_features, out_channels)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        LogNormal(scope=scope, mean=mean, std=None)
