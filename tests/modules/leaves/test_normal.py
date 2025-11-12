from itertools import product

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.leaves import Normal

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


def make_leaf(mean, std) -> Normal:
    """
    Create a Normal leaves node.
    Args:
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
    """
    scope = Scope(list(range(mean.shape[0])))
    return Normal(scope=scope, mean=mean, std=std)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_negative_std(out_features: int, out_channels: int):
    """Test the constructor of a Normal distribution with negative std."""
    mean, std = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(mean=mean, std=-1.0 * std)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_zero_std(out_features: int, out_channels: int):
    """Test the constructor of a Normal distribution with zero std."""
    mean, std = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(mean=mean, std=0.0 * std)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_missing_mean(out_features: int, out_channels: int):
    """Test the constructor of a Normal distribution with missing mean."""
    mean, std = make_params(out_features, out_channels)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        Normal(scope=scope, mean=None, std=std)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_missing_std(out_features: int, out_channels: int):
    """Test the constructor of a Normal distribution with missing std."""
    mean, std = make_params(out_features, out_channels)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        Normal(scope=scope, mean=mean, std=None)
