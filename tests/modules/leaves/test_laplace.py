from itertools import product

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.leaves import Laplace

num_repetition_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_params(out_features: int, out_channels: int, num_repetitions: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create parameters for a laplace distribution.

    Args:
        out_features: Number of features.
        out_channels: Number of channels.
        num_repetitions: Number of repetitions.

    Returns:
        loc: loc of the laplace distribution.
        scale: Standard deviation of the laplace distribution.
    """
    shape = (out_features, out_channels, num_repetitions)
    return torch.randn(shape), torch.rand(shape)


def make_leaf(loc, scale) -> Laplace:
    """Create a Laplace leaves node.

    Args:
        loc: loc of the laplace distribution.
        scale: Standard deviation of the laplace distribution.
    """
    scope = Scope(list(range(loc.shape[0])))
    return Laplace(scope=scope, loc=loc, scale=scale)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_negative_scale(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Laplace distribution with negative scale."""
    loc, scale = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_leaf(loc=loc, scale=-1.0 * scale).distribution


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_zero_scale(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Laplace distribution with zero scale."""
    loc, scale = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_leaf(loc=loc, scale=0.0 * scale).distribution


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_missing_loc(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Laplace distribution with missing loc."""
    loc, scale = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        Laplace(scope=scope, loc=None, scale=scale)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_missing_scale(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Laplace distribution with missing scale."""
    loc, scale = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        Laplace(scope=scope, loc=loc, scale=None)
