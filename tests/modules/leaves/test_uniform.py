from itertools import product

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.leaves import Uniform

num_repetition_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_params(
    out_features: int, out_channels: int, num_repetitions: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create parameters for a uniform distribution.

    Args:
        out_features: Number of features.
        out_channels: Number of channels.
        num_repetitions: Number of repetitions.

    Returns:
        low: Lower bound of the uniform distribution.
        high: Upper bound of the uniform distribution.
    """
    shape = (out_features, out_channels, num_repetitions)
    low = torch.rand(shape)
    high = low + torch.rand(shape)
    return low, high


def make_leaf(low, high) -> Uniform:
    """Create a Uniform leaves node.

    Args:
        low: Lower bound of the uniform distribution.
        high: Upper bound of the uniform distribution.
    """
    scope = Scope(list(range(low.shape[0])))
    return Uniform(scope=scope, low=low, high=high)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_uniform_constructor_missing_params(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Uniform distribution with missing parameters."""
    low, high = make_params(out_features, out_channels, num_repetitions)
    scope = Scope(list(range(out_features)))

    with pytest.raises(InvalidParameterCombinationError):
        Uniform(scope=scope, low=None, high=high)

    with pytest.raises(InvalidParameterCombinationError):
        Uniform(scope=scope, low=low, high=None)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_uniform_constructor_equal_bounds(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Uniform distribution with identical bounds."""
    low, _ = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_leaf(low=low, high=low).distribution()


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_uniform_constructor_low_greater_than_high(
    out_features: int, out_channels: int, num_repetitions: int
):
    """Test the constructor of a Uniform distribution with low greater than high."""
    low, high = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_leaf(low=high, high=low).distribution()


def test_uniform_non_finite_params():
    """Test that Uniform raises ValueError when non-finite parameters are provided."""
    scope = Scope([0])
    low = torch.tensor([[[float("nan")]]])
    high = torch.tensor([[[1.0]]])
    with pytest.raises(ValueError):
        Uniform(scope=scope, low=low, high=high)


def test_uniform_log_likelihood_handles_nans_without_broadcast_errors():
    """Uniform log_likelihood should marginalize NaNs without shape/broadcast issues."""
    out_features, out_channels, num_repetitions = 3, 2, 4
    shape = (out_features, out_channels, num_repetitions)
    low = torch.zeros(shape)
    high = torch.ones(shape) * 2.0
    leaf = make_leaf(low=low, high=high)

    data = torch.rand(5, out_features)
    data[1, 0] = torch.nan
    data[3, 2] = torch.nan
    original = data.clone()

    log_prob = leaf.log_likelihood(data)

    assert log_prob.shape == (data.shape[0], out_features, out_channels, num_repetitions)
    torch.testing.assert_close(data, original, equal_nan=True)

    marg_mask = torch.isnan(data).unsqueeze(2).unsqueeze(-1).expand_as(log_prob)
    assert marg_mask.any()
    torch.testing.assert_close(log_prob[marg_mask], torch.zeros_like(log_prob[marg_mask]))
