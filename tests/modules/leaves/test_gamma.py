from itertools import product

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.leaves import Gamma

# Constants
num_repetition_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_params(out_features: int, out_channels: int, num_repetitions: int) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (out_features, out_channels, num_repetitions)
    return torch.rand(shape), torch.rand(shape)


def make_module(concentration, rate) -> Gamma:
    scope = Scope(list(range(concentration.shape[0])))
    return Gamma(scope=scope, concentration=concentration, rate=rate)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_negative_concentration(out_features: int, out_channels: int, num_repetitions: int):
    concentration, rate = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_module(concentration=torch.full_like(concentration, -1.0), rate=rate).distribution


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_negative_rate(out_features: int, out_channels: int, num_repetitions: int):
    concentration, rate = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_module(concentration=concentration, rate=torch.full_like(rate, -1.0)).distribution


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_zero_concentration(out_features: int, out_channels: int, num_repetitions: int):
    concentration, rate = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_module(concentration=torch.full_like(concentration, 0.0), rate=rate).distribution


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_zero_rate(out_features: int, out_channels: int, num_repetitions: int):
    concentration, rate = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_module(concentration=concentration, rate=torch.full_like(rate, 0.0)).distribution


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_missing_concentration(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Gamma distribution with missing concentration (only rate provided)."""
    concentration, rate = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        Gamma(scope=scope, concentration=None, rate=rate)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_missing_rate(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Gamma distribution with missing rate (only concentration provided)."""
    concentration, rate = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        Gamma(scope=scope, concentration=concentration, rate=None)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_both_none(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Gamma distribution with both concentration and rate as None."""
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        Gamma(scope=scope, concentration=None, rate=None)
