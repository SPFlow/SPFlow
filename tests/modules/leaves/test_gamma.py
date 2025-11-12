from itertools import product

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.leaves import Gamma

# Constants
out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_params(out_features: int, out_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.rand(out_features, out_channels), torch.rand(out_features, out_channels)


def make_module(alpha, beta) -> Gamma:
    scope = Scope(list(range(alpha.shape[0])))
    return Gamma(scope=scope, alpha=alpha, beta=beta)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_negative_alpha(out_features: int, out_channels: int):
    alpha, beta = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_module(alpha=torch.full_like(alpha, -1.0), beta=beta)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_negative_beta(out_features: int, out_channels: int):
    alpha, beta = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_module(alpha=alpha, beta=torch.full_like(beta, -1.0))


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_zero_alpha(out_features: int, out_channels: int):
    alpha, beta = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_module(alpha=torch.full_like(alpha, 0.0), beta=beta)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_zero_beta(out_features: int, out_channels: int):
    alpha, beta = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_module(alpha=alpha, beta=torch.full_like(beta, 0.0))


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_missing_alpha(out_features: int, out_channels: int):
    """Test the constructor of a Gamma distribution with missing alpha (only beta provided)."""
    alpha, beta = make_params(out_features, out_channels)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        Gamma(scope=scope, alpha=None, beta=beta)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_missing_beta(out_features: int, out_channels: int):
    """Test the constructor of a Gamma distribution with missing beta (only alpha provided)."""
    alpha, beta = make_params(out_features, out_channels)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        Gamma(scope=scope, alpha=alpha, beta=None)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_both_none(out_features: int, out_channels: int):
    """Test the constructor of a Gamma distribution with both alpha and beta as None."""
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        Gamma(scope=scope, alpha=None, beta=None)
