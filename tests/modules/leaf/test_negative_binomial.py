import unittest
from itertools import product
from spflow.exceptions import InvalidParameterCombinationError
from tests.fixtures import auto_set_test_seed, auto_set_test_device
import pytest
import torch
from spflow.meta.data import Scope
from spflow.modules.leaf.negative_binomial import NegativeBinomial

out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_params(out_features: int, out_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create parameters for a binomial distribution.

    Args:
        out_features: Number of features.
        out_channels: Number of channels.

    Returns:
        n: Number of trials.
        p: Probability of success in each trial.

    """
    return torch.randint(1, 10, (out_features, out_channels)), torch.rand(out_features, out_channels)


def make_module(n, p) -> NegativeBinomial:
    """
    Create a NegativeBinomial leaf node.
    Args:
        n: Number of trials.
        p: Probability of success in each trial.
    """
    scope = Scope(list(range(p.shape[0])))
    return NegativeBinomial(scope=scope, n=n, p=p)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_p_greater_than_one(out_features: int, out_channels: int):
    """Test the constructor of a Bernoulli distribution with p greater than 1.0."""
    n, p = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_module(n=n, p=1.5 + p)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_p_smaller_than_zero(out_features: int, out_channels: int):
    """Test the constructor of a Bernoulli distribution with p smaller than 1.0."""
    n, p = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_module(n=n, p=p - 1.5)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_n_negative(out_features: int, out_channels: int):
    """Test the constructor of a Bernoulli distribution with p smaller than 1.0."""
    n, p = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_module(n=torch.full_like(n, -10.0), p=p)
