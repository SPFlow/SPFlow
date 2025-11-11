import unittest
from itertools import product
from spflow import InvalidParameterCombinationError
from tests.fixtures import auto_set_test_seed, auto_set_test_device
import pytest
import torch
from spflow.meta import Scope
from spflow.modules.leaf.binomial import Binomial

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


def make_module(n, p) -> Binomial:
    """
    Create a Binomial leaf node.
    Args:
        n: Number of trials.
        p: Probability of success in each trial.
    """
    scope = Scope(list(range(p.shape[0])))
    return Binomial(scope=scope, n=n, p=p)


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
        make_module(n=torch.full_like(n, 0.0), p=p)


def test_binomial_support_handles_channels_and_repetitions():
    scope = Scope([0])
    n = torch.full((1, 2, 2), 5.0)
    p = torch.full((1, 2, 2), 0.5)
    leaf = Binomial(scope=scope, n=n, p=p, num_repetitions=2)

    valid = torch.tensor([[[[3.0]]]])
    invalid_fraction = torch.tensor([[[[1.5]]]])
    invalid_high = torch.tensor([[[[6.0]]]])

    assert torch.all(leaf.check_support(valid))
    assert not torch.all(leaf.check_support(invalid_fraction))
    assert not torch.all(leaf.check_support(invalid_high))
