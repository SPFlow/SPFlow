import unittest
from itertools import product
from spflow.exceptions import InvalidParameterCombinationError
from tests.fixtures import auto_set_test_seed
import pytest
import torch
from spflow.meta.data import Scope
from spflow.modules.leaf.cond_normal import CondNormal
from tests.utils.leaves import make_data
from spflow.modules import leaf
from spflow import log_likelihood

out_channels_values = [1, 5]
out_features_values = [1, 4]
out_features = 4


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


def make_leaf() -> CondNormal:
    """
    Create a CondNormal leaf node.
    Args:
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
    """

    mean = torch.tensor([0.0, 0.0, 0.0, 0.0]).unsqueeze(1)
    std = torch.tensor([1.0, 1.0, 1.0, 1.0]).unsqueeze(1)
    cond_f = lambda data: {"mean": mean, "std": std}
    scope = Scope(list(range(mean.shape[0])))
    return CondNormal(scope=scope, cond_f=cond_f)


def test_log_likelihood():
    """Test the log likelihood of a normal distribution."""
    module = make_leaf()
    data = make_data(cls=leaf.Normal, out_features=out_features, n_samples=5)
    #mean = torch.tensor([0.0, 0.0, 0.0, 0.0]).unsqueeze(1)
    #std = torch.tensor([1.0, 1.0, 1.0, 1.0]).unsqueeze(1)
    #cond_f = lambda data: {"mean": mean, "std": std}
    #module.set_cond_f(cond_f)
    result = log_likelihood(module, data)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_negative_std(out_features: int, out_channels: int):
    """Test the constructor of a CondNormal distribution with negative std."""
    mean, std = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(mean=mean, std=-1.0 * std)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_zero_std(out_features: int, out_channels: int):
    """Test the constructor of a CondNormal distribution with zero std."""
    mean, std = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(mean=mean, std=0.0 * std)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_missing_mean(out_features: int, out_channels: int):
    """Test the constructor of a CondNormal distribution with missing mean."""
    mean, std = make_params(out_features, out_channels)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        CondNormal(scope=scope, mean=None, std=std)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_missing_std(out_features: int, out_channels: int):
    """Test the constructor of a CondNormal distribution with missing std."""
    mean, std = make_params(out_features, out_channels)
    with pytest.raises(InvalidParameterCombinationError):
        scope = Scope(list(range(out_features)))
        CondNormal(scope=scope, mean=mean, std=None)
