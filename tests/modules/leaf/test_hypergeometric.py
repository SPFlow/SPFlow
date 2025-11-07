from tests.fixtures import auto_set_test_seed, auto_set_test_device
import unittest
from itertools import product

from spflow.meta.dispatch import init_default_sampling_context
import pytest
import torch
from pytest import raises
from scipy.stats import hypergeom

from spflow import maximum_likelihood_estimation, sample, marginalize, log_likelihood
from spflow.meta import Scope
from spflow.modules.leaf import Hypergeometric


import unittest
from itertools import product
from spflow import InvalidParameterCombinationError
from tests.fixtures import auto_set_test_seed, auto_set_test_device
import pytest
import torch
from spflow.meta import Scope
from spflow.modules.leaf import Hypergeometric

out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_params(out_features: int, out_channels: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create parameters for a hypergeometric distribution.

    Args:
        out_features: Number of features.
        out_channels: Number of channels.

    Returns:
        K: Number of success states in the population.
        N: Population size.
        n: Number of draws.
    """
    N = torch.randint(10, 100, (out_features, out_channels))
    K = torch.randint(1, N.max().item(), (out_features, out_channels)).clamp(max=N - 1)
    n = torch.randint(1, N.max().item(), (out_features, out_channels)).clamp(max=N - 1)
    return K, N, n


def make_leaf(K, N, n) -> Hypergeometric:
    """
    Create a Hypergeometric leaf node.
    Args:
        K: Number of success states in the population.
        N: Population size.
        n: Number of draws.
    """
    scope = Scope(list(range(K.shape[0])))
    return Hypergeometric(scope=scope, K=K, N=N, n=n)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_negative_N(out_features: int, out_channels: int):
    """Test the constructor of a Hypergeometric distribution with negative N."""
    K, N, n = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(K=K, N=-1.0 * N, n=n)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_negative_n(out_features: int, out_channels: int):
    """Test the constructor of a Hypergeometric distribution with negative n."""
    K, N, n = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(K=K, N=N, n=-1.0 * n)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_negative_K(out_features: int, out_channels: int):
    """Test the constructor of a Hypergeometric distribution with negative K."""
    K, N, n = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(K=-1.0 * K, N=N, n=n)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_n_greater_than_N(out_features: int, out_channels: int):
    """Test the constructor of a Hypergeometric distribution with n > N."""
    K, N, n = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(K=K, N=N, n=N + 1)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_K_greater_than_N(out_features: int, out_channels: int):
    """Test the constructor of a Hypergeometric distribution with K > N."""
    K, N, n = make_params(out_features, out_channels)
    with pytest.raises(ValueError):
        make_leaf(K=N + 1, N=N, n=n)
