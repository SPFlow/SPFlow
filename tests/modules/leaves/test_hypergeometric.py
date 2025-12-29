from itertools import product

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.leaves import Hypergeometric

num_repetition_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_params(
    out_features: int,
    out_channels: int,
    num_repetitions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create parameters for a hypergeometric distribution.

    Args:
        out_features: Number of features.
        out_channels: Number of channels.
        num_repetitions: Number of repetitions.

    Returns:
        K: Number of success states in the population.
        N: Population size.
        n: Number of draws.
    """
    shape = (out_features, out_channels, num_repetitions)
    N = torch.randint(10, 100, shape)
    K = torch.randint(1, int(N.max().item()), shape).clamp(max=N - 1)
    n = torch.randint(1, int(N.max().item()), shape).clamp(max=N - 1)
    return K, N, n


def make_leaf(K, N, n) -> Hypergeometric:
    """Create a Hypergeometric leaves node.

    Args:
        K: Number of success states in the population.
        N: Population size.
        n: Number of draws.
    """
    scope = Scope(list(range(K.shape[0])))
    return Hypergeometric(scope=scope, K=K, N=N, n=n)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_negative_N(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Hypergeometric distribution with negative N."""
    K, N, n = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_leaf(K=K, N=-1.0 * N, n=n).distribution


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_negative_n(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Hypergeometric distribution with negative n."""
    K, N, n = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_leaf(K=K, N=N, n=-1.0 * n).distribution


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_negative_K(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Hypergeometric distribution with negative K."""
    K, N, n = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_leaf(K=-1.0 * K, N=N, n=n).distribution


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_n_greater_than_N(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Hypergeometric distribution with n > N."""
    K, N, n = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_leaf(K=K, N=N, n=N + 1).distribution


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_K_greater_than_N(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Hypergeometric distribution with K > N."""
    K, N, n = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_leaf(K=N + 1, N=N, n=n).distribution


def test_hypergeometric_missing_parameters():
    """Test that Hypergeometric raises InvalidParameterCombinationError when parameters are missing."""
    scope = Scope([0])
    K = torch.tensor([[5.0]])
    N = torch.tensor([[10.0]])
    with pytest.raises(InvalidParameterCombinationError, match="parameters are required"):
        Hypergeometric(scope=scope, K=K, N=N, n=None)
