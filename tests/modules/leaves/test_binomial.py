from itertools import product

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.leaves.binomial import Binomial

num_repetition_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_params(
    out_features: int, out_channels: int, num_repetitions: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create parameters for a binomial distribution.

    Args:
        out_features: Number of features.
        out_channels: Number of channels.
        num_repetitions: Number of repetitions.

    Returns:
        n: Number of trials.
        p: Probability of success in each trial.

    """
    shape = (out_features, out_channels, num_repetitions)
    return torch.randint(1, 10, shape), torch.rand(shape)


def make_module(n, p=None, logits=None) -> Binomial:
    """Create a Binomial leaves node.

    Args:
        n: Number of trials.
        p: Probability of success in each trial.
    """
    tensor = p if p is not None else logits
    if tensor is None:
        raise ValueError("Either p or logits must be provided")
    scope = Scope(list(range(tensor.shape[0])))
    return Binomial(scope=scope, total_count=n, probs=p, logits=logits)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_accepts_probs(out_features: int, out_channels: int, num_repetitions: int):
    """Binomial accepts probability tensors."""
    n, p = make_params(out_features, out_channels, num_repetitions)
    node = make_module(n=n, p=p)
    assert node.probs.shape == p.shape
    torch.testing.assert_close(node.params()["logits"], node.logits, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_accepts_logits(out_features: int, out_channels: int, num_repetitions: int):
    """Binomial accepts logits tensors."""
    n, _ = make_params(out_features, out_channels, num_repetitions)
    logits = torch.randn(out_features, out_channels, num_repetitions)
    node = make_module(n=n, logits=logits)
    torch.testing.assert_close(node.params()["logits"], logits, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_probs_greater_than_one(out_features: int, out_channels: int, num_repetitions: int):
    """Invalid probabilities (>1.0) trigger torch validation."""
    n, p = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_module(n=n, p=1.5 + p).distribution


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_probs_negative(out_features: int, out_channels: int, num_repetitions: int):
    """Invalid probabilities (<0.0) trigger torch validation."""
    n, p = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_module(n=n, p=p - 1.5).distribution


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_n_negative(out_features: int, out_channels: int, num_repetitions: int):
    """Negative n values are invalid."""
    n, p = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_module(n=torch.full_like(n, -1.0), p=p).distribution


def test_binomial_invalid_parameter_combination():
    """Test that Binomial raises InvalidParameterCombinationError when both probs and logits are given."""
    scope = Scope([0])
    n = torch.tensor([10])
    probs = torch.tensor([0.5])
    logits = torch.tensor([0.0])
    with pytest.raises(InvalidParameterCombinationError):
        Binomial(scope=scope, total_count=n, probs=probs, logits=logits)


def test_binomial_missing_n():
    """Test that Binomial raises InvalidParameterCombinationError when n is missing."""
    scope = Scope([0])
    probs = torch.tensor([0.5])
    with pytest.raises(InvalidParameterCombinationError):
        Binomial(scope=scope, total_count=None, probs=probs)
