from itertools import product

import pytest
import torch

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
    assert torch.allclose(node.params()["logits"], node.logits)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_accepts_logits(out_features: int, out_channels: int, num_repetitions: int):
    """Binomial accepts logits tensors."""
    n, _ = make_params(out_features, out_channels, num_repetitions)
    logits = torch.randn(out_features, out_channels, num_repetitions)
    node = make_module(n=n, logits=logits)
    assert torch.allclose(node.params()["logits"], logits)


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
