from itertools import product

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.leaves.bernoulli import Bernoulli

num_repetition_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_module(*, probs: torch.Tensor | None = None, logits: torch.Tensor | None = None) -> Bernoulli:
    """Create a Bernoulli leaves node."""
    tensor = probs if probs is not None else logits
    if tensor is None:
        raise ValueError("Either probs or logits must be provided")
    scope = Scope(list(range(tensor.shape[0])))
    return Bernoulli(scope=scope, probs=probs, logits=logits)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_accepts_probs(out_features: int, out_channels: int, num_repetitions: int):
    probs = torch.rand(out_features, out_channels, num_repetitions)
    node = make_module(probs=probs)
    assert node.probs.shape == probs.shape
    torch.testing.assert_close(node.params()["logits"], node.logits, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_accepts_logits(out_features: int, out_channels: int, num_repetitions: int):
    logits = torch.randn(out_features, out_channels, num_repetitions)
    node = make_module(logits=logits)
    assert node.logits.shape == logits.shape
    torch.testing.assert_close(node.params()["logits"], logits, rtol=0.0, atol=0.0)


def test_bernoulli_invalid_parameter_combination():
    """Test that Bernoulli raises InvalidParameterCombinationError when both probs and logits are given."""
    scope = Scope([0])
    probs = torch.tensor([0.5])
    logits = torch.tensor([0.0])
    with pytest.raises(InvalidParameterCombinationError):
        Bernoulli(scope=scope, probs=probs, logits=logits)
