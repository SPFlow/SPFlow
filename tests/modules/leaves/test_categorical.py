from itertools import product

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.leaves import Categorical

num_repetition_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]

NUM_CATEGORIES = 4


def make_module(*, probs: torch.Tensor | None = None, logits: torch.Tensor | None = None) -> Categorical:
    """Create a Categorical leaves node."""
    tensor = probs if probs is not None else logits
    if tensor is None:
        raise ValueError("Either probs or logits must be provided")
    scope = Scope(list(range(tensor.shape[0])))
    return Categorical(scope=scope, probs=probs, logits=logits)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_accepts_probs(out_features: int, out_channels: int, num_repetitions: int):
    """Categorical accepts probability tensors."""
    probs = torch.rand(out_features, out_channels, num_repetitions, NUM_CATEGORIES)
    node = make_module(probs=probs)
    assert node.probs.shape == probs.shape
    probs_sum = node.probs.sum(dim=-1)
    torch.testing.assert_close(probs_sum, torch.ones_like(probs_sum), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_accepts_logits(out_features: int, out_channels: int, num_repetitions: int):
    """Categorical accepts logits parameterization."""
    logits = torch.randn(out_features, out_channels, num_repetitions, NUM_CATEGORIES)
    node = make_module(logits=logits)
    torch.testing.assert_close(node.params()["logits"], logits, rtol=0.0, atol=0.0)


def test_categorical_invalid_parameter_combination():
    """Test that Categorical raises InvalidParameterCombinationError when both probs and logits are given."""
    scope = Scope([0])
    probs = torch.tensor([[0.25, 0.75]])
    logits = torch.tensor([[0.0, 0.0]])
    with pytest.raises(InvalidParameterCombinationError, match="accepts either probs or logits, not both"):
        Categorical(scope=scope, probs=probs, logits=logits)
