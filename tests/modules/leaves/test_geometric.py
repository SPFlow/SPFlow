from itertools import product

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.leaves.geometric import Geometric

num_repetition_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_module(*, probs: torch.Tensor | None = None, logits: torch.Tensor | None = None) -> Geometric:
    """Create a Geometric leaves node."""
    tensor = probs if probs is not None else logits
    if tensor is None:
        raise ValueError("Either probs or logits must be provided")
    scope = Scope(list(range(tensor.shape[0])))
    return Geometric(scope=scope, probs=probs, logits=logits)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_accepts_probs(out_features: int, out_channels: int, num_repetitions: int):
    probs = torch.rand(out_features, out_channels, num_repetitions)
    node = make_module(probs=probs)
    assert node.probs.shape == probs.shape


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_accepts_logits(out_features: int, out_channels: int, num_repetitions: int):
    logits = torch.randn(out_features, out_channels, num_repetitions)
    node = make_module(logits=logits)
    torch.testing.assert_close(node.params()["logits"], logits, rtol=0.0, atol=0.0)


def test_geometric_invalid_parameter_combination():
    """Test that Geometric raises InvalidParameterCombinationError when both probs and logits are given."""
    scope = Scope([0])
    probs = torch.tensor([0.5])
    logits = torch.tensor([0.0])
    with pytest.raises(InvalidParameterCombinationError, match="accepts either probs or logits, not both"):
        Geometric(scope=scope, probs=probs, logits=logits)
