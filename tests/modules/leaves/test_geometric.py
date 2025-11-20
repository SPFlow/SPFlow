from itertools import product

import pytest
import torch

from spflow.meta import Scope
from spflow.modules.leaves.geometric import Geometric

num_repetition_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_module(*, probs: torch.Tensor | None = None, logits: torch.Tensor | None = None) -> Geometric:
    """Create a Geometric leaves node."""
    tensor = probs if probs is not None else logits
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
    assert torch.allclose(node.params()["logits"], logits)
