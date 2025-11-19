from itertools import product

import pytest
import torch

from spflow.meta import Scope
from spflow.modules.leaves import Categorical

out_channels_values = [1, 5]
out_features_values = [1, 6]

NUM_CATEGORIES = 4


def make_module(*, probs: torch.Tensor | None = None, logits: torch.Tensor | None = None) -> Categorical:
    """Create a Categorical leaves node."""
    tensor = probs if probs is not None else logits
    scope = Scope(list(range(tensor.shape[0])))
    return Categorical(scope=scope, probs=probs, logits=logits)


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_accepts_probs(out_features: int, out_channels: int):
    """Categorical accepts probability tensors."""
    probs = torch.rand(out_features, out_channels, NUM_CATEGORIES)
    node = make_module(probs=probs)
    assert node.probs.shape == probs.shape
    assert torch.allclose(node.probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)))


@pytest.mark.parametrize("out_features,out_channels", product(out_features_values, out_channels_values))
def test_constructor_accepts_logits(out_features: int, out_channels: int):
    """Categorical accepts logits parameterization."""
    logits = torch.randn(out_features, out_channels, NUM_CATEGORIES)
    node = make_module(logits=logits)
    assert torch.allclose(node.params()["logits"], logits)
