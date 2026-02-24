from itertools import product

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.leaves.bernoulli import Bernoulli
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, to_one_hot

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


def test_differentiable_sampling_path_produces_finite_probabilities():
    leaf = Bernoulli(scope=Scope([0]), out_channels=3, num_repetitions=2)
    n_samples = 9
    data = torch.full((n_samples, 1), float("nan"))
    sampling_ctx = SamplingContext(
        channel_index=to_one_hot(
            torch.randint(low=0, high=leaf.out_shape.channels, size=(n_samples, 1)),
            dim=-1,
            dim_size=leaf.out_shape.channels,
        ),
        mask=torch.ones((n_samples, 1), dtype=torch.bool),
        repetition_index=to_one_hot(
            torch.randint(low=0, high=leaf.out_shape.repetitions, size=(n_samples,)),
            dim=-1,
            dim_size=leaf.out_shape.repetitions,
        ),
        is_differentiable=True,
        is_mpe=False,
    )

    samples = leaf._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())

    assert samples.shape == (n_samples, 1)
    assert torch.isfinite(samples).all()
    assert (samples >= 0.0).all()
    assert (samples <= 1.0).all()
