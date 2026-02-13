"""Differentiable sampling tests for LinsumLayer."""

import pytest
import torch

from spflow.utils.cache import Cache
from spflow.utils.sampling_context import DifferentiableSamplingContext
from tests.modules.einsum.layer_test_utils import make_linsum_single_input, make_linsum_two_inputs


def _random_simplex(*shape: int) -> torch.Tensor:
    probs = torch.rand(*shape)
    return probs / probs.sum(dim=-1, keepdim=True)


def test_rsample_two_inputs_routes_and_backpropagates():
    module = make_linsum_two_inputs(in_channels=2, out_channels=3, in_features=4, num_repetitions=2)

    batch_size = 7
    data = torch.full((batch_size, 8), torch.nan)
    channel_probs = _random_simplex(batch_size, module.out_shape.features, module.out_shape.channels)
    repetition_probs = _random_simplex(batch_size, module.out_shape.repetitions)
    sampling_ctx = DifferentiableSamplingContext(
        channel_probs=channel_probs,
        mask=torch.ones((batch_size, module.out_shape.features), dtype=torch.bool),
        repetition_probs=repetition_probs,
    )

    samples = module.rsample(
        data=data,
        diff_method="gumbel",
    )

    assert samples.shape == data.shape
    assert torch.isfinite(samples[:, module.scope.query]).all()

    loss = samples.square().mean()
    loss.backward()

    assert module.logits.grad is not None
    assert module.inputs[0].loc.grad is not None
    assert module.inputs[1].loc.grad is not None


def test_rsample_single_input_split_routes_and_backpropagates():
    module = make_linsum_single_input(in_channels=2, out_channels=3, in_features=4, num_repetitions=2)

    batch_size = 6
    data = torch.full((batch_size, 4), torch.nan)
    channel_probs = _random_simplex(batch_size, module.out_shape.features, module.out_shape.channels)
    repetition_probs = _random_simplex(batch_size, module.out_shape.repetitions)
    sampling_ctx = DifferentiableSamplingContext(
        channel_probs=channel_probs,
        mask=torch.ones((batch_size, module.out_shape.features), dtype=torch.bool),
        repetition_probs=repetition_probs,
    )

    samples = module.rsample(data=data)

    assert samples.shape == data.shape
    assert torch.isfinite(samples).all()

    loss = samples.square().mean()
    loss.backward()

    assert module.logits.grad is not None
    assert module.inputs.inputs.loc.grad is not None


def test_rsample_accepts_singleton_repetition_probs_for_multi_repetition_modules():
    module = make_linsum_single_input(in_channels=2, out_channels=3, in_features=4, num_repetitions=2)
    batch_size = 4
    data = torch.full((batch_size, 4), torch.nan)
    channel_probs = _random_simplex(batch_size, module.out_shape.features, module.out_shape.channels)
    sampling_ctx = DifferentiableSamplingContext(
        channel_probs=channel_probs,
        mask=torch.ones((batch_size, module.out_shape.features), dtype=torch.bool),
    )

    out = module._rsample(data=data, sampling_ctx=sampling_ctx, cache=Cache())
    if sampling_ctx.sample_accum is not None and sampling_ctx.sample_mass is not None:
        out = sampling_ctx.finalize_with_evidence(out)
    assert out.shape == data.shape
    assert torch.isfinite(out).all()
