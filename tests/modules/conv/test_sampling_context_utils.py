"""Unit tests for convolutional sampling-context utilities."""

import torch

from spflow.modules.conv.utils import (
    upsample_differentiable_sampling_context,
    upsample_sampling_context,
)
from spflow.utils.sampling_context import DifferentiableSamplingContext, SamplingContext


def test_upsample_sampling_context_upsamples_indices_and_mask() -> None:
    ctx = SamplingContext(
        channel_index=torch.zeros((2, 4), dtype=torch.long),
        mask=torch.ones((2, 4), dtype=torch.bool),
    )

    upsample_sampling_context(ctx, current_height=2, current_width=2, scale_h=2, scale_w=2)

    assert ctx.channel_index.shape == (2, 16)
    assert ctx.mask.shape == (2, 16)


def test_upsample_differentiable_sampling_context_upsamples_probs_and_mask() -> None:
    channel_probs = torch.ones((2, 4, 3), dtype=torch.get_default_dtype()) / 3.0
    ctx = DifferentiableSamplingContext(
        channel_probs=channel_probs,
        mask=torch.ones((2, 4), dtype=torch.bool),
    )

    upsample_differentiable_sampling_context(
        ctx,
        current_height=2,
        current_width=2,
        scale_h=2,
        scale_w=2,
    )

    assert ctx.channel_probs.shape == (2, 16, 3)
    assert ctx.mask.shape == (2, 16)
