"""Unit tests for convolutional sampling-context utilities."""

import torch

from spflow.modules.conv.utils import upsample_sampling_context
from spflow.utils.sampling_context import SamplingContext, to_one_hot


def test_upsample_sampling_context_upsamples_indices_and_mask() -> None:
    ctx = SamplingContext(
        channel_index=torch.zeros((2, 4), dtype=torch.long),
        mask=torch.ones((2, 4), dtype=torch.bool),
        repetition_index=torch.zeros((2,), dtype=torch.long),
    )

    upsample_sampling_context(ctx, current_height=2, current_width=2, scale_h=2, scale_w=2)

    assert ctx.channel_index.shape == (2, 16)
    assert ctx.mask.shape == (2, 16)


def test_upsample_sampling_context_upsamples_differentiable_indices_and_mask() -> None:
    channel_index = torch.zeros((2, 4, 3), dtype=torch.get_default_dtype())
    channel_index[:, :, 0] = 1.0
    ctx = SamplingContext(
        channel_index=channel_index,
        mask=torch.ones((2, 4), dtype=torch.bool),
        repetition_index=torch.ones((2, 1), dtype=torch.get_default_dtype()),
        is_differentiable=True,
    )

    upsample_sampling_context(ctx, current_height=2, current_width=2, scale_h=2, scale_w=2)

    assert ctx.channel_index.shape == (2, 16, 3)
    assert ctx.mask.shape == (2, 16)


def test_upsample_sampling_context_diff_matches_non_diff() -> None:
    channel_index = torch.tensor(
        [[0, 1, 2, 1], [2, 0, 1, 0]],
        dtype=torch.long,
    )
    mask = torch.tensor(
        [[True, False, True, True], [True, True, False, True]],
        dtype=torch.bool,
    )
    repetition_index = torch.zeros((2,), dtype=torch.long)
    ctx_a = SamplingContext(
        channel_index=channel_index.clone(),
        mask=mask.clone(),
        repetition_index=repetition_index.clone(),
    )
    ctx_b = SamplingContext(
        channel_index=to_one_hot(channel_index, dim=-1, dim_size=3),
        mask=mask.clone(),
        repetition_index=to_one_hot(repetition_index, dim=-1, dim_size=1),
        is_differentiable=True,
    )

    upsample_sampling_context(ctx_a, current_height=2, current_width=2, scale_h=2, scale_w=2)
    upsample_sampling_context(ctx_b, current_height=2, current_width=2, scale_h=2, scale_w=2)

    # Boolean mask semantics should stay identical regardless of routing representation.
    assert torch.equal(ctx_a.mask, ctx_b.mask)
    # One-hot diff path must encode the same discrete routing decisions after upsampling.
    torch.testing.assert_close(
        ctx_b.channel_index,
        to_one_hot(ctx_a.channel_index, dim=-1, dim_size=3),
        rtol=0.0,
        atol=0.0,
    )
