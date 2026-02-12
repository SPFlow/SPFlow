"""Unit tests for convolutional sampling-context utilities."""

import pytest
import torch

from spflow.exceptions import ShapeError
from spflow.modules.conv.utils import _maybe_resize_selector_features, upsample_sampling_context
from spflow.utils.sampling_context import SamplingContext


def test_maybe_resize_selector_features_raises_on_incompatible_width() -> None:
    ctx = SamplingContext(
        channel_index=torch.zeros((2, 4), dtype=torch.long),
        mask=torch.ones((2, 4), dtype=torch.bool),
    )
    ctx.channel_select = torch.zeros((2, 3, 2))

    with pytest.raises(ShapeError, match="channel_select has incompatible feature width"):
        _maybe_resize_selector_features(ctx, current_features=4, target_features=8)


def test_upsample_sampling_context_raises_on_incompatible_selector_width() -> None:
    ctx = SamplingContext(
        channel_index=torch.zeros((2, 4), dtype=torch.long),
        mask=torch.ones((2, 4), dtype=torch.bool),
    )
    ctx.channel_select = torch.zeros((2, 3, 2))

    with pytest.raises(ShapeError, match="channel_select has incompatible feature width"):
        upsample_sampling_context(ctx, current_height=2, current_width=2, scale_h=2, scale_w=2)


def test_upsample_sampling_context_broadcasts_singleton_selectors() -> None:
    ctx = SamplingContext(
        channel_index=torch.zeros((2, 4), dtype=torch.long),
        mask=torch.ones((2, 4), dtype=torch.bool),
    )
    ctx.channel_select = torch.zeros((2, 1, 3))
    ctx.repetition_select = torch.zeros((2, 1, 2))

    upsample_sampling_context(ctx, current_height=2, current_width=2, scale_h=2, scale_w=2)

    assert ctx.channel_index.shape == (2, 16)
    assert ctx.mask.shape == (2, 16)
    assert ctx.channel_select.shape == (2, 16, 3)
    assert ctx.repetition_select.shape == (2, 16, 2)
