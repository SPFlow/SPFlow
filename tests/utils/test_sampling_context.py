"""Tests for sampling context utilities."""

import torch
import pytest

from spflow.utils.sampling_context import SamplingContext, _check_mask_bool


def test_sampling_context_init_defaults():
    """SamplingContext initializes default mask and channel index when none are provided."""
    ctx = SamplingContext(num_samples=3)

    assert ctx.mask.shape == (3, 1)
    assert ctx.channel_index.shape == (3, 1)
    assert ctx.mask.dtype == torch.bool
    assert ctx.channel_index.dtype == torch.long
    assert ctx.samples_mask.tolist() == [True, True, True]


def test_sampling_context_init_with_tensors():
    """SamplingContext keeps provided tensors and device."""
    channel_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    mask = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    repetition_index = torch.tensor([0, 1], dtype=torch.long)

    ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)

    assert ctx.channel_index is channel_index
    assert ctx.mask is mask
    assert ctx.repetition_idx is repetition_index
    assert ctx.samples_mask.tolist() == [True, True]
    assert torch.equal(ctx.channel_index_masked, channel_index[[0, 1]])


def test_sampling_context_init_shape_mismatch():
    """channel_index and mask must have matching shapes."""
    channel_index = torch.zeros((2, 2), dtype=torch.long)
    mask = torch.zeros((2, 1), dtype=torch.bool)

    with pytest.raises(ValueError, match="same shape"):
        SamplingContext(channel_index=channel_index, mask=mask)


def test_sampling_context_init_xor_error():
    """Providing only one of channel_index or mask triggers XOR validation."""
    channel_index = torch.zeros((1, 1), dtype=torch.long)

    with pytest.raises(ValueError, match="both None or both not None"):
        SamplingContext(channel_index=channel_index, mask=None)


def test_sampling_context_init_rejects_non_bool_mask():
    """Masks must be boolean tensors."""
    channel_index = torch.zeros((1, 1), dtype=torch.long)
    mask = torch.ones((1, 1), dtype=torch.float32)

    with pytest.raises(ValueError, match="Mask must be of type torch.bool"):
        SamplingContext(channel_index=channel_index, mask=mask)


def test_sampling_context_update_validation():
    """update enforces matching shapes and boolean mask."""
    channel_index = torch.zeros((2, 1), dtype=torch.long)
    mask = torch.ones((2, 1), dtype=torch.bool)
    ctx = SamplingContext(channel_index=channel_index, mask=mask)

    new_channel_index = torch.ones((2, 1), dtype=torch.long)
    new_mask = torch.tensor([[True], [False]], dtype=torch.bool)
    ctx.update(channel_index=new_channel_index, mask=new_mask)

    assert torch.equal(ctx.channel_index, new_channel_index)
    assert torch.equal(ctx.mask, new_mask)

    with pytest.raises(ValueError, match="same shape"):
        ctx.update(channel_index=torch.zeros((1, 1), dtype=torch.long), mask=new_mask)

    with pytest.raises(ValueError, match="Mask must be of type torch.bool"):
        ctx.update(channel_index=new_channel_index, mask=torch.ones((2, 1), dtype=torch.int64))


def test_sampling_context_property_setters():
    """channel_index and mask setters validate shapes and dtypes."""
    channel_index = torch.zeros((2, 2), dtype=torch.long)
    mask = torch.ones((2, 2), dtype=torch.bool)
    ctx = SamplingContext(channel_index=channel_index, mask=mask)

    with pytest.raises(ValueError, match="same shape"):
        ctx.channel_index = torch.zeros((1, 2), dtype=torch.long)

    with pytest.raises(ValueError, match="same shape"):
        ctx.mask = torch.ones((1, 2), dtype=torch.bool)

    with pytest.raises(ValueError, match="Mask must be of type torch.bool"):
        ctx.mask = torch.ones((2, 2), dtype=torch.int64)

    updated_mask = torch.tensor([[True, False], [True, True]], dtype=torch.bool)
    ctx.mask = updated_mask
    assert torch.equal(ctx.mask, updated_mask)


def test_sampling_context_samples_and_channels_masking():
    """samples_mask and channel_index_masked reflect masked samples."""
    channel_index = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)
    mask = torch.tensor([[True, False], [False, False], [True, True]], dtype=torch.bool)
    ctx = SamplingContext(channel_index=channel_index, mask=mask)

    assert ctx.samples_mask.tolist() == [True, False, True]
    expected_masked_channels = torch.tensor([[0, 1], [4, 5]], dtype=torch.long)
    assert torch.equal(ctx.channel_index_masked, expected_masked_channels)


def test_sampling_context_copy_is_deep():
    """copy returns independent tensors, including repetition index."""
    channel_index = torch.tensor([[0], [1]], dtype=torch.long)
    mask = torch.tensor([[True], [False]], dtype=torch.bool)
    repetition_index = torch.tensor([2, 3], dtype=torch.long)
    ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)

    copied = ctx.copy()

    copied.channel_index[0, 0] = 5
    copied.mask[1, 0] = True
    assert copied.repetition_idx is not ctx.repetition_idx
    assert torch.equal(ctx.channel_index, channel_index)
    assert torch.equal(ctx.mask, mask)
    assert torch.equal(ctx.repetition_idx, repetition_index)


def test_check_mask_bool_helper():
    """_check_mask_bool raises on non-boolean masks."""
    with pytest.raises(ValueError, match="Mask must be of type torch.bool"):
        _check_mask_bool(torch.ones((1, 1), dtype=torch.float32))
