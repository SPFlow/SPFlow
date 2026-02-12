"""Tests for sampling context utilities."""

import torch
import pytest

from spflow.exceptions import InvalidParameterError, ShapeError
from spflow.utils.sampling_context import (
    SamplingContext,
    build_root_sampling_context,
    init_default_sampling_context,
)


def test_sampling_context_init_defaults():
    """SamplingContext initializes default mask and channel index when none are provided."""
    ctx = SamplingContext(num_samples=3)

    assert ctx.mask.shape == (3, 1)
    assert ctx.channel_index.shape == (3, 1)
    assert ctx.mask.dtype == torch.bool
    assert ctx.channel_index.dtype == torch.long
    assert ctx.samples_mask.tolist() == [True, True, True]


def test_sampling_context_requires_num_samples_when_no_tensors_provided():
    """SamplingContext requires num_samples if channel_index/mask are not provided."""
    with pytest.raises(ValueError):
        SamplingContext()


def test_init_default_sampling_context_requires_num_samples():
    """init_default_sampling_context requires num_samples when no context is provided."""
    with pytest.raises(ValueError):
        init_default_sampling_context(sampling_ctx=None, num_samples=None)


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

    with pytest.raises(ValueError):
        SamplingContext(channel_index=channel_index, mask=mask)


def test_sampling_context_init_xor_error():
    """Providing only one of channel_index or mask triggers XOR validation."""
    channel_index = torch.zeros((1, 1), dtype=torch.long)

    with pytest.raises(ValueError):
        SamplingContext(channel_index=channel_index, mask=None)


def test_sampling_context_init_rejects_non_bool_mask():
    """Masks must be boolean tensors."""
    channel_index = torch.zeros((1, 1), dtype=torch.long)
    mask = torch.ones((1, 1), dtype=torch.float32)

    with pytest.raises(ValueError):
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

    with pytest.raises(ValueError):
        ctx.update(channel_index=torch.zeros((1, 1), dtype=torch.long), mask=new_mask)

    with pytest.raises(ValueError):
        ctx.update(channel_index=new_channel_index, mask=torch.ones((2, 1), dtype=torch.int64))


def test_sampling_context_property_setters():
    """channel_index and mask setters validate shapes and dtypes."""
    channel_index = torch.zeros((2, 2), dtype=torch.long)
    mask = torch.ones((2, 2), dtype=torch.bool)
    ctx = SamplingContext(channel_index=channel_index, mask=mask)

    with pytest.raises(ValueError):
        ctx.channel_index = torch.zeros((1, 2), dtype=torch.long)

    with pytest.raises(ValueError):
        ctx.mask = torch.ones((1, 2), dtype=torch.bool)

    with pytest.raises(ValueError):
        ctx.mask = torch.ones((2, 3), dtype=torch.bool)

    with pytest.raises(ValueError):
        ctx.mask = torch.ones((2, 2), dtype=torch.int64)

    updated_mask = torch.tensor([[True, False], [True, True]], dtype=torch.bool)
    ctx.mask = updated_mask
    assert torch.equal(ctx.mask, updated_mask)


def test_build_root_sampling_context_validates_feature_width():
    """Provided root context must match expected feature width."""
    ctx = SamplingContext(
        channel_index=torch.zeros((2, 3), dtype=torch.long),
        mask=torch.ones((2, 3), dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="features=3, expected 1"):
        build_root_sampling_context(
            sampling_ctx=ctx,
            module_name="TestRoot",
            num_samples=2,
            num_features=1,
        )


def test_build_root_sampling_context_validates_channel_mask_consistency():
    """Root context validation should catch inconsistent channel/mask shapes."""
    ctx = SamplingContext(
        channel_index=torch.zeros((2, 2), dtype=torch.long),
        mask=torch.ones((2, 2), dtype=torch.bool),
    )
    # Bypass setter validations to simulate corrupted state.
    ctx._mask = torch.ones((2, 3), dtype=torch.bool)  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="mismatched channel_index/mask shapes"):
        build_root_sampling_context(
            sampling_ctx=ctx,
            module_name="TestRoot",
            num_samples=2,
            num_features=2,
        )


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


def test_require_feature_width_accepts_matching_width():
    ctx = SamplingContext(
        channel_index=torch.zeros((2, 3), dtype=torch.long),
        mask=torch.ones((2, 3), dtype=torch.bool),
    )
    ctx.require_feature_width(expected_features=3)


def test_require_feature_width_rejects_mismatch():
    ctx = SamplingContext(
        channel_index=torch.zeros((2, 2), dtype=torch.long),
        mask=torch.ones((2, 2), dtype=torch.bool),
    )
    with pytest.raises(ShapeError, match="got 2, expected 3"):
        ctx.require_feature_width(expected_features=3)


def test_broadcast_feature_width_from_singleton():
    ctx = SamplingContext(
        channel_index=torch.tensor([[1], [2]], dtype=torch.long),
        mask=torch.tensor([[True], [False]], dtype=torch.bool),
    )
    ctx.broadcast_feature_width(target_features=4)
    assert ctx.channel_index.shape == (2, 4)
    assert ctx.mask.shape == (2, 4)
    assert torch.equal(ctx.channel_index[0], torch.tensor([1, 1, 1, 1]))


def test_broadcast_feature_width_is_noop_when_already_matching():
    channel_index = torch.tensor([[1, 2, 3]], dtype=torch.long)
    mask = torch.tensor([[True, False, True]], dtype=torch.bool)
    ctx = SamplingContext(channel_index=channel_index, mask=mask)
    before_ptr = ctx.channel_index.data_ptr()
    ctx.broadcast_feature_width(target_features=3)
    assert ctx.channel_index.data_ptr() == before_ptr
    assert torch.equal(ctx.mask, mask)


def test_broadcast_feature_width_rejects_non_singleton_context():
    ctx = SamplingContext(
        channel_index=torch.tensor([[1, 2]], dtype=torch.long),
        mask=torch.tensor([[True, True]], dtype=torch.bool),
    )
    with pytest.raises(ShapeError, match="expected 4 or 1"):
        ctx.broadcast_feature_width(target_features=4)


def test_broadcast_feature_width_respects_allow_from_one_flag():
    ctx = SamplingContext(
        channel_index=torch.tensor([[5]], dtype=torch.long),
        mask=torch.tensor([[True]], dtype=torch.bool),
    )
    with pytest.raises(ShapeError, match="expected 3"):
        ctx.broadcast_feature_width(target_features=3, allow_from_one=False)


def test_repeat_split_feature_width_expands_split_sized_context():
    ctx = SamplingContext(
        channel_index=torch.tensor([[3, 7]], dtype=torch.long),
        mask=torch.tensor([[True, False]], dtype=torch.bool),
    )
    ctx.repeat_split_feature_width(
        num_splits=2,
        target_features=4,
    )
    assert torch.equal(ctx.channel_index, torch.tensor([[3, 7, 3, 7]], dtype=torch.long))
    assert torch.equal(ctx.mask, torch.tensor([[True, False, True, False]], dtype=torch.bool))


def test_repeat_split_feature_width_is_noop_when_already_matching():
    channel_index = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    mask = torch.tensor([[True, True, True, False]], dtype=torch.bool)
    ctx = SamplingContext(channel_index=channel_index, mask=mask)
    before = ctx.channel_index.clone()
    ctx.repeat_split_feature_width(num_splits=2, target_features=4)
    assert torch.equal(ctx.channel_index, before)
    assert torch.equal(ctx.mask, mask)


def test_repeat_split_feature_width_rejects_invalid_num_splits():
    ctx = SamplingContext(
        channel_index=torch.tensor([[0, 1]], dtype=torch.long),
        mask=torch.tensor([[True, True]], dtype=torch.bool),
    )
    with pytest.raises(InvalidParameterError, match="num_splits must be >= 1"):
        ctx.repeat_split_feature_width(num_splits=0, target_features=4)


def test_repeat_split_feature_width_rejects_uneven_target():
    ctx = SamplingContext(
        channel_index=torch.tensor([[0, 1]], dtype=torch.long),
        mask=torch.tensor([[True, True]], dtype=torch.bool),
    )
    with pytest.raises(ShapeError, match="not divisible by num_splits"):
        ctx.repeat_split_feature_width(
            num_splits=2,
            target_features=5,
        )


def test_repeat_split_feature_width_rejects_incompatible_current_width():
    ctx = SamplingContext(
        channel_index=torch.tensor([[0, 1, 2]], dtype=torch.long),
        mask=torch.tensor([[True, True, True]], dtype=torch.bool),
    )
    with pytest.raises(ShapeError, match="expected 8 or split width 4"):
        ctx.repeat_split_feature_width(num_splits=2, target_features=8)


def test_scatter_split_groups_to_input_width_scatter_mapping():
    ctx = SamplingContext(
        channel_index=torch.tensor([[4, 9]], dtype=torch.long),
        mask=torch.tensor([[True, False]], dtype=torch.bool),
    )
    ctx.scatter_split_groups_to_input_width(
        index_groups=[[0, 2], [1, 3]],
        input_features=4,
    )
    assert torch.equal(ctx.channel_index, torch.tensor([[4, 4, 9, 9]], dtype=torch.long))
    assert torch.equal(ctx.mask, torch.tensor([[True, True, False, False]], dtype=torch.bool))


def test_scatter_split_groups_to_input_width_is_noop_when_already_full_width():
    channel_index = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    mask = torch.tensor([[True, False, True, True]], dtype=torch.bool)
    ctx = SamplingContext(channel_index=channel_index, mask=mask)
    ctx.scatter_split_groups_to_input_width(index_groups=[[0, 2], [1, 3]], input_features=4)
    assert torch.equal(ctx.channel_index, channel_index)
    assert torch.equal(ctx.mask, mask)


def test_scatter_split_groups_to_input_width_requires_exact_cover():
    ctx = SamplingContext(
        channel_index=torch.tensor([[1, 2]], dtype=torch.long),
        mask=torch.tensor([[True, True]], dtype=torch.bool),
    )
    with pytest.raises(InvalidParameterError, match="cover all input features"):
        ctx.scatter_split_groups_to_input_width(
            index_groups=[[0, 1], [1, 2]],
            input_features=3,
        )


def test_scatter_split_groups_to_input_width_rejects_width_mismatch():
    ctx = SamplingContext(
        channel_index=torch.tensor([[1]], dtype=torch.long),
        mask=torch.tensor([[True]], dtype=torch.bool),
    )
    with pytest.raises(ShapeError, match="expected 4 or common split width"):
        ctx.scatter_split_groups_to_input_width(index_groups=[[0, 1], [2, 3]], input_features=4)


def test_slice_feature_ranges_returns_per_child_views():
    ctx = SamplingContext(
        channel_index=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        mask=torch.tensor([[True, True, False, False]], dtype=torch.bool),
    )
    chunks = ctx.slice_feature_ranges(ranges=[(0, 1), (1, 4)])
    assert len(chunks) == 2
    assert torch.equal(chunks[0][0], torch.tensor([[0]], dtype=torch.long))
    assert torch.equal(chunks[1][0], torch.tensor([[1, 2, 3]], dtype=torch.long))


@pytest.mark.parametrize("ranges", [[(-1, 1)], [(2, 1)], [(0, 5)]])
def test_slice_feature_ranges_rejects_invalid_ranges(ranges):
    ctx = SamplingContext(
        channel_index=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        mask=torch.tensor([[True, True, True, True]], dtype=torch.bool),
    )
    with pytest.raises(InvalidParameterError, match="invalid feature slice range"):
        ctx.slice_feature_ranges(ranges=ranges)


def test_route_channel_offsets_routes_masks_and_indices():
    ctx = SamplingContext(
        channel_index=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        mask=torch.tensor([[True, True, True, True]], dtype=torch.bool),
    )
    routes = ctx.route_channel_offsets(
        child_channel_counts=[1, 3],
    )
    assert len(routes) == 2
    # Child 0 only owns global channel 0.
    assert torch.equal(routes[0][0], torch.tensor([[0, 0, 0, 0]], dtype=torch.long))
    assert torch.equal(routes[0][1], torch.tensor([[True, False, False, False]], dtype=torch.bool))
    # Child 1 owns channels [1, 2, 3] and should see local ids [0, 1, 2].
    assert torch.equal(routes[1][0], torch.tensor([[0, 0, 1, 2]], dtype=torch.long))
    assert torch.equal(routes[1][1], torch.tensor([[False, True, True, True]], dtype=torch.bool))


def test_route_channel_offsets_rejects_non_positive_child_count():
    ctx = SamplingContext(
        channel_index=torch.tensor([[0, 1]], dtype=torch.long),
        mask=torch.tensor([[True, True]], dtype=torch.bool),
    )
    with pytest.raises(InvalidParameterError, match="must all be >= 1"):
        ctx.route_channel_offsets(child_channel_counts=[1, 0])


def test_route_channel_offsets_applies_existing_mask():
    ctx = SamplingContext(
        channel_index=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        mask=torch.tensor([[True, False, True, True]], dtype=torch.bool),
    )
    routes = ctx.route_channel_offsets(child_channel_counts=[2, 2])
    # Child 0 covers channels [0, 1], but second position is disabled by mask.
    assert torch.equal(routes[0][1], torch.tensor([[True, False, False, False]], dtype=torch.bool))
