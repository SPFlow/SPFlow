"""Utility functions for convolutional probabilistic circuit sampling.

Provides helper functions for manipulating SamplingContext objects
during sampling in convolutional layers, handling spatial upsampling
and feature expansion.
"""

from __future__ import annotations

import torch
from einops import repeat

from spflow.exceptions import ShapeError
from spflow.utils.sampling_context import SamplingContext


def _maybe_resize_selector_features(
    sampling_ctx: SamplingContext,
    *,
    current_features: int,
    target_features: int,
) -> None:
    """Resize differentiable selector tensors along feature axis when present.

    This keeps ``channel_select`` / ``repetition_select`` aligned with
    ``channel_index`` updates performed by context expansion/upsampling helpers.
    """
    for attr in ("channel_select", "repetition_select"):
        selector = getattr(sampling_ctx, attr, None)
        if selector is None:
            continue
        if selector.dim() < 2:
            continue

        if selector.shape[1] == target_features:
            continue

        if selector.shape[1] == 1:
            expanded = repeat(selector, "b 1 ... -> b f ...", f=target_features).contiguous()
            setattr(sampling_ctx, attr, expanded)
            continue

        if selector.shape[1] != current_features:
            raise ShapeError(
                f"{attr} has incompatible feature width {selector.shape[1]}; "
                f"expected 1, {current_features}, or {target_features}."
            )

        if current_features == target_features:
            continue

        if current_features == 1:
            expanded = repeat(selector, "b 1 ... -> b f ...", f=target_features).contiguous()
            setattr(sampling_ctx, attr, expanded)
            continue

        # Non-trivial reshape (e.g. spatial upsampling) is handled in the caller.


def upsample_sampling_context(
    sampling_ctx: SamplingContext,
    current_height: int,
    current_width: int,
    scale_h: int,
    scale_w: int,
) -> None:
    """Upsample sampling context tensors to higher spatial resolution.

    Used when propagating from a smaller spatial layer to a larger one
    (e.g., going from ProdConv output back to its input during sampling).
    Applies repeat_interleave to expand each position.

    Modifies the sampling context in-place.

    Args:
        sampling_ctx: The sampling context to modify.
        current_height: Current spatial height of the context tensors.
        current_width: Current spatial width of the context tensors.
        scale_h: Upsampling factor in height dimension.
        scale_w: Upsampling factor in width dimension.
    """
    batch_size = sampling_ctx.channel_index.shape[0]
    channel_idx = sampling_ctx.channel_index
    mask = sampling_ctx.mask

    # Reshape to spatial form
    channel_idx = channel_idx.view(batch_size, current_height, current_width)
    mask = mask.view(batch_size, current_height, current_width)

    # Upsample by repeating each spatial position along height and width.
    channel_idx = repeat(channel_idx, "b h w -> b (h sh) (w sw)", sh=scale_h, sw=scale_w)
    mask = repeat(mask, "b h w -> b (h sh) (w sw)", sh=scale_h, sw=scale_w)

    # Flatten back to (batch, features)
    new_features = current_height * scale_h * current_width * scale_w
    channel_idx = channel_idx.view(batch_size, new_features)
    mask = mask.view(batch_size, new_features)

    sampling_ctx.update(channel_index=channel_idx, mask=mask)

    # Keep differentiable selector tensors (if available) in sync with feature upsampling.
    for attr in ("channel_select", "repetition_select"):
        selector = getattr(sampling_ctx, attr, None)
        if selector is None or selector.dim() < 2:
            continue
        if selector.shape[1] == new_features:
            continue
        if selector.shape[1] == 1:
            setattr(
                sampling_ctx,
                attr,
                repeat(selector, "b 1 ... -> b f ...", f=new_features).contiguous(),
            )
            continue
        if selector.shape[1] != current_height * current_width:
            _maybe_resize_selector_features(
                sampling_ctx,
                current_features=current_height * current_width,
                target_features=new_features,
            )
            continue

        selector_view = selector.view(batch_size, current_height, current_width, *selector.shape[2:])
        selector_view = repeat(selector_view, "b h w ... -> b (h sh) (w sw) ...", sh=scale_h, sw=scale_w)
        setattr(
            sampling_ctx,
            attr,
            selector_view.view(batch_size, new_features, *selector.shape[2:]).contiguous(),
        )
