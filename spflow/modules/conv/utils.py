"""Utility functions for convolutional probabilistic circuit sampling.

Provides helper functions for manipulating SamplingContext objects
during sampling in convolutional layers, handling spatial upsampling
and feature expansion.
"""

from __future__ import annotations

from einops import rearrange, repeat

from spflow.utils.sampling_context import SamplingContext


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
    channel_idx = sampling_ctx.channel_index
    mask = sampling_ctx.mask

    # Reshape to spatial form
    if sampling_ctx.is_differentiable:
        channel_idx = rearrange(
            channel_idx,
            "b (h w) c -> b h w c",
            h=current_height,
            w=current_width,
        )
    else:
        channel_idx = rearrange(
            channel_idx,
            "b (h w) -> b h w",
            h=current_height,
            w=current_width,
        )
    mask = rearrange(
        mask,
        "b (h w) -> b h w",
        h=current_height,
        w=current_width,
    )

    # Upsample by repeating each spatial position along height and width.
    if sampling_ctx.is_differentiable:
        channel_idx = repeat(channel_idx, "b h w c -> b (h sh) (w sw) c", sh=scale_h, sw=scale_w)
    else:
        channel_idx = repeat(channel_idx, "b h w -> b (h sh) (w sw)", sh=scale_h, sw=scale_w)
    mask = repeat(mask, "b h w -> b (h sh) (w sw)", sh=scale_h, sw=scale_w)

    # Flatten back to (batch, features)
    if sampling_ctx.is_differentiable:
        channel_idx = rearrange(channel_idx, "b h w c -> b (h w) c")
    else:
        channel_idx = rearrange(channel_idx, "b h w -> b (h w)")
    mask = rearrange(mask, "b h w -> b (h w)")

    sampling_ctx.update(
        channel_index=channel_idx,
        mask=mask,
    )
