"""Utility functions for convolutional probabilistic circuit sampling.

Provides helper functions for manipulating SamplingContext objects
during sampling in convolutional layers, handling spatial upsampling
and feature expansion.
"""

from __future__ import annotations

from einops import repeat

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

    sampling_ctx.update(
        channel_index=channel_idx,
        mask=mask,
    )
