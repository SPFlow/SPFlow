"""Image sampling context utilities for convolutional probabilistic circuits.

This module provides a 2D sampling context specifically designed for image data,
tracking spatial dimensions (height, width) instead of flattened features.

The ImageSamplingContext is essential for:
- Proper spatial propagation in convolutional layers
- Handling upsampling/downsampling of channel indices
- Maintaining 2D structure during conditional sampling
"""
from __future__ import annotations

import torch
from torch import Tensor


class ImageSamplingContext:
    """2D sampling context for image-based probabilistic circuits.

    Manages sampling state with explicit height and width dimensions,
    rather than flattened features. This allows proper spatial handling
    in convolutional layers.

    Attributes:
        num_samples (int): Number of samples (batch size).
        height (int): Current spatial height.
        width (int): Current spatial width.
        channel_index (Tensor): Channel indices (batch, height, width).
        mask (Tensor): Boolean mask (batch, height, width).
        device (torch.device): Device for tensors.
        repetition_idx (Tensor | None): Optional repetition indices.
    """

    def __init__(
        self,
        num_samples: int,
        height: int,
        width: int,
        channel_index: Tensor | None = None,
        mask: Tensor | None = None,
        device: torch.device | None = None,
        repetition_idx: Tensor | None = None,
    ) -> None:
        """Initialize ImageSamplingContext.

        Args:
            num_samples: Batch size.
            height: Spatial height.
            width: Spatial width.
            channel_index: Optional pre-defined channel indices (batch, H, W).
            mask: Optional pre-defined mask (batch, H, W).
            device: Device for tensor storage.
            repetition_idx: Optional repetition indices.
        """
        if device is None:
            if hasattr(torch, "get_default_device"):
                device = torch.get_default_device()
            else:
                device = torch.device("cpu")

        self.num_samples = num_samples
        self.height = height
        self.width = width
        self.device = device
        self.repetition_idx = repetition_idx

        if channel_index is not None and mask is not None:
            if channel_index.shape != mask.shape:
                raise ValueError("channel_index and mask must have the same shape.")
            if not mask.dtype == torch.bool:
                raise ValueError("mask must be torch.bool dtype.")
            self._channel_index = channel_index
            self._mask = mask
        elif channel_index is None and mask is None:
            # Default: zeros for channel_index, True for mask
            self._channel_index = torch.zeros(
                (num_samples, height, width), dtype=torch.long, device=device
            )
            self._mask = torch.ones(
                (num_samples, height, width), dtype=torch.bool, device=device
            )
        else:
            raise ValueError("channel_index and mask must both be None or both be provided.")

    @property
    def channel_index(self) -> Tensor:
        """Channel indices tensor (batch, height, width)."""
        return self._channel_index

    @channel_index.setter
    def channel_index(self, value: Tensor) -> None:
        """Set channel indices, validating shape."""
        if value.shape != self._mask.shape:
            raise ValueError(
                f"channel_index shape {value.shape} must match mask shape {self._mask.shape}"
            )
        self._channel_index = value

    @property
    def mask(self) -> Tensor:
        """Boolean mask tensor (batch, height, width)."""
        return self._mask

    @mask.setter
    def mask(self, value: Tensor) -> None:
        """Set mask, validating dtype and shape."""
        if value.shape != self._channel_index.shape:
            raise ValueError(
                f"mask shape {value.shape} must match channel_index shape {self._channel_index.shape}"
            )
        if value.dtype != torch.bool:
            raise ValueError("mask must be torch.bool dtype.")
        self._mask = value

    @property
    def channel_index_flat(self) -> Tensor:
        """Flattened channel indices (batch, height * width)."""
        return self._channel_index.view(self.num_samples, -1)

    @property
    def mask_flat(self) -> Tensor:
        """Flattened mask (batch, height * width)."""
        return self._mask.view(self.num_samples, -1)

    def update(
        self,
        channel_index: Tensor,
        mask: Tensor,
        height: int | None = None,
        width: int | None = None,
    ) -> None:
        """Update context with new tensors and optionally new dimensions.

        Args:
            channel_index: New channel indices (batch, H, W).
            mask: New boolean mask (batch, H, W).
            height: New height (inferred from tensor if None).
            width: New width (inferred from tensor if None).
        """
        if channel_index.shape != mask.shape:
            raise ValueError("channel_index and mask must have the same shape.")
        if mask.dtype != torch.bool:
            raise ValueError("mask must be torch.bool dtype.")

        self._channel_index = channel_index
        self._mask = mask

        # Update dimensions
        if height is not None:
            self.height = height
        else:
            self.height = channel_index.shape[1]

        if width is not None:
            self.width = width
        else:
            self.width = channel_index.shape[2]

    def upsample(self, scale_h: int, scale_w: int) -> ImageSamplingContext:
        """Create upsampled context for going to higher resolution.

        Used when propagating from a smaller spatial layer to a larger one
        (e.g., going from ProdConv output back to its input).

        Args:
            scale_h: Upsampling factor in height.
            scale_w: Upsampling factor in width.

        Returns:
            New ImageSamplingContext with upsampled dimensions.
        """
        new_height = self.height * scale_h
        new_width = self.width * scale_w

        # Repeat each position scale times
        upsampled_idx = self._channel_index.repeat_interleave(scale_h, dim=1)
        upsampled_idx = upsampled_idx.repeat_interleave(scale_w, dim=2)

        upsampled_mask = self._mask.repeat_interleave(scale_h, dim=1)
        upsampled_mask = upsampled_mask.repeat_interleave(scale_w, dim=2)

        return ImageSamplingContext(
            num_samples=self.num_samples,
            height=new_height,
            width=new_width,
            channel_index=upsampled_idx,
            mask=upsampled_mask,
            device=self.device,
            repetition_idx=self.repetition_idx,
        )

    def downsample(self, scale_h: int, scale_w: int) -> ImageSamplingContext:
        """Create downsampled context for going to lower resolution.

        Used when propagating from a larger spatial layer to a smaller one.
        Takes the top-left value of each block.

        Args:
            scale_h: Downsampling factor in height.
            scale_w: Downsampling factor in width.

        Returns:
            New ImageSamplingContext with downsampled dimensions.
        """
        new_height = self.height // scale_h
        new_width = self.width // scale_w

        # Take every scale-th position (strided sampling)
        downsampled_idx = self._channel_index[:, ::scale_h, ::scale_w]
        downsampled_mask = self._mask[:, ::scale_h, ::scale_w]

        return ImageSamplingContext(
            num_samples=self.num_samples,
            height=new_height,
            width=new_width,
            channel_index=downsampled_idx.contiguous(),
            mask=downsampled_mask.contiguous(),
            device=self.device,
            repetition_idx=self.repetition_idx,
        )

    def copy(self) -> ImageSamplingContext:
        """Return a deep copy of this context."""
        return ImageSamplingContext(
            num_samples=self.num_samples,
            height=self.height,
            width=self.width,
            channel_index=self._channel_index.clone(),
            mask=self._mask.clone(),
            device=self.device,
            repetition_idx=self.repetition_idx.clone() if self.repetition_idx is not None else None,
        )

    def __repr__(self) -> str:
        return (
            f"ImageSamplingContext(num_samples={self.num_samples}, "
            f"height={self.height}, width={self.width}, "
            f"channel_index.shape={self._channel_index.shape})"
        )
