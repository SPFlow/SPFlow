"""Sampling context utilities for SPFlow probabilistic circuits.

This module provides context management for sampling operations in SPFlow,
including the SamplingContext class that tracks sampling state across
recursive module calls and utility functions for context initialization
and management.

The sampling context is essential for:
- Managing which instances and features to sample
- Handling evidence during sampling (conditional sampling)
- Tracking channel indices for multi-output modules
- Supporting repetition-based sampling structures
"""

import torch
from torch import Tensor


def _check_mask_bool(mask: Tensor) -> None:
    """Check if mask tensor has boolean dtype.

    Args:
        mask: Tensor to check for boolean dtype.

    Raises:
        ValueError: If mask tensor does not have torch.bool dtype.
    """
    if not mask.dtype == torch.bool:
        raise ValueError("Mask must be of type torch.bool.")


class SamplingContext:
    """Context information manager for sampling operations in probabilistic circuits.

    Manages sampling state across recursive module calls, tracking which instances
    to sample, which output channels to use, and any evidence constraints. This
    is essential for conditional sampling, structured sampling, and maintaining
    consistency across complex circuit hierarchies. channel_index determines which output channels to use for sampling,
    mask allows selective sampling of subsets of instances/features, repetition_index supports structured circuit representations,
    and all tensors must have compatible shapes for proper operation.

    Attributes:
        num_samples (int): Number of samples to generate.
        device (torch.device): Device on which tensors are stored.
        channel_index (Tensor | None): Channel indices to sample from for each
            instance and feature. Shape: (batch_size, num_features).
        mask (Tensor | None): Boolean mask indicating which instances/features
            should be sampled. Shape matches channel_index.
        repetition_index (Tensor | None): Indices for repetition-based structures.
    """

    def __init__(
        self,
        num_samples: int | None = None,
        device: torch.device | None = None,
        channel_index: Tensor | None = None,
        mask: Tensor | None = None,
        repetition_index: Tensor | None = None,
    ) -> None:
        """Initialize SamplingContext for managing sampling operations.

        Creates a sampling context with specified parameters. The context manages
        which instances and features to sample, handles evidence constraints, and supports
        structured sampling through repetition indices. Channel_index determines which output
        channels to use for sampling, mask allows selective sampling of subsets of
        instances/features, and repetition_index supports structured circuit representations.
        All tensors must have compatible shapes for proper operation.

        Args:
            num_samples (int | None, optional): Number of samples to generate.
                Can be inferred from other tensor dimensions if not provided.
                Defaults to None.
            device (torch.device | None, optional): PyTorch device for tensor storage.
                If None, uses torch.get_default_device() or CPU. Defaults to None.
            channel_index (Tensor | None, optional): Channel indices for each sample
                and feature. Shape: (batch_size, num_features). If None, samples
                from all channels. Defaults to None.
            mask (Tensor | None, optional): Boolean mask indicating which instances/
                features to sample from. Must be torch.bool type. Shape must match
                channel_index if provided. Defaults to None.
            repetition_index (Tensor | None, optional): Indices for repetition-based
                sampling structures. Used by circuits with repeated computations.
                Defaults to None.

        Raises:
            ValueError: If tensor shapes are incompatible, mask has wrong dtype,
                or num_samples conflicts with tensor dimensions.
        """
        if device is None:
            # device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
            if hasattr(torch, "get_default_device"):
                device = torch.get_default_device()
            else:
                device = torch.device("cpu")

        if channel_index is not None and mask is not None:
            if not channel_index.shape == mask.shape:
                raise ValueError("channel_index and mask must have the same shape.")

            if num_samples is not None and num_samples != channel_index.shape[0]:
                raise ValueError(
                    "num_samples must be equal to the number of samples in channel_index or be ommitted."
                )

        if channel_index is not None and mask is None:
            if num_samples is not None and num_samples != channel_index.shape[0]:
                raise ValueError(
                    "num_samples must be equal to the number of samples in channel_index or be ommitted."
                )
            num_samples = channel_index.shape[0]

        if channel_index is None and mask is not None:
            if num_samples is not None and num_samples != mask.shape[0]:
                raise ValueError("num_samples must be equal to the number of samples in mask or be ommitted.")
            num_samples = mask.shape[0]

        if (channel_index is None) ^ (mask is None):
            # channel_index and mask must be both None or both not None
            raise ValueError("channel_index and mask must be both None or both not None.")
        elif channel_index is not None and mask is not None:
            # channel_index and mask are both not None
            _check_mask_bool(mask)
            self._mask = mask
            self._channel_index = channel_index
            self.device = device
        else:
            # channel_index and mask are both None
            self._mask = torch.full((num_samples, 1), True, dtype=torch.bool, device=device)
            self._channel_index = torch.zeros((num_samples, 1), dtype=torch.long, device=device)
            self.device = self.mask.device

        self.repetition_idx = repetition_index

    def update(self, channel_index: Tensor, mask: Tensor):
        """Updates the sampling context with new channel index and mask.

        Args:
            channel_index:
                Tensor containing the channel indices to sample from.
            mask:
                Tensor containing the mask to apply to the samples.
        """
        if not channel_index.shape == mask.shape:
            raise ValueError("channel_index and mask must have the same shape.")

        _check_mask_bool(mask)

        self._channel_index = channel_index
        self._mask = mask

    @property
    def channel_index(self):
        return self._channel_index

    @channel_index.setter
    def channel_index(self, channel_index):
        if channel_index.shape != self._mask.shape:
            raise ValueError("New channel_index and previous mask must have the same shape.")
        self._channel_index = channel_index

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask.shape[0] != self._channel_index.shape[0]:
            raise ValueError("New mask and previous channel_index must have the same shape.")
        _check_mask_bool(mask)
        self._mask = mask

    @property
    def samples_mask(self):
        return self.mask.sum(1) > 0

    @property
    def channel_index_masked(self):
        return self.channel_index[self.samples_mask]

    def copy(self):
        """Returns a copy of the sampling context."""
        return SamplingContext(
            channel_index=self.channel_index.clone(),
            mask=self.mask.clone(),
            repetition_index=self.repetition_idx.clone() if self.repetition_idx is not None else None,
        )

    def __repr__(self) -> str:
        return f"SamplingContext(channel_index.shape={self.channel_index.shape}), mask.shape={self.mask.shape}), num_samples={self.channel_index.shape[0]})"


def init_default_sampling_context(
    sampling_ctx: SamplingContext | None, num_samples: int | None = None, device: torch.device | None = None
) -> SamplingContext:
    """Initializes sampling context if not already initialized.

    Args:
        sampling_ctx: SamplingContext object or None.
        num_samples: Integer specifying the number of samples.
        device: PyTorch device for tensor storage.

    Returns:
        Original sampling context if not None or a new initialized sampling context.
    """
    # Ensure, that either sampling_ctx or num_samples is not None
    if sampling_ctx is not None:
        return sampling_ctx
    else:
        return SamplingContext(num_samples=num_samples, device=device)
