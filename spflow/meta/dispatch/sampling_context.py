"""Contains the sampling context used in SPFlow

Typical usage example:

    sampling_ctx = SamplingDispatch(instance_ids, output_ids)
"""
from typing import Optional, Union

import torch
from torch import Tensor


def _check_mask_bool(mask):
    if not mask.dtype == torch.bool:
        raise ValueError("Mask must be of type torch.bool.")


class SamplingContext:
    """Class for storing context information during sampling.

    Keeps track of instance indices to sample and which output indices of a module to sample from (relevant for modules with multiple outputs).

    Attributes:
        instance_ids:
            List of integers representing the instances of a data set to sample.
            Required to correctly place sampled values into target data set and take potential evidence into account.
        channel_index:
            List of lists of integers representing the output ids for the corresponding instances to sample from (relevant for multi-output module).
            As a shorthand convention, ``[]`` implies to sample from all outputs for a given instance.
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        device: Optional[torch.device] = None,
        channel_index: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> None:
        """Initializes 'SamplingContext' object.

        Args:
            num_samples:
                Integer specifying the number of samples to initialize.
            device:
                Device to store the tensors on.
            channel_index:
                Tensor containing the channel indices to sample from.
            mask:
                Tensor containing the mask to apply to the samples.
                If None, all samples are considered.
        """

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
        return SamplingContext(channel_index=self.channel_index.clone(), mask=self.mask.clone())

    def __repr__(self) -> str:
        return f"SamplingContext(channel_index.shape={self.channel_index.shape}), mask.shape={self.mask.shape}), num_samples={self.channel_index.shape[0]})"


def init_default_sampling_context(
    sampling_ctx: Optional[SamplingContext], num_samples: Optional[int] = None
) -> SamplingContext:
    """Initializes sampling context, if it is not already initialized.

    Args
        sampling_ctx:
            ``SamplingContext`` object or None.
        num_samples:
            Integer specifying the number of samples.

    Returns:
        Original sampling context if not None or a new initialized sampling context.
    """

    # Ensure, that either sampling_ctx or num_samples is not None
    if sampling_ctx is not None:
        return sampling_ctx
    else:
        return SamplingContext(num_samples=num_samples)
