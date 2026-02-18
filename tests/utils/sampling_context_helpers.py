"""Helpers for constructing sampling contexts in tests."""

from __future__ import annotations

import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterError
from spflow.utils.sampling_context import SamplingContext


def make_sampling_context(
    *,
    num_samples: int,
    num_features: int = 1,
    num_channels: int = 1,
    num_repetitions: int = 1,
    device: torch.device | None = None,
    channel_index: Tensor | None = None,
    mask: Tensor | None = None,
    repetition_index: Tensor | None = None,
) -> SamplingContext:
    """Build a non-root sampling context for unit tests."""
    if num_samples < 1:
        raise InvalidParameterError(f"num_samples must be >= 1, got {num_samples}.")
    if num_features < 1:
        raise InvalidParameterError(f"num_features must be >= 1, got {num_features}.")
    if num_channels < 1:
        raise InvalidParameterError(f"num_channels must be >= 1, got {num_channels}.")
    if num_repetitions < 1:
        raise InvalidParameterError(f"num_repetitions must be >= 1, got {num_repetitions}.")

    if device is None:
        device = torch.get_default_device()

    if channel_index is None:
        channel_index = torch.zeros((num_samples, num_features), dtype=torch.long, device=device)
    if mask is None:
        mask = torch.ones((num_samples, num_features), dtype=torch.bool, device=device)
    if repetition_index is None:
        repetition_index = torch.zeros((num_samples,), dtype=torch.long, device=device)

    active_channels = channel_index[mask]
    if active_channels.numel() > 0:
        invalid = (active_channels < 0) | (active_channels >= num_channels)
        if invalid.any():
            raise InvalidParameterError(
                f"channel_index has active entries outside [0, {num_channels - 1}] for test helper context."
            )

    invalid_reps = (repetition_index < 0) | (repetition_index >= num_repetitions)
    if invalid_reps.any():
        raise InvalidParameterError(
            f"repetition_index has entries outside [0, {num_repetitions - 1}] for test helper context."
        )

    return SamplingContext(
        channel_index=channel_index,
        mask=mask,
        repetition_index=repetition_index,
    )
