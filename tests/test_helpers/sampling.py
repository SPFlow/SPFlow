"""Shared sampling-context constructors for tests."""

from __future__ import annotations

import torch

from spflow.utils.sampling_context import SamplingContext


def make_dense_sampling_context(
    *, n_samples: int, n_features: int, n_channels: int, n_repetitions: int
) -> SamplingContext:
    channel_index = torch.randint(low=0, high=n_channels, size=(n_samples, n_features))
    mask = torch.full((n_samples, n_features), True)
    repetition_index = torch.randint(low=0, high=n_repetitions, size=(n_samples,))
    return SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)
