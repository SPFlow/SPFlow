"""Sampling-context helpers for non-leaf contract tests."""

from __future__ import annotations

import torch

from spflow.utils.sampling_context import SamplingContext, to_one_hot


def make_sampling_context_int(
    *,
    num_samples: int,
    num_features: int,
    num_channels: int,
    num_repetitions: int,
    is_mpe: bool = False,
) -> SamplingContext:
    return SamplingContext(
        channel_index=torch.randint(0, num_channels, (num_samples, num_features)),
        mask=torch.ones((num_samples, num_features), dtype=torch.bool),
        repetition_index=torch.randint(0, num_repetitions, (num_samples,)),
        is_mpe=is_mpe,
        is_differentiable=False,
    )


def make_sampling_context_diff(
    *,
    num_samples: int,
    num_features: int,
    num_channels: int,
    num_repetitions: int,
    is_mpe: bool = False,
    hard: bool = True,
) -> SamplingContext:
    channel_index = torch.randint(0, num_channels, (num_samples, num_features))
    repetition_index = torch.randint(0, num_repetitions, (num_samples,))
    return SamplingContext(
        channel_index=to_one_hot(channel_index, dim=-1, dim_size=num_channels),
        mask=torch.ones((num_samples, num_features), dtype=torch.bool),
        repetition_index=to_one_hot(repetition_index, dim=-1, dim_size=num_repetitions),
        is_mpe=is_mpe,
        is_differentiable=True,
        hard=hard,
    )
