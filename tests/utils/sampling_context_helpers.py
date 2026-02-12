"""Test helpers for constructing explicit sampling contexts."""

from __future__ import annotations

import torch

from spflow.modules.module import Module
from spflow.utils.sampling_context import SamplingContext


def make_sampling_context(
    *,
    batch_size: int,
    num_features: int,
    device: torch.device | None = None,
    channel_value: int = 0,
    mask_value: bool = True,
    repetition_idx: torch.Tensor | None = None,
) -> SamplingContext:
    """Create a concrete sampling context for tests."""
    channel_index = torch.full((batch_size, num_features), channel_value, dtype=torch.long, device=device)
    mask = torch.full((batch_size, num_features), mask_value, dtype=torch.bool, device=device)
    return SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_idx)


def make_module_sampling_context(
    module: Module,
    *,
    batch_size: int,
    num_features: int | None = None,
    repetition_idx: torch.Tensor | None = None,
) -> SamplingContext:
    """Create a sampling context aligned to a module's output feature width."""
    features = module.out_shape.features if num_features is None else num_features
    return make_sampling_context(
        batch_size=batch_size,
        num_features=features,
        device=module.device,
        repetition_idx=repetition_idx,
    )
