"""Helpers for constructing sampling contexts in tests."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterError
from spflow.utils.sampling_context import SamplingContext, to_one_hot


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


def patch_simple_as_categorical_one_hot(monkeypatch: Any) -> None:
    """Patch SIMPLE to categorical one-hot sampling for deterministic parity tests."""

    def _simple_as_categorical_one_hot(
        logits: Tensor | None = None,
        log_weights: Tensor | None = None,
        dim: int = -1,
        is_mpe: bool = False,
        hard: bool = True,
        tau: float = 1.0,
    ) -> Tensor:
        del is_mpe
        del hard
        del tau
        x = logits if logits is not None else log_weights
        if x is None:
            raise ValueError("Either logits or log_weights must be provided.")
        dim_norm = dim if dim >= 0 else x.dim() + dim
        x_last = x if dim_norm == x.dim() - 1 else x.movedim(dim_norm, -1)
        sampled = torch.distributions.Categorical(logits=x_last).sample()
        out = to_one_hot(sampled, dim=-1, dim_size=x_last.shape[-1], dtype=x_last.dtype)
        return out if dim_norm == x.dim() - 1 else out.movedim(-1, dim_norm)

    import spflow.utils.sampling_context as sp_sampling_context

    monkeypatch.setattr(sp_sampling_context, "SIMPLE", _simple_as_categorical_one_hot)


def assert_nonzero_finite_grad(tensor: Tensor, name: str = "tensor") -> None:
    """Assert gradients exist, are finite, and are non-zero."""
    if tensor.grad is None:
        raise AssertionError(f"{name}.grad is None")
    if not torch.isfinite(tensor.grad).all():
        raise AssertionError(f"{name}.grad contains non-finite values")
    if not bool((tensor.grad.abs().sum() > 0).item()):
        raise AssertionError(f"{name}.grad is all zeros")


def make_diff_routing_from_logits(
    *,
    num_samples: int,
    num_features: int,
    num_channels: int,
    num_repetitions: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create differentiable routing tensors from learnable logits."""
    if device is None:
        device = torch.get_default_device()
    if dtype is None:
        dtype = torch.get_default_dtype()

    channel_logits = torch.randn(
        (num_samples, num_features, num_channels),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    repetition_logits = torch.randn(
        (num_samples, num_repetitions),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    channel_index = channel_logits.softmax(dim=-1)
    repetition_index = repetition_logits.softmax(dim=-1)
    return channel_logits, repetition_logits, channel_index, repetition_index
