"""Differentiable categorical sampling utilities.

This module centralizes the categorical routing primitives used by
``_rsample(...)`` paths.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from spflow.exceptions import InvalidParameterError

DiffSampleMethod = Literal["simple", "gumbel"]


def sample_gumbel(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = 1e-20,
) -> Tensor:
    """Sample Gumbel(0, 1) noise."""
    uniform = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(uniform + eps) + eps)


def _simple_st(
    logits: Tensor,
    *,
    dim: int,
    is_mpe: bool,
    temperature: float,
) -> Tensor:
    """Straight-through SIMPLE estimator."""
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=dim)
    perturbed = scaled_logits
    if not is_mpe:
        perturbed = scaled_logits + sample_gumbel(
            tuple(scaled_logits.shape),
            device=scaled_logits.device,
            dtype=scaled_logits.dtype,
        )
    indices = perturbed.argmax(dim=dim, keepdim=True)
    hard = torch.zeros_like(probs)
    hard.scatter_(dim, indices, 1.0)
    return (hard - probs).detach() + probs


def _gumbel(
    logits: Tensor,
    *,
    dim: int,
    is_mpe: bool,
    hard: bool,
    temperature: float,
) -> Tensor:
    """Gumbel-Softmax sampling with differentiable MPE fallback."""
    scaled_logits = logits / temperature
    if is_mpe:
        probs = F.softmax(scaled_logits, dim=dim)
        indices = probs.argmax(dim=dim, keepdim=True)
        hard_assign = torch.zeros_like(probs)
        hard_assign.scatter_(dim, indices, 1.0)
        return (hard_assign - probs).detach() + probs
    return F.gumbel_softmax(logits=scaled_logits, tau=1.0, hard=hard, dim=dim)


def sample_categorical_differentiably(
    *,
    logits: Tensor,
    dim: int,
    method: DiffSampleMethod,
    is_mpe: bool,
    hard: bool,
    temperature: float,
) -> Tensor:
    """Sample differentiable categorical assignments along ``dim``."""
    if temperature <= 0.0:
        raise InvalidParameterError(f"temperature must be > 0, got {temperature}.")
    if method == "simple":
        return _simple_st(logits=logits, dim=dim, is_mpe=is_mpe, temperature=temperature)
    if method == "gumbel":
        return _gumbel(logits=logits, dim=dim, is_mpe=is_mpe, hard=hard, temperature=temperature)
    raise InvalidParameterError(f"Unknown differentiable sampling method: {method!r}.")
