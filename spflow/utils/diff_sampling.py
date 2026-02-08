from __future__ import annotations

from enum import Enum
from operator import xor

import torch
from torch import Tensor
from torch.nn import functional as F


class DiffSampleMethod(str, Enum):
    """Differentiable sampling estimators for categorical choices."""

    SIMPLE = "simple"
    GUMBEL = "gumbel"


def sample_gumbel(shape, eps: float = 1e-20, device: str | torch.device = "cpu") -> Tensor:
    """Sample Gumbel(0, 1) noise."""
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)


def simple_st_one_hot(
    *,
    logits: Tensor | None = None,
    log_weights: Tensor | None = None,
    dim: int = -1,
    is_mpe: bool = False,
) -> Tensor:
    """SIMPLE straight-through categorical sample (hard forward, soft backward)."""
    assert xor(logits is not None, log_weights is not None), "Either logits or log_weights must be given."

    if logits is not None:
        y_soft = F.softmax(logits, dim=dim)
        base = logits
    else:
        y_soft = log_weights.exp()
        base = log_weights

    if not is_mpe:
        base = base + sample_gumbel(base.size(), device=base.device).to(base.dtype)

    index = base.argmax(dim=dim, keepdim=True)
    y_hard = torch.zeros_like(y_soft)
    y_hard.scatter_(dim, index, 1.0)
    return (y_hard - y_soft).detach() + y_soft


def sample_categorical_differentiably(
    *,
    dim: int,
    is_mpe: bool,
    hard: bool = True,
    tau: float = 1.0,
    logits: Tensor | None = None,
    log_weights: Tensor | None = None,
    method: DiffSampleMethod = DiffSampleMethod.SIMPLE,
) -> Tensor:
    """Differentiable categorical sample returning one-hot-like selectors."""
    assert xor(logits is not None, log_weights is not None), "Either logits or log_weights must be given."

    if method == DiffSampleMethod.SIMPLE:
        return simple_st_one_hot(logits=logits, log_weights=log_weights, dim=dim, is_mpe=is_mpe)

    # Optional alternative estimator fallback.
    if is_mpe:
        src = logits if logits is not None else log_weights
        y_soft = src.softmax(dim)
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(src).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    src = logits if logits is not None else log_weights
    return F.gumbel_softmax(src, tau=tau, hard=hard, dim=dim)


def select_with_soft_or_hard(
    tensor: Tensor,
    *,
    index: Tensor | None = None,
    selector: Tensor | None = None,
    dim: int = -1,
) -> Tensor:
    """Select values using hard gather or soft weighted selection."""
    if selector is not None:
        if tensor.shape[dim] != selector.shape[dim]:
            raise ValueError(
                f"selector shape mismatch at dim={dim}: tensor has {tensor.shape[dim]}, selector has {selector.shape[dim]}"
            )
        return (tensor * selector).sum(dim=dim)

    if index is None:
        raise ValueError("Either selector or index must be provided.")

    return tensor.gather(dim=dim, index=index).squeeze(dim)
