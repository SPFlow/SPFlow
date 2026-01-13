"""Range/interval inference utilities for SPFlow.

This module provides a convenience wrapper for interval-based log-likelihood
computation. The main functionality is now integrated into the Module classes
via IntervalEvidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from spflow.meta.data.interval_evidence import IntervalEvidence
from spflow.utils.cache import Cache

if TYPE_CHECKING:
    from spflow.modules.module import Module


def log_likelihood_interval(
    model: Module,
    low: Tensor,
    high: Tensor,
    cache: Cache | None = None,
) -> Tensor:
    """Compute log P(low <= X <= high) through the circuit.

    This is a convenience wrapper that creates IntervalEvidence and calls
    model.log_likelihood().

    Args:
        model: Root module of the circuit.
        low: Lower bounds tensor of shape (batch, features).
            Use NaN to indicate no lower bound (-inf).
        high: Upper bounds tensor of shape (batch, features).
            Use NaN to indicate no upper bound (+inf).
        cache: Optional cache for intermediate results.

    Returns:
        Log-likelihood tensor with shape depending on model output.

    Raises:
        ValueError: If low > high for finite entries, or shapes don't match.
        NotImplementedError: If model contains unsupported module/leaf types.

    Example:
        >>> import torch
        >>> from spflow.modules.leaves.uniform import Uniform
        >>> leaf = Uniform(scope=0, low=torch.tensor([0.0]), high=torch.tensor([1.0]))
        >>> low = torch.tensor([[0.2]])
        >>> high = torch.tensor([[0.8]])
        >>> log_prob = log_likelihood_interval(leaf, low, high)
        >>> torch.exp(log_prob)
        tensor([[[[0.6000]]]])
    """
    evidence = IntervalEvidence(low=low, high=high)
    return model.log_likelihood(evidence, cache=cache)
