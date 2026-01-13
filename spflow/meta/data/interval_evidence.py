"""Interval evidence container for range inference.

Allows passing interval bounds (low, high) to log_likelihood instead of point values.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class IntervalEvidence:
    """Represents interval bounds for range inference queries.

    Use this instead of a point evidence tensor when computing P(low <= X <= high).

    Attributes:
        low: Lower bounds tensor of shape (batch, features). Use NaN for no lower bound.
        high: Upper bounds tensor of shape (batch, features). Use NaN for no upper bound.

    Example:
        >>> import torch
        >>> evidence = IntervalEvidence(
        ...     low=torch.tensor([[0.2, 0.3]]),
        ...     high=torch.tensor([[0.8, 0.7]])
        ... )
        >>> evidence.shape
        torch.Size([1, 2])
    """

    low: Tensor
    high: Tensor

    def __post_init__(self) -> None:
        """Validate interval bounds after initialization."""
        if self.low.shape != self.high.shape:
            raise ValueError(
                f"low and high must have same shape, got {self.low.shape} and {self.high.shape}"
            )

        if self.low.dim() != 2:
            raise ValueError(
                f"Tensors must be 2-dimensional (batch, features), got {self.low.dim()}D"
            )

        # Check low <= high for finite entries
        finite_mask = torch.isfinite(self.low) & torch.isfinite(self.high)
        if (self.low[finite_mask] > self.high[finite_mask]).any():
            raise ValueError("Invalid interval: low > high for some entries")

    @property
    def shape(self) -> torch.Size:
        """Return shape of the evidence tensors."""
        return self.low.shape

    @property
    def device(self) -> torch.device:
        """Return device of the evidence tensors."""
        return self.low.device

    def to(self, device: torch.device) -> "IntervalEvidence":
        """Move evidence to specified device."""
        return IntervalEvidence(low=self.low.to(device), high=self.high.to(device))
