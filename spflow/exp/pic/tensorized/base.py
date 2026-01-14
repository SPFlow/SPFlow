"""Base class for tensorized layers in probabilistic circuits.

Provides the abstract base for fused sum-product layers used in tensorized PC evaluation.
"""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional

import torch
from torch import Tensor, nn


class TensorizedLayer(nn.Module, ABC):
    """Abstract base class for tensorized probabilistic circuit layers.

    Tensorized layers operate on folded tensors with shape (F, H, K, *B) and
    produce outputs of shape (F, K, *B), where:
        - F: number of folds (regions batched together)
        - H: arity (number of children per product unit)
        - K: number of units per fold
        - *B: batch dimensions

    These layers fuse sum and product operations to avoid materializing
    intermediate K² tensors, enabling efficient evaluation of large circuits.

    Attributes:
        num_input_units: Number of input units (K for children).
        num_output_units: Number of output units (K for this layer).
        arity: Number of children per product unit.
        num_folds: Number of folds (batched regions).
        fold_mask: Optional mask of shape (F, H) for valid folds.
    """

    fold_mask: Optional[Tensor]

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        num_folds: int = 1,
        fold_mask: Optional[Tensor] = None,
    ) -> None:
        """Initialize the tensorized layer.

        Args:
            num_input_units: Number of input units (channels from children).
            num_output_units: Number of output units (channels produced).
            arity: Arity of product units (typically 2). Defaults to 2.
            num_folds: Number of folds (batched regions). Defaults to 1.
            fold_mask: Optional mask of shape (F, H) indicating valid folds.
                       Defaults to None (all valid).
        """
        super().__init__()

        if num_input_units <= 0:
            raise ValueError("num_input_units must be positive.")
        if num_output_units <= 0:
            raise ValueError("num_output_units must be positive.")
        if arity <= 0:
            raise ValueError("arity must be positive.")
        if num_folds <= 0:
            raise ValueError("num_folds must be positive.")

        self.num_input_units = num_input_units
        self.num_output_units = num_output_units
        self.arity = arity
        self.num_folds = num_folds
        self.register_buffer("fold_mask", fold_mask)

    @cached_property
    def num_params(self) -> int:
        """Total number of parameters in this layer."""
        return sum(param.numel() for param in self.parameters())

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset parameters to default initialization: U(0.01, 0.99)."""
        for param in self.parameters():
            nn.init.uniform_(param, 0.01, 0.99)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x: Input tensor of shape (F, H, K, *B), where:
                - F: number of folds
                - H: arity
                - K: number of input units
                - *B: batch dimensions

        Returns:
            Output tensor of shape (F, K, *B).
        """
