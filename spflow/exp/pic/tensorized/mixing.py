"""Mixing sum layer for tensorized probabilistic circuits.

Implements a pure sum layer (no product) for mixing across children,
typically used for block-diagonal mixing as in Eq. 4 of the PIC paper.
"""

from typing import Optional

import torch
from torch import Tensor, nn

from spflow.exp.pic.tensorized.base import TensorizedLayer
from spflow.exp.pic.tensorized.utils import eval_mixing


class MixingSumLayer(TensorizedLayer):
    """Sum layer for mixing across arity dimension.

    Computes a weighted sum across the H children (arity dimension).
    Typically used for block-diagonal mixing patterns (Eq. 4).

    Unlike sum-product layers, this layer only performs summation
    and does not change the number of units.

    Attributes:
        _params: Weight tensor of shape (F, H, K).
    """

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        num_folds: int = 1,
        fold_mask: Optional[Tensor] = None,
    ) -> None:
        """Initialize Mixing Sum layer.

        Args:
            num_input_units: Number of input units (must equal num_output_units).
            num_output_units: Number of output units (must equal num_input_units).
            arity: Number of children to mix.
            num_folds: Number of folds.
            fold_mask: Optional mask of shape (F, H) for valid folds.

        Raises:
            AssertionError: If num_input_units != num_output_units.
        """
        if num_input_units != num_output_units:
            raise ValueError(
                f"MixingSumLayer requires num_input_units == num_output_units, "
                f"got {num_input_units} and {num_output_units}."
            )

        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
        )

        # Weight tensor: (F, H, K)
        self._params = nn.Parameter(torch.empty(num_folds, arity, num_output_units))

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset parameters to normalized uniform distribution."""
        nn.init.uniform_(self._params, 0.01, 0.99)
        # Normalize across arity dimension
        self._params /= self._params.sum(dim=1, keepdim=True)

    @property
    def params(self) -> Tensor:
        """Return the weight tensor of shape (F, H, K)."""
        return self._params

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass in log-space.

        Args:
            x: Input tensor of shape (F, H, K, *B).

        Returns:
            Output tensor of shape (F, K, *B).
        """
        return eval_mixing(x, self._params, self.fold_mask)
