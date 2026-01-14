"""Tucker decomposition layer for tensorized probabilistic circuits.

Implements a fused sum-product layer using Tucker decomposition for arity-2 products.
"""

from typing import Literal

import torch
from torch import Tensor, nn

from spflow.exp.pic.tensorized.base import TensorizedLayer
from spflow.exp.pic.tensorized.utils import eval_tucker


class TuckerLayer(TensorizedLayer):
    """Tucker decomposition layer for fused sum-product computation.

    Computes the output O[f,o,b] = log(sum_{i,j} W[f,i,j,o] * exp(L[f,i,b]) * exp(R[f,j,b]))
    where L and R are the left and right child log-probabilities.

    This avoids materializing the K² outer product by using a single einsum operation
    wrapped in log_func_exp for numerical stability.

    Attributes:
        _params: Weight tensor of shape (F, I, J, O).
    """

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: Literal[2] = 2,
        num_folds: int = 1,
        fold_mask: None = None,
    ) -> None:
        """Initialize Tucker layer.

        Args:
            num_input_units: Number of input units from each child (I = J = K).
            num_output_units: Number of output units (O).
            arity: Must be 2 (binary products). Defaults to 2.
            num_folds: Number of folds.
            fold_mask: Must be None (Tucker doesn't support masking).

        Raises:
            NotImplementedError: If arity != 2.
            AssertionError: If fold_mask is not None.
        """
        if arity != 2:
            raise NotImplementedError("Tucker layers only implement binary product units.")
        if fold_mask is not None:
            raise ValueError("Input for Tucker layer should not be masked.")

        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=None,
        )

        # Weight tensor: (F, I, J, O)
        self._params = nn.Parameter(
            torch.empty(num_folds, num_input_units, num_input_units, num_output_units)
        )

        self.reset_parameters()

    @property
    def params(self) -> Tensor:
        """Return the weight tensor of shape (F, I, J, O)."""
        return self._params

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass in log-space.

        Args:
            x: Input tensor of shape (F, H=2, K, *B).
               x[:, 0] is left child, x[:, 1] is right child.

        Returns:
            Output tensor of shape (F, O, *B).
        """
        return eval_tucker(x[:, 0], x[:, 1], self._params)
