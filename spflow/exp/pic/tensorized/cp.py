"""CP decomposition layers for tensorized probabilistic circuits.

Implements fused sum-product layers using Canonical Polyadic (CP) decomposition,
supporting collapsed, uncollapsed, and shared parameter variants.
"""

from typing import Optional

import torch
from torch import Tensor, nn

from spflow.exp.pic.tensorized.base import TensorizedLayer
from spflow.exp.pic.tensorized.utils import eval_collapsed_cp, log_func_exp


class CollapsedCPLayer(TensorizedLayer):
    """CP layer with collapsed matrix C.

    Computes a per-child weighted sum followed by product reduction:
        1. For each child h: weighted_x[h] = sum_i W[f,h,i,o] * exp(x[f,h,i,b])
        2. Product across children: prod_h weighted_x[h]
        3. Return log of result

    This is equivalent to CP decomposition where all rank-1 terms are collapsed.

    Attributes:
        _params: Weight tensor of shape (F, H, I, O).
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
        """Initialize Collapsed CP layer.

        Args:
            num_input_units: Number of input units from each child.
            num_output_units: Number of output units.
            arity: Number of children.
            num_folds: Number of folds.
            fold_mask: Optional mask of shape (F, H) for valid folds.
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
        )

        # Weight tensor: (F, H, I, O)
        self._params = nn.Parameter(torch.empty(num_folds, arity, num_input_units, num_output_units))

        self.reset_parameters()

    @property
    def params(self) -> Tensor:
        """Return the weight tensor of shape (F, H, I, O)."""
        return self._params

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass in log-space.

        Args:
            x: Input tensor of shape (F, H, K, *B).

        Returns:
            Output tensor of shape (F, O, *B).
        """
        return eval_collapsed_cp(x, self._params, self.fold_mask)


class UncollapsedCPLayer(TensorizedLayer):
    """CP layer with explicit rank decomposition.

    Uses separate weight matrices for input (W_in) and output (W_out):
        - W_in: (F, H, I, R) maps input to rank-R intermediate
        - W_out: (F, R, O) maps intermediate to output

    Attributes:
        _params_in: Input weight tensor of shape (F, H, I, R).
        _params_out: Output weight tensor of shape (F, R, O).
    """

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        num_folds: int = 1,
        fold_mask: Optional[Tensor] = None,
        rank: int = 1,
    ) -> None:
        """Initialize Uncollapsed CP layer.

        Args:
            num_input_units: Number of input units from each child.
            num_output_units: Number of output units.
            arity: Number of children.
            num_folds: Number of folds.
            fold_mask: Optional mask of shape (F, H) for valid folds.
            rank: Rank of CP decomposition (intermediate dimension).
        """
        if rank <= 0:
            raise ValueError("rank must be positive.")

        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
        )

        self.rank = rank

        # Weight tensors
        self._params_in = nn.Parameter(torch.empty(num_folds, arity, num_input_units, rank))
        self._params_out = nn.Parameter(torch.empty(num_folds, rank, num_output_units))

        self.reset_parameters()

    @property
    def params_in(self) -> Tensor:
        """Return the input weight tensor of shape (F, H, I, R)."""
        return self._params_in

    @property
    def params_out(self) -> Tensor:
        """Return the output weight tensor of shape (F, R, O)."""
        return self._params_out

    def _forward_in_linear(self, x: Tensor) -> Tensor:
        """Apply per-child weighted sum to rank dimension."""
        # einsum: "fhir,fhi...->fhr..."
        return torch.einsum("fhir,fhi...->fhr...", self._params_in, x)

    def _forward_reduce_log(self, x: Tensor) -> Tensor:
        """Reduce across arity dimension with optional masking."""
        if self.fold_mask is not None:
            mask_shape = self.fold_mask.shape + (1,) * (x.ndim - self.fold_mask.ndim)
            x = x * self.fold_mask.view(mask_shape)
        return x.sum(dim=1)

    def _forward_out_linear(self, x: Tensor) -> Tensor:
        """Apply output weight matrix."""
        # einsum: "fro,fr...->fo..."
        return torch.einsum("fro,fr...->fo...", self._params_out, x)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass in log-space.

        Args:
            x: Input tensor of shape (F, H, K, *B).

        Returns:
            Output tensor of shape (F, O, *B).
        """
        # Apply input weights
        x = log_func_exp(x, func=self._forward_in_linear, dim=2, keepdim=True)
        # (F, H, R, *B)

        # Reduce across children
        x = self._forward_reduce_log(x)
        # (F, R, *B)

        # Apply output weights
        x = log_func_exp(x, func=self._forward_out_linear, dim=1, keepdim=True)
        # (F, O, *B)

        return x


class SharedCPLayer(TensorizedLayer):
    """CP layer with parameters shared across folds.

    Uses a single weight matrix shared across all folds, reducing parameters.
    The weight tensor has shape (H, I, O) instead of (F, H, I, O).

    Attributes:
        _params: Shared weight tensor of shape (H, I, O).
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
        """Initialize Shared CP layer.

        Args:
            num_input_units: Number of input units from each child.
            num_output_units: Number of output units.
            arity: Number of children.
            num_folds: Number of folds (used for shape compatibility).
            fold_mask: Ignored for SharedCPLayer.
        """
        # SharedCPLayer ignores fold_mask
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=None,
        )

        # Shared weight tensor: (H, I, O)
        self._params = nn.Parameter(torch.empty(arity, num_input_units, num_output_units))

        self.reset_parameters()

    @property
    def params(self) -> Tensor:
        """Return the shared weight tensor of shape (H, I, O)."""
        return self._params

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass in log-space.

        Args:
            x: Input tensor of shape (F, H, K, *B).

        Returns:
            Output tensor of shape (F, O, *B).
        """

        def _forward_in_linear(x_prob: Tensor) -> Tensor:
            """Apply shared weighted sum."""
            # einsum: "hio,fhi...->fho..."
            return torch.einsum("hio,fhi...->fho...", self._params, x_prob)

        # Apply shared weighted sum
        x = log_func_exp(x, func=_forward_in_linear, dim=2, keepdim=True)
        # (F, H, O, *B)

        # Reduce across children
        x = x.sum(dim=1)
        # (F, O, *B)

        return x


def CPLayer(
    *,
    num_input_units: int,
    num_output_units: int,
    arity: int = 2,
    num_folds: int = 1,
    fold_mask: Optional[Tensor] = None,
    rank: int = 1,
    collapsed: bool = True,
    shared: bool = False,
) -> TensorizedLayer:
    """Factory function to create CP layer variants.

    Args:
        num_input_units: Number of input units from each child.
        num_output_units: Number of output units.
        arity: Number of children.
        num_folds: Number of folds.
        fold_mask: Optional mask of shape (F, H) for valid folds.
        rank: Rank for uncollapsed CP (ignored for collapsed/shared).
        collapsed: If True, use CollapsedCPLayer. Defaults to True.
        shared: If True, use SharedCPLayer (parameters shared across folds).

    Returns:
        Appropriate CP layer variant.

    Raises:
        NotImplementedError: For shared + uncollapsed combination.
    """
    if shared and collapsed:
        return SharedCPLayer(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=None,
        )

    if shared and not collapsed:
        raise NotImplementedError("Shared uncollapsed CP is not implemented.")

    if collapsed:
        return CollapsedCPLayer(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
        )

    return UncollapsedCPLayer(
        num_input_units=num_input_units,
        num_output_units=num_output_units,
        arity=arity,
        num_folds=num_folds,
        fold_mask=fold_mask,
        rank=rank,
    )
