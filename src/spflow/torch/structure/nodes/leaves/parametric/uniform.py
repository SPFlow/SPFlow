"""
Created on November 06, 2021

@authors: Philipp Deibert
"""

import numpy as np
import torch
import torch.distributions as D
from typing import List, Tuple
from .parametric import TorchParametricLeaf
from spflow.base.structure.nodes.leaves.parametric.statistical_types import ParametricType
from spflow.base.structure.nodes.leaves.parametric import Uniform

from multipledispatch import dispatch  # type: ignore


class TorchUniform(TorchParametricLeaf):
    r"""(Univariate) continuous Uniform distribution.

    .. math::

        \text{PDF}(x) = \frac{1}{\text{end} - \text{start}}\mathbf{1}_{[\text{start}, \text{end}]}(x)

    where
        - :math:`x` is the input observation
        - :math:`\mathbf{1}_{[\text{start}, \text{end}]}` is the indicator function for the given interval (evaluating to 0 if x is not in the interval)

    Args:
        scope:
            List of integers specifying the variable scope.
        start:
            Start of the interval.
        end:
            End of interval (must be larger than start).
        support_outside:
            Boolean specifying whether or not values outside of the interval are part of the support (defaults to False).
    """

    ptype = ParametricType.CONTINUOUS

    def __init__(
        self, scope: List[int], start: float, end: float, support_outside: bool = True
    ) -> None:

        if len(scope) != 1:
            raise ValueError(f"Scope size for TorchUniform should be 1, but was: {len(scope)}")

        super(TorchUniform, self).__init__(scope)

        # register interval bounds as torch buffers (should not be changed)
        self.register_buffer("start", torch.empty(size=[]))
        self.register_buffer("end", torch.empty(size=[]))

        # set parameters
        self.set_params(start, end, support_outside)

    def set_params(self, start: float, end: float, support_outside: bool = True) -> None:

        if not start < end:
            raise ValueError(
                f"Lower bound for TorchUniform distribution must be less than upper bound, but were: {start}, {end}"
            )
        if not (np.isfinite(start) and np.isfinite(end)):
            raise ValueError(f"Lower and upper bound must be finite, but were: {start}, {end}")

        # since torch Uniform distribution excludes the upper bound, compute next largest number
        end_next = torch.nextafter(torch.tensor(end), torch.tensor(float("Inf")))  # type: ignore

        self.start.data = torch.tensor(float(start))  # type: ignore
        self.end.data = torch.tensor(float(end_next))  # type: ignore
        self.support_outside = support_outside

        # create Torch distribution with specified parameters
        self.dist = D.Uniform(low=self.start, high=end_next)

    def get_params(self) -> Tuple[float, float, bool]:
        return self.start.cpu().numpy(), self.end.cpu().numpy(), self.support_outside  # type: ignore

    def check_support(self, scope_data: torch.Tensor) -> torch.Tensor:
        r"""Checks if instances are part of the support of the Uniform distribution.

        .. math::

            \text{supp}(\text{Uniform})=\begin{cases} [start,end] & \text{if support\_outside}=\text{false}\\
                                                 (-\infty,\infty) & \text{if support\_outside}=\text{true} \end{cases}
        where
            - :math:`start` is the start of the interval
            - :math:`end` is the end of the interval
            - :math:`\text{support\_outside}` is a truth value indicating whether values outside of the interval are part of the support

        Args:
            scope_data:
                Torch tensor containing possible distribution instances.
        Returns:
            Torch tensor indicating for each possible distribution instance, whether they are part of the support (True) or not (False).
        """

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scope):
            raise ValueError(
                f"Expected scope_data to be of shape (n,{len(self.scope)}), but was: {scope_data.shape}"
            )

        # torch distribution support is an interval, despite representing a distribution over a half-open interval
        # end is adjusted to the next largest number to make sure that desired end is part of the distribution interval
        # may cause issues with the support check; easier to do a manual check instead
        valid = torch.ones(scope_data.shape, dtype=torch.bool)

        # check for infinite values
        valid &= ~scope_data.isinf().sum(dim=-1, keepdim=True).bool()

        # check if values are within valid range
        if not self.support_outside:

            mask = valid.clone()
            valid[mask] &= (scope_data[mask] >= self.start) & (scope_data[mask] < self.end)

        return valid.squeeze(1)


@dispatch(Uniform)  # type: ignore[no-redef]
def toTorch(node: Uniform) -> TorchUniform:
    return TorchUniform(node.scope, node.start, node.end)


@dispatch(TorchUniform)  # type: ignore[no-redef]
def toNodes(torch_node: TorchUniform) -> Uniform:
    return Uniform(torch_node.scope, torch_node.start.cpu().numpy(), torch_node.end.cpu().numpy())  # type: ignore
