"""
Created on November 06, 2021

@authors: Philipp Deibert
"""
import numpy as np
import torch
import torch.distributions as D
from typing import List, Tuple, Optional
from spflow.meta.scope.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.nodes.node import LeafNode
from spflow.base.structure.nodes.leaves.parametric.uniform import Uniform as BaseUniform


class Uniform(LeafNode):
    r"""(Univariate) continuous Uniform distribution for Torch backend.

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
    def __init__(
        self, scope: Scope, start: float, end: float, support_outside: bool = True
    ) -> None:

        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for Poisson should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for Poisson should be empty, but was {scope.evidence}.")

        super(Uniform, self).__init__(scope=scope)

        # register interval bounds as torch buffers (should not be changed)
        self.register_buffer("start", torch.empty(size=[]))
        self.register_buffer("end", torch.empty(size=[]))
        self.register_buffer("end_next", torch.empty(size=[]))

        # set parameters
        self.set_params(start, end, support_outside)

    def set_params(self, start: float, end: float, support_outside: bool = True) -> None:

        if not start < end:
            raise ValueError(
                f"Lower bound for Uniform distribution must be less than upper bound, but were: {start}, {end}"
            )
        if not (np.isfinite(start) and np.isfinite(end)):
            raise ValueError(f"Lower and upper bound must be finite, but were: {start}, {end}")

        # since torch Uniform distribution excludes the upper bound, compute next largest number
        end_next = torch.nextafter(torch.tensor(end), torch.tensor(float("Inf")))  # type: ignore

        self.start.data = torch.tensor(float(start))  # type: ignore
        self.end.data = torch.tensor(float(end))  # type: ignore
        self.end_next.data = torch.tensor(float(end_next))
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

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scope.query):
            raise ValueError(
                f"Expected scope_data to be of shape (n,{len(self.scope.query)}), but was: {scope_data.shape}"
            )
        
        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        # torch distribution support is an interval, despite representing a distribution over a half-open interval
        # end is adjusted to the next largest number to make sure that desired end is part of the distribution interval
        # may cause issues with the support check; easier to do a manual check instead
        valid = torch.ones(scope_data.shape[0], 1, dtype=torch.bool)

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf().squeeze(-1)

        # check if values are within valid range
        if not self.support_outside:
            valid[~nan_mask & valid] &= ((scope_data[~nan_mask & valid] >= self.start) & (scope_data[~nan_mask & valid] < self.end_next)).squeeze(-1)

        return valid


@dispatch(memoize=True)
def toTorch(node: BaseUniform, dispatch_ctx: Optional[DispatchContext]=None) -> Uniform:
    return Uniform(node.scope, node.start, node.end)


@dispatch(memoize=True)
def toBase(torch_node: Uniform, dispatch_ctx: Optional[DispatchContext]=None) -> BaseUniform:
    return BaseUniform(torch_node.scope, torch_node.start.cpu().numpy(), torch_node.end.cpu().numpy())