"""
Created on November 6, 2021

@authors: Philipp Deibert, Bennet Wittelsbach
"""
from typing import Tuple
import numpy as np
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import LeafNode


class Uniform(LeafNode):
    r"""(Univariate) continuous Uniform distribution.

    .. math::

        \text{PDF}(x) = \frac{1}{\text{end} - \text{start}}\mathbf{1}_{[\text{start}, \text{end}]}(x)

    where
        - :math:`x` is the input observation
        - :math:`\mathbf{1}_{[\text{start}, \text{end}]}` is the indicator function for the given interval (evaluating to 0 if x is not in the interval)

    Args:
        scope:
            Scope object specifying the variable scope.
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
        self.set_params(start, end, support_outside)

    def set_params(self, start: float, end: float, support_outside: bool = True) -> None:

        if not start < end:
            raise ValueError(
                f"Lower bound for Uniform distribution must be less than upper bound, but were: {start}, {end}"
            )
        if not (np.isfinite(start) and np.isfinite(end)):
            raise ValueError(f"Lower and upper bound must be finite, but were: {start}, {end}")

        self.start = start
        self.end = end
        self.support_outside = support_outside

    def get_params(self) -> Tuple[float, float, bool]:
        return self.start, self.end, self.support_outside

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
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

        valid = np.ones(scope_data.shape[0], dtype=bool)

        # check for infinite values
        valid &= ~np.isinf(scope_data).sum(axis=-1).astype(bool)

        # check if values are in valid range
        if not self.support_outside:
            valid[valid] &= (
                ((scope_data[valid] >= self.start) & (scope_data[valid] <= self.end))
                .sum(axis=-1)
                .astype(bool)
            )

        return valid