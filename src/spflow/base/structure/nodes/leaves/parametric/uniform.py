# -*- coding: utf-8 -*-
"""Contains Uniform leaf node for SPFlow in the ``base`` backend.
"""
from typing import Tuple
import numpy as np
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import uniform  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore


class Uniform(LeafNode):
    r"""(Univariate) continuous Uniform distribution leaf node in the ``base`` backend.

    Represents an univariate Uniform distribution, with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{\text{end} - \text{start}}\mathbf{1}_{[\text{start}, \text{end}]}(x)

    where
        - :math:`x` is the input observation
        - :math:`\mathbf{1}_{[\text{start}, \text{end}]}` is the indicator function for the given interval (evaluating to 0 if x is not in the interval)

    Attributes:
        start:
            Floating point value representing the start of the interval (including).
        end:
            Floating point value representing the end of the interval (including). Must be larger than 'start'.
        support_outside:
            Boolean indicating whether or not values outside of the interval are part of the support.
    """

    def __init__(
        self,
        scope: Scope,
        start: float,
        end: float,
        support_outside: bool = True,
    ) -> None:
        r"""Initializes ``Uniform`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            start:
                Floating point value representing the start of the interval (including).
            end:
                Floating point value representing the end of the interval (including). Must be larger than 'start'.
            support_outside:
                Boolean indicating whether or not values outside of the interval are part of the support.
                Defaults to True.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'Uniform' should be 1, but was: {len(scope.query)}."
            )
        if len(scope.evidence):
            raise ValueError(
                f"Evidence scope for 'Uniform' should be empty, but was {scope.evidence}."
            )

        super(Uniform, self).__init__(scope=scope)
        self.set_params(start, end, support_outside)

    @property
    def dist(self) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return uniform(loc=self.start, scale=self.end - self.start)

    def set_params(
        self, start: float, end: float, support_outside: bool = True
    ) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            start:
                Floating point value representing the start of the interval (including).
            end:
                Floating point value representing the end of the interval (including). Must be larger than ``start``.
            support_outside:
                Boolean indicating whether or not values outside of the interval are part of the support.
                Defaults to True.
        """
        if not start < end:
            raise ValueError(
                f"Value of 'start' for 'Uniform' must be less than value of 'end', but were: {start}, {end}."
            )
        if not (np.isfinite(start) and np.isfinite(end)):
            raise ValueError(
                f"Values of 'start' and 'end' for 'Uniform' must be finite, but were: {start}, {end}."
            )

        self.start = start
        self.end = end
        self.support_outside = support_outside

    def get_params(self) -> Tuple[float, float, bool]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of the floating point values representing the start and end of the interval and the boolean indicating whether or not values outside of the interval are part of the support.
        """
        return self.start, self.end, self.support_outside

    def check_support(self, data: np.ndarray, is_scope_data: bool=False) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Uniform distribution, which is:

        .. math::

            \text{supp}(\text{Uniform})=\begin{cases} [start,end] & \text{if support\_outside}=\text{false}\\
                                                 (-\infty,\infty) & \text{if support\_outside}=\text{true} \end{cases}
        where
            - :math:`start` is the start of the interval
            - :math:`end` is the end of the interval
            - :math:`\text{support\_outside}` is a truth value indicating whether values outside of the interval are part of the support

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional NumPy array containing sample instances.
                Each row is regarded as a sample.
                Unless ``is_scope_data`` is set to True, it is assumed that the relevant data is located in the columns corresponding to the scope indices.
            is_scope_data:
                Boolean indicating if the given data already contains the relevant data for the leaf's scope in the correct order (True) or if it needs to be extracted from the full data set.
                Defaults to False.

        Returns:
            Two-dimensional NumPy array indicating for each instance, whether they are part of the support (True) or not (False).
        """
        if is_scope_data:
            scope_data = data
        else:
            # select relevant data for scope
            scope_data = data[:, self.scope.query]

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scope):
            raise ValueError(
                f"Expected 'scope_data' to be of shape (n,{len(self.scope)}), but was: {scope_data.shape}"
            )

        valid = np.ones(scope_data.shape, dtype=bool)

        # nan entries (regarded as valid)
        nan_mask = np.isnan(scope_data)

        # check for infinite values
        valid[~nan_mask] &= ~np.isinf(scope_data[~nan_mask])

        # check if values are in valid range
        if not self.support_outside:
            valid[valid & ~nan_mask] &= (
                scope_data[valid & ~nan_mask] >= self.start
            ) & (scope_data[valid & ~nan_mask] <= self.end)

        return valid
