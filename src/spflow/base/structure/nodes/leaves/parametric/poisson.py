# -*- coding: utf-8 -*-
"""Contains Poisson leaf node for SPFlow in the 'base' backend.
"""
from typing import Tuple, Optional
import numpy as np
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import poisson  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore


class Poisson(LeafNode):
    r"""(Univariate) Poisson distribution leaf node in the 'base' backend.

    Represents a univariate Poisson distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \lambda^k\frac{e^{-\lambda}}{k!}

    where
        - :math:`k` is the number of occurrences
        - :math:`\lambda` is the rate parameter

    Attributes:
        l:
            Floating point value representing the rate parameter (:math:`\lambda`), expected value and variance of the Poisson distribution (must be greater than or equal to 0).
    """
    def __init__(self, scope: Scope, l: float=1.0) -> None:
        r"""Initializes ``Poisson`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            l:
                Floating point value representing the rate parameter (:math:`\lambda`), expected value and variance of the Poisson distribution (must be greater than or equal to 0).
                Defaults to 1.0.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'Poisson' should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for 'Poisson' should be empty, but was {scope.evidence}.")

        super(Poisson, self).__init__(scope=scope)
        self.set_params(l)
    
    @property
    def dist(self) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.
        
        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return poisson(mu=self.l)

    def set_params(self, l: float) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            l:
                Floating point value representing the rate parameter (:math:`\lambda`), expected value and variance of the Poisson distribution (must be greater than or equal to 0).
        """
        if not np.isfinite(l):
            raise ValueError(f"Value of 'l' for 'Poisson' must be finite, but was: {l}")

        if l < 0:
            raise ValueError(
                f"Value of 'l' for 'Poisson' must be non-negative, but was: {l}"
            )

        self.l = float(l)

    def get_params(self) -> Tuple[float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Floating point value representing the rate parameter, expected value and variance.
        """
        return (self.l,)

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Poisson distribution, which is:

        .. math::

            \text{supp}(\text{Poisson})=\mathbb{N}\cup\{0\}

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            scope_data:
                Two-dimensional NumPy array containing sample instances.
                Each row is regarded as a sample.
        Returns:
            Two dimensional NumPy array indicating for each instance, whether they are part of the support (True) or not (False).
        """
        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scopes_out[0].query):
            raise ValueError(
                f"Expected scope_data to be of shape (n,{len(self.scopes_out[0].query)}), but was: {scope_data.shape}"
            )

        valid = np.ones(scope_data.shape, dtype=bool)

        # nan entries (regarded as valid)
        nan_mask = np.isnan(scope_data)

        # check for infinite values
        valid[~nan_mask] &= ~np.isinf(scope_data[~nan_mask])

        # check if all values are valid integers
        # TODO: runtime warning due to nan values
        valid[valid & ~nan_mask] &= (np.remainder(scope_data[valid & ~nan_mask], 1) == 0)

        # check if values are in valid range
        valid[valid & ~nan_mask] &= (scope_data[valid & ~nan_mask] >= 0)

        return valid