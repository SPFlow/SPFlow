"""
Created on November 6, 2021

@authors: Philipp Deibert, Bennet Wittelsbach
"""
from typing import Tuple, Optional
import numpy as np
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import LeafNode


class Poisson(LeafNode):
    r"""(Univariate) Poisson distribution.

    .. math::

        \text{PMF}(k) = \lambda^k\frac{e^{-\lambda}}{k!}

    where
        - :math:`k` is the number of occurrences
        - :math:`\lambda` is the rate parameter

    Args:
        scope:
            Scope object specifying the variable scope.
        l:
            Rate parameter (:math:`\lambda`), expected value and variance of the Poisson distribution (must be greater than or equal to 0; default 1.0).
    """
    def __init__(self, scope: Scope, l: Optional[float]=1.0) -> None:

        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for Poisson should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for Poisson should be empty, but was {scope.evidence}.")

        super(Poisson, self).__init__(scope=scope)
        self.set_params(l)

    def set_params(self, l: float) -> None:

        if not np.isfinite(l):
            raise ValueError(f"Value of l for Poisson distribution must be finite, but was: {l}")

        if l < 0:
            raise ValueError(
                f"Value of l for Poisson distribution must be non-negative, but was: {l}"
            )

        self.l = float(l)

    def get_params(self) -> Tuple[float]:
        return (self.l,)

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if instances are part of the support of the Poisson distribution.

        .. math::

            \text{supp}(\text{Poisson})=\mathbb{N}\cup\{0\}

        Args:
            scope_data:
                Torch tensor containing possible distribution instances.
        Returns:
            Torch tensor indicating for each possible distribution instance, whether they are part of the support (True) or not (False).
        """

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scopes_out[0].query):
            raise ValueError(
                f"Expected scope_data to be of shape (n,{len(self.scopes_out[0].query)}), but was: {scope_data.shape}"
            )

        valid = np.ones(scope_data.shape[0], dtype=bool)

        # check for infinite values
        valid &= ~np.isinf(scope_data).sum(axis=-1).astype(bool)

        # check if all values are valid integers
        # TODO: runtime warning due to nan values
        valid[valid] &= (np.remainder(scope_data[valid], 1) == 0).sum(axis=-1).astype(bool)

        # check if values are in valid range
        valid[valid] &= (scope_data[valid] >= 0).sum(axis=-1).astype(bool)

        return valid