"""
Created on November 6, 2021

@authors: Philipp Deibert, Bennet Wittelsbach
"""
from typing import Tuple, Optional
import numpy as np
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import nbinom
from scipy.stats.distributions import rv_frozen


class NegativeBinomial(LeafNode):
    r"""(Univariate) Negative Binomial distribution.

    .. math::

        \text{PMF}(k) = \binom{k+n-1}{n-1}p^n(1-p)^k

    where
        - :math:`k` is the number of failures
        - :math:`n` is the maximum number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Args:
        scope:
            Scope object specifying the variable scope.
        n:
            Number of i.i.d. successes (greater or equal to 0).
        p:
            Probability of success for each trial in the range :math:`(0,1]` (default 0.5).
    """
    def __init__(self, scope: Scope, n: int, p: Optional[float]=0.5) -> None:

        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for NegativeBinomial should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for NegativeBinomial should be empty, but was {scope.evidence}.")

        super(NegativeBinomial, self).__init__(scope=scope)
        self.set_params(n, p)

    @property
    def dist(self) -> rv_frozen:
        return nbinom(n=self.n, p=self.p)

    def set_params(self, n: int, p: float) -> None:

        if p <= 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for NegativeBinomial distribution must to be between 0.0 and 1.0, but was: {p}"
            )
        if n < 0 or not np.isfinite(n):
            raise ValueError(
                f"Value of n for NegativeBinomial distribution must to greater of equal to 0, but was: {n}"
            )

        if not (np.remainder(n, 1.0) == 0.0):
            raise ValueError(
                f"Value of n for NegativeBinomial distribution must be (equal to) an integer value, but was: {n}"
            )

        self.n = n
        self.p = p

    def get_params(self) -> Tuple[int, float]:
        return self.n, self.p

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if instances are part of the support of the NegativeBinomial distribution.

        .. math::

            \text{supp}(\text{NegativeBinomial})=\mathbb{N}\cup\{0\}

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

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