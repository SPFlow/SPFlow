# -*- coding: utf-8 -*-
"""Contains Binomial leaf node for SPFlow in the 'base' backend.
"""
from typing import Tuple, Optional
import numpy as np
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import binom  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore


class Binomial(LeafNode):
    r"""(Univariate) Binomial distribution leaf node in the 'base' backend.

    Represents an univariate Binomial distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \binom{n}{k}p^k(1-p)^{n-k}

    where
        - :math:`p` is the success probability of each trial in :math:`[0,1]`
        - :math:`n` is the number of total trials
        - :math:`k` is the number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Attributes:
        n:
            Integer representing the number of i.i.d. Bernoulli trials (greater or equal to 0).
        p:
            Floating point value representing the success probability of each trial between zero and one.
    """
    def __init__(self, scope: Scope, n: int, p: float=0.5) -> None:
        r"""Initializes ``Binomial`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            n:
                Integer representing the number of i.i.d. Bernoulli trials (greater or equal to 0).
            p:
                Floating point value representing the success probability of each trial between zero and one.
                Defaults to 0.5.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'Binomial' should be 1, but was {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for 'Binomial' should be empty, but was {scope.evidence}.")

        super(Binomial, self).__init__(scope=scope)
        self.set_params(n, p)
    
    @property
    def dist(self) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.
        
        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return binom(n=self.n, p=self.p)

    def set_params(self, n: int, p: float) -> None:
        """Sets the parameters for the represented distribution.

        Args:
            n:
                Integer representing the number of i.i.d. Bernoulli trials (greater or equal to 0).
            p:
                Floating point value representing the success probability of the Binomial distribution between zero and one.
        """
        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of 'p' for 'Binomial' must to be between 0.0 and 1.0, but was: {p}"
            )

        if n < 0 or not np.isfinite(n):
            raise ValueError(
                f"Value of 'n' for 'Binomial' must to greater of equal to 0, but was: {n}"
            )

        if not (np.remainder(n, 1.0) == 0.0):
            raise ValueError(
                f"Value of 'n' for 'Binomial' must be (equal to) an integer value, but was: {n}"
            )

        self.n = n
        self.p = p

    def get_params(self) -> Tuple[int, float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of the number of i.i.d. Bernoulli trials and the floating point value representing the success probability.
        """
        return self.n, self.p

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Binomial distribution, which is:

        .. math::

            \text{supp}(\text{Binomial})=\{0,\hdots,n\}
        
        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            scope_data:
                Two-dimensional NumPy array containing sample instances.
                Each row is regarded as a sample.
        Returns:
            Two dimensional NumPy array indicating for each instance, whether they are part of the support (True) or not (False).
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
        valid[valid & ~nan_mask] &= np.remainder(scope_data[valid & ~nan_mask], 1) == 0

        # check if values are in valid range
        valid[valid & ~nan_mask] &= (scope_data[valid & ~nan_mask] >= 0) & (scope_data[valid & ~nan_mask] <= self.n)

        return valid