# -*- coding: utf-8 -*-
"""Contains Hypergeometric leaf node for SPFlow in the ``base`` backend.
"""
from typing import Tuple, List
import numpy as np
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import hypergeom  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore


class Hypergeometric(LeafNode):
    r"""(Univariate) Hypergeometric distribution leaf node in the 'base' backend.

    Represents an univariate Hypergeometric distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \frac{\binom{M}{k}\binom{N-M}{n-k}}{\binom{N}{n}}

    where
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)
        - :math:`N` is the total number of entities
        - :math:`M` is the number of entities with property of interest
        - :math:`n` is the number of draws
        - :math:`k` s the number of observed entities

    Attributes:
        N:
            Integer specifying the total number of entities (in the population), greater or equal to 0.
        M:
            Integer specifying the number of entities with property of interest (in the population), greater or equal to zero and less than or equal to N.
        n:
            Integer specifying the number of draws, greater of equal to zero and less than or equal to N.
    """

    def __init__(self, scope: Scope, N: int, M: int, n: int) -> None:
        r"""Initializes 'Hypergeometric' leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            N:
                Integer specifying the total number of entities (in the population), greater or equal to 0.
            M:
                Integer specifying the number of entities with property of interest (in the population), greater or equal to zero and less than or equal to N.
            n:
                Integer specifying the number of draws, greater of equal to zero and less than or equal to N.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'Hypergeometric' should be 1, but was: {len(scope.query)}."
            )
        if len(scope.evidence):
            raise ValueError(
                f"Evidence scope for 'Hypergeometric' should be empty, but was {scope.evidence}."
            )

        super(Hypergeometric, self).__init__(scope=scope)
        self.set_params(N, M, n)

    @property
    def dist(self) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return hypergeom(M=self.N, n=self.M, N=self.n)

    def set_params(self, N: int, M: int, n: int) -> None:
        """Sets the parameters for the represented distribution.

        Args:
            N:
                Integer specifying the total number of entities (in the population), greater or equal to 0.
            M:
                Integer specifying the number of entities with property of interest (in the population), greater or equal to zero and less than or equal to N.
            n:
                Integer specifying the number of draws, greater of equal to zero and less than or equal to N.
        """
        if N < 0 or not np.isfinite(N):
            raise ValueError(
                f"Value of 'N' for 'Hypergeometric' must be greater of equal to 0, but was: {N}"
            )
        if not (np.remainder(N, 1.0) == 0.0):
            raise ValueError(
                f"Value of 'N' for 'Hypergeometric' must be (equal to) an integer value, but was: {N}"
            )

        if M < 0 or M > N or not np.isfinite(M):
            raise ValueError(
                f"Value of 'M' for 'Hypergeometric' must be greater of equal to 0 and less or equal to N, but was: {M}"
            )
        if not (np.remainder(M, 1.0) == 0.0):
            raise ValueError(
                f"Value of 'M' for 'Hypergeometric' must be (equal to) an integer value, but was: {M}"
            )

        if n < 0 or n > N or not np.isfinite(n):
            raise ValueError(
                f"Value of 'n' for 'Hypergeometric' must be greater of equal to 0 and less or equal to N, but was: {n}"
            )
        if not (np.remainder(n, 1.0) == 0.0):
            raise ValueError(
                f"Value of 'n' for 'Hypergeometric' must be (equal to) an integer value, but was: {n}"
            )

        self.N = N
        self.M = M
        self.n = n

    def get_params(self) -> Tuple[int, int, int]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of integer values representing the size of the total population, the size of the population of interest and the number of draws.
        """
        return self.N, self.M, self.n

    def check_support(
        self, data: np.ndarray, is_scope_data: bool = False
    ) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Hypergeometric distribution, which is:

        .. math::

            \text{supp}(\text{Hypergeometric})={\max(0,n+M-N),...,\min(n,M)}

        where
            - :math:`N` is the total number of entities
            - :math:`M` is the number of entities with property of interest
            - :math:`n` is the number of draws

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

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scope.query):
            raise ValueError(
                f"Expected 'scope_data' to be of shape (n,{len(self.scope.query)}), but was: {scope_data.shape}"
            )

        valid = np.ones(scope_data.shape, dtype=bool)

        # nan entries (regarded as valid)
        nan_mask = np.isnan(scope_data)

        # check for infinite values
        valid[~nan_mask] &= ~np.isinf(scope_data[~nan_mask])

        # check if all values are valid integers
        valid[valid & ~nan_mask] &= (
            np.remainder(scope_data[valid & ~nan_mask], 1) == 0
        )

        # check if values are in valid range
        valid[valid & ~nan_mask] &= (
            scope_data[valid & ~nan_mask] >= max(0, self.n + self.M - self.N)
        ) & (scope_data[valid & ~nan_mask] <= min(self.n, self.M))

        return valid
