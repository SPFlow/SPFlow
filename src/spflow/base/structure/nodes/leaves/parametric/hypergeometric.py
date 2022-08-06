"""
Created on November 6, 2021

@authors: Philipp Deibert, Bennet Wittelsbach
"""
from typing import Tuple, List
import numpy as np
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import LeafNode


class Hypergeometric(LeafNode):
    r"""(Univariate) Hypergeometric distribution.

    .. math::

        \text{PMF}(k) = \frac{\binom{M}{k}\binom{N-M}{n-k}}{\binom{N}{n}}

    where
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)
        - :math:`N` is the total number of entities
        - :math:`M` is the number of entities with property of interest
        - :math:`n` is the number of draws
        - :math:`k` s the number of observed entities

    Args:
        scope:
            Scope object specifying the variable scope.
        N:
            Total number of entities (in the population), greater or equal to 0.
        M:
            Number of entities with property of interest (in the population), greater or equal to zero and less than or equal to N.
        n:
            Number of draws, greater of euqal to zero and less than or equal to N.
    """
    def __init__(self, scope: Scope, N: int, M: int, n: int) -> None:

        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for Hypergeometric should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for Hypergeometric should be empty, but was {scope.evidence}.")

        super(Hypergeometric, self).__init__(scope=scope)
        self.set_params(N, M, n)

    def set_params(self, N: int, M: int, n: int) -> None:

        if N < 0 or not np.isfinite(N):
            raise ValueError(
                f"Value of N for Hypergeometric distribution must be greater of equal to 0, but was: {N}"
            )
        if not (np.remainder(N, 1.0) == 0.0):
            raise ValueError(
                f"Value of N for Hypergeometric distribution must be (equal to) an integer value, but was: {N}"
            )

        if M < 0 or M > N or not np.isfinite(M):
            raise ValueError(
                f"Value of M for Hypergeometric distribution must be greater of equal to 0 and less or equal to N, but was: {M}"
            )
        if not (np.remainder(M, 1.0) == 0.0):
            raise ValueError(
                f"Value of M for Hypergeometric distribution must be (equal to) an integer value, but was: {M}"
            )

        if n < 0 or n > N or not np.isfinite(n):
            raise ValueError(
                f"Value of n for Hypergeometric distribution must be greater of equal to 0 and less or equal to N, but was: {n}"
            )
        if not (np.remainder(n, 1.0) == 0.0):
            raise ValueError(
                f"Value of n for Hypergeometric distribution must be (equal to) an integer value, but was: {n}"
            )

        self.N = N
        self.M = M
        self.n = n

    def get_params(self) -> Tuple[int, int, int]:
        return self.N, self.M, self.n

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if instances are part of the support of the Hypergeometric distribution.

        .. math::

            \text{supp}(\text{Hypergeometric})={\max(0,n+M-N),...,\min(n,M)}

        where
            - :math:`N` is the total number of entities
            - :math:`M` is the number of entities with property of interest
            - :math:`n` is the number of draws

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

        valid = np.ones(scope_data.shape[0], dtype=bool)

        # check for infinite values
        valid &= ~np.isinf(scope_data).sum(axis=-1).astype(bool)

        # check if all values are valid integers
        # TODO: runtime warning due to nan values
        valid[valid] &= (np.remainder(scope_data[valid], 1) == 0).sum(axis=-1).astype(bool)

        # check if values are in valid range
        valid[valid] &= (
            (
                (scope_data[valid] >= max(0, self.n + self.M - self.N))
                & (scope_data[valid] <= min(self.n, self.M))
            )
            .sum(axis=-1)
            .astype(bool)
        )

        return valid