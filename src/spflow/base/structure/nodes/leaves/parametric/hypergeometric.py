"""
Created on November 6, 2021

@authors: Bennet Wittelsbach, Philipp Deibert
"""

from .parametric import ParametricLeaf
from .statistical_types import ParametricType
from .exceptions import InvalidParametersError
from typing import Tuple, Dict, List
import numpy as np
from scipy.stats import hypergeom  # type: ignore
from scipy.stats._distn_infrastructure import rv_discrete  # type: ignore

from multipledispatch import dispatch  # type: ignore


class Hypergeometric(ParametricLeaf):
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
            List of integers specifying the variable scope.
        N:
            Total number of entities (in the population), greater or equal to 0.
        M:
            Number of entities with property of interest (in the population), greater or equal to zero and less than or equal to N.
        n:
            Number of draws, greater of euqal to zero and less than or equal to N.
    """

    type = ParametricType.COUNT

    def __init__(self, scope: List[int], N: int, M: int, n: int) -> None:

        if len(scope) != 1:
            raise ValueError(f"Scope size for Hypergeometric should be 1, but was: {len(scope)}")

        super().__init__(scope)
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

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scope):
            raise ValueError(
                f"Expected scope_data to be of shape (n,{len(self.scope)}), but was: {scope_data.shape}"
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


@dispatch(Hypergeometric)  # type: ignore[no-redef]
def get_scipy_object(node: Hypergeometric) -> rv_discrete:
    return hypergeom


@dispatch(Hypergeometric)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Hypergeometric) -> Dict[str, int]:
    if node.N is None:
        raise InvalidParametersError(f"Parameter 'N' of {node} must not be None")
    if node.M is None:
        raise InvalidParametersError(f"Parameter 'M' of {node} must not be None")
    if node.n is None:
        raise InvalidParametersError(f"Parameter 'n' of {node} must not be None")
    # note: scipy hypergeom has switched semantics for the parameters
    parameters = {"M": node.N, "n": node.M, "N": node.n}
    return parameters
