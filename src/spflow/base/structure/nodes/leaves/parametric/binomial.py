"""
Created on November 6, 2021

@authors: Bennet Wittelsbach, Philipp Deibert
"""

from .parametric import ParametricLeaf
from .statistical_types import ParametricType
from .exceptions import InvalidParametersError
from typing import Tuple, Dict, List, Union
import numpy as np
from scipy.stats import binom  # type: ignore
from scipy.stats._distn_infrastructure import rv_discrete  # type: ignore

from multipledispatch import dispatch  # type: ignore


class Binomial(ParametricLeaf):
    r"""(Univariate) Binomial distribution.

    .. math::

        \text{PMF}(k) = \binom{n}{k}p^k(1-p)^{n-k}

    where
        - :math:`p` is the success probability of each trial
        - :math:`n` is the number of total trials
        - :math:`k` is the number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Args:
        scope:
            List of integers specifying the variable scope.
        n:
            Number of i.i.d. Bernoulli trials (greater of equal to 0).
        p:
            Probability of success of each trial in the range :math:`[0,1]`.
    """

    type = ParametricType.COUNT

    def __init__(self, scope: List[int], n: int, p: float) -> None:

        if len(scope) != 1:
            raise ValueError(f"Scope size for Binomial should be 1, but was: {len(scope)}")

        super().__init__(scope)
        self.set_params(n, p)

    def set_params(self, n: int, p: float) -> None:

        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for Binomial distribution must to be between 0.0 and 1.0, but was: {p}"
            )

        if n < 0 or not np.isfinite(n):
            raise ValueError(
                f"Value of n for Binomial distribution must to greater of equal to 0, but was: {n}"
            )

        if not (np.remainder(n, 1.0) == 0.0):
            raise ValueError(
                f"Value of n for Binomial distribution must be (equal to) an integer value, but was: {n}"
            )

        self.n = n
        self.p = p

    def get_params(self) -> Tuple[int, float]:
        return self.n, self.p

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if instances are part of the support of the Binomial distribution.

        .. math::

            \text{supp}(\text{Binomial})=\{0,\hdots,n\}

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
            ((scope_data[valid] >= 0) & (scope_data[valid] <= self.n)).sum(axis=-1).astype(bool)
        )

        return valid


@dispatch(Binomial)  # type: ignore[no-redef]
def get_scipy_object(node: Binomial) -> rv_discrete:
    return binom


@dispatch(Binomial)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Binomial) -> Dict[str, Union[int, float]]:
    if node.n is None:
        raise InvalidParametersError(f"Parameter 'n' of {node} must not be None")
    if node.p is None:
        raise InvalidParametersError(f"Parameter 'p' of {node} must not be None")
    parameters = {"n": node.n, "p": node.p}
    return parameters
