"""
Created on November 6, 2021

@authors: Bennet Wittelsbach, Philipp Deibert
"""

from .parametric import ParametricLeaf
from .statistical_types import ParametricType
from .exceptions import InvalidParametersError
from typing import Tuple, Dict, List
import numpy as np
from scipy.stats import poisson  # type: ignore
from scipy.stats._distn_infrastructure import rv_discrete  # type: ignore

from multipledispatch import dispatch  # type: ignore


class Poisson(ParametricLeaf):
    r"""(Univariate) Poisson distribution.

    .. math::

        \text{PMF}(k) = \lambda^k\frac{e^{-\lambda}}{k!}

    where
        - :math:`k` is the number of occurrences
        - :math:`\lambda` is the rate parameter

    Args:
        scope:
            List of integers specifying the variable scope.
        l:
            Rate parameter (:math:`\lambda`), expected value and variance of the Poisson distribution (must be greater than or equal to 0).
    """

    type = ParametricType.COUNT

    def __init__(self, scope: List[int], l: float) -> None:

        if len(scope) != 1:
            raise ValueError(f"Scope size for Poisson should be 1, but was: {len(scope)}")

        super().__init__(scope)
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

        valid = np.ones(scope_data.shape, dtype=bool)

        # check for infinite values
        valid &= ~np.isinf(scope_data)

        # check if all values are valid integers
        # TODO: runtime warning due to nan values
        valid[valid] &= np.remainder(scope_data[valid], 1) == 0

        # check if values are in valid range
        valid[valid] &= scope_data[valid] >= 0

        return valid


@dispatch(Poisson)  # type: ignore[no-redef]
def get_scipy_object(node: Poisson) -> rv_discrete:
    return poisson


@dispatch(Poisson)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Poisson) -> Dict[str, float]:
    if node.l is None:
        raise InvalidParametersError(f"Parameter 'l' of {node} must not be None")
    parameters = {"mu": node.l}
    return parameters
