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
    """(Univariate) Poisson distribution.

    PMF(k) =
        l^k * exp(-l) / k!

    Attributes:
        l:
            Expected value (& variance) of the Poisson distribution (usually denoted as lambda).
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

        self.l = l

    def get_params(self) -> Tuple[float]:
        return (self.l,)

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:

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
