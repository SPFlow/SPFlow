"""
Created on November 6, 2021

@authors: Bennet Wittelsbach, Philipp Deibert
"""

from .parametric import ParametricLeaf
from .statistical_types import ParametricType
from .exceptions import InvalidParametersError
from typing import Tuple, Dict, List
import numpy as np
from scipy.stats import bernoulli  # type: ignore
from scipy.stats._distn_infrastructure import rv_discrete  # type: ignore

from multipledispatch import dispatch  # type: ignore


class Bernoulli(ParametricLeaf):
    """(Univariate) Binomial distribution

    PMF(k) =
        p   , if k=1
        1-p , if k=0

    Attributes:
        p:
            Probability of success in the range [0,1].
    """

    type = ParametricType.BINARY

    def __init__(self, scope: List[int], p: float) -> None:

        if len(scope) != 1:
            raise ValueError(f"Scope size for Bernoulli should be 1, but was: {len(scope)}")

        super().__init__(scope)
        self.set_params(p)

    def set_params(self, p: float) -> None:

        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for Bernoulli distribution must to be between 0.0 and 1.0, but was: {p}"
            )

        self.p = p

    def get_params(self) -> Tuple[float]:
        return (self.p,)

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:

        valid = np.ones(scope_data.shape, dtype=bool)

        # check for infinite values
        valid &= ~np.isinf(scope_data)

        # check if all values are valid integers
        # TODO: runtime warning due to nan values
        valid[valid] &= np.remainder(scope_data[valid], 1) == 0

        # check if values are in valid range
        valid[valid] &= (scope_data[valid] >= 0) & (scope_data[valid] <= 1)

        return valid


@dispatch(Bernoulli)  # type: ignore[no-redef]
def get_scipy_object(node: Bernoulli) -> rv_discrete:
    return bernoulli


@dispatch(Bernoulli)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Bernoulli) -> Dict[str, float]:
    if node.p is None:
        raise InvalidParametersError(f"Parameter 'p' of {node} must not be None")
    parameters = {"p": node.p}
    return parameters
