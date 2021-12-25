"""
Created on November 6, 2021

@authors: Bennet Wittelsbach, Philipp Deibert
"""

from .parametric import ParametricLeaf
from .statistical_types import ParametricType
from .exceptions import InvalidParametersError
from typing import Tuple, Dict, List
import numpy as np
from scipy.stats import uniform  # type: ignore
from scipy.stats._distn_infrastructure import rv_continuous  # type: ignore

from multipledispatch import dispatch  # type: ignore


class Uniform(ParametricLeaf):
    """(Univariate) continuous Uniform distribution.

    PDF(x) =
        1 / (end - start) * 1_[start, end], where
            - 1_[start, end] is the indicator function of the given interval (evaluating to 0 if x is not in the interval)


    Attributes:
        start:
            Start of interval.
        end:
            End of interval (must be larger the interval start).
    """

    type = ParametricType.CONTINUOUS

    def __init__(
        self, scope: List[int], start: float, end: float, support_outside: bool = True
    ) -> None:

        if len(scope) != 1:
            raise ValueError(f"Scope size for Poisson should be 1, but was: {len(scope)}")

        super().__init__(scope)
        self.set_params(start, end, support_outside)

    def set_params(self, start: float, end: float, support_outside: bool = True) -> None:

        if not start < end:
            raise ValueError(
                f"Lower bound for Uniform distribution must be less than upper bound, but were: {start}, {end}"
            )
        if not (np.isfinite(start) and np.isfinite(end)):
            raise ValueError(f"Lower and upper bound must be finite, but were: {start}, {end}")

        self.start = start
        self.end = end
        self.support_outside = support_outside

    def get_params(self) -> Tuple[float, float, bool]:
        return self.start, self.end, self.support_outside

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:

        valid = np.ones(scope_data.shape, dtype=bool)

        # check for infinite values
        valid &= ~np.isinf(scope_data)

        # check if values are in valid range
        if not self.support_outside:
            valid[valid] &= (scope_data[valid] >= self.start) & (scope_data[valid] <= self.end)

        return valid


@dispatch(Uniform)  # type: ignore[no-redef]
def get_scipy_object(node: Uniform) -> rv_continuous:
    return uniform


@dispatch(Uniform)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Uniform) -> Dict[str, float]:
    if node.start is None:
        raise InvalidParametersError(f"Parameter 'start' of {node} must not be None")
    if node.end is None:
        raise InvalidParametersError(f"Parameter 'end' of {node} must not be None")
    parameters = {"loc": node.start, "scale": node.end - node.start}
    return parameters
