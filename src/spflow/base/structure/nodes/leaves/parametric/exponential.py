"""
Created on November 6, 2021

@authors: Bennet Wittelsbach, Philipp Deibert
"""

from .parametric import ParametricLeaf
from .statistical_types import ParametricType
from .exceptions import InvalidParametersError
from typing import Tuple, Dict, List
import numpy as np
from scipy.stats import expon  # type: ignore
from scipy.stats._distn_infrastructure import rv_continuous  # type: ignore

from multipledispatch import dispatch  # type: ignore


class Exponential(ParametricLeaf):
    """(Univariate) Exponential distribution.

    PDF(x) =
        l * exp(-l * x) , if x > 0
        0               , if x <= 0

    Attributes:
        l:
            Rate parameter of the Exponential distribution (usually denoted as lambda, must be greater than 0).
    """

    type = ParametricType.POSITIVE

    def __init__(self, scope: List[int], l: float) -> None:

        if len(scope) != 1:
            raise ValueError(f"Scope size for Exponential should be 1, but was: {len(scope)}")

        super().__init__(scope)
        self.set_params(l)

    def set_params(self, l: float) -> None:

        if l <= 0.0 or not np.isfinite(l):
            raise ValueError(
                f"Value of l for Exponential distribution must be greater than 0, but was: {l}"
            )

        self.l = l

    def get_params(self) -> Tuple[float]:
        return (self.l,)


@dispatch(Exponential)  # type: ignore[no-redef]
def get_scipy_object(node: Exponential) -> rv_continuous:
    return expon


@dispatch(Exponential)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Exponential) -> Dict[str, float]:
    if node.l is None:
        raise InvalidParametersError(f"Parameter 'l' of {node} must not be None")
    parameters = {"scale": 1.0 / node.l}
    return parameters
