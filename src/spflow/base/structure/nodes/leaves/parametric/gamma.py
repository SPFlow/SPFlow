"""
Created on November 6, 2021

@authors: Bennet Wittelsbach, Philipp Deibert
"""

from .parametric import ParametricLeaf
from .statistical_types import ParametricType
from .exceptions import InvalidParametersError
from typing import Tuple, Dict, List
import numpy as np
from scipy.stats import gamma  # type: ignore
from scipy.stats._distn_infrastructure import rv_continuous  # type: ignore

from multipledispatch import dispatch  # type: ignore


class Gamma(ParametricLeaf):
    """(Univariate) Gamma distribution.

    PDF(x) =
        1/G(alpha) * beta^alpha * x^(alpha-1) * exp(-x*beta)   , if x > 0
        0                                                      , if x <= 0, where
            - G(beta) is the Gamma function

    Attributes:
        alpha:
            Shape parameter, greater than 0.
        beta:
            Scale parameter, greater than 0.
    """

    type = ParametricType.POSITIVE

    def __init__(self, scope: List[int], alpha: float, beta: float) -> None:

        if len(scope) != 1:
            raise ValueError(f"Scope size for Gamma should be 1, but was: {len(scope)}")

        super().__init__(scope)
        self.set_params(alpha, beta)

    def set_params(self, alpha: float, beta: float) -> None:

        if alpha <= 0.0 or not np.isfinite(alpha):
            raise ValueError(
                f"Value of alpha for Gamma distribution must be greater than 0, but was: {alpha}"
            )
        if beta <= 0.0 or not np.isfinite(beta):
            raise ValueError(
                f"Value of beta for Gamma distribution must be greater than 0, but was: {beta}"
            )

        self.alpha = alpha
        self.beta = beta

    def get_params(self) -> Tuple[float, float]:
        return self.alpha, self.beta


@dispatch(Gamma)  # type: ignore[no-redef]
def get_scipy_object(node: Gamma) -> rv_continuous:
    return gamma


@dispatch(Gamma)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Gamma) -> Dict[str, float]:
    if node.alpha is None:
        raise InvalidParametersError(f"Parameter 'alpha' of {node} must not be None")
    if node.beta is None:
        raise InvalidParametersError(f"Parameter 'beta' of {node} must not be None")
    parameters = {"a": node.alpha, "scale": 1.0 / node.beta}
    return parameters
