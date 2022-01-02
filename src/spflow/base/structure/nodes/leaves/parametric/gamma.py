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
    r"""(Univariate) Gamma distribution.
    
    .. math::
    
        \text{PDF}(x) = \begin{cases} \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x} & \text{if } x > 0\\
                                      0 & \text{if } x <= 0\end{cases}

    where
        - :math:`x` is the input observation
        - :math:`\Gamma` is the Gamma function
        - :math:`\alpha` is the shape parameter
        - :math:`\beta` is the rate parameter

    TODO: check
    
    Args:
        scope:
            List of integers specifying the variable scope.
        alpha:
            Shape parameter (:math:`\alpha`), greater than 0.
        beta:
            Rate parameter (:math:`\beta`), greater than 0.
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

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if instances are part of the support of the Gamma distribution.

        .. math::

            \text{supp}(\text{Gamma})=(0,+\infty)

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

        # check if values are in valid range
        valid[valid] &= (scope_data[valid] > 0).sum(axis=-1).astype(bool)

        return valid


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
