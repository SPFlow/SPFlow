"""
Created on November 6, 2021

@authors: Philipp Deibert, Bennet Wittelsbach
"""
from typing import Tuple, Optional
import numpy as np
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import gamma
from scipy.stats.distributions import rv_frozen


class Gamma(LeafNode):
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
            Scope object specifying the variable scope.
        alpha:
            Shape parameter (:math:`\alpha`), greater than 0 (default 1.0).
        beta:
            Rate parameter (:math:`\beta`), greater than 0 (default 1.0).
    """
    def __init__(self, scope: Scope, alpha: Optional[float]=1.0, beta: Optional[float]=1.0) -> None:

        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for Gamma should be 1, but was {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for Gamma should be empty, but was {scope.evidence}.")

        super(Gamma, self).__init__(scope=scope)
        self.set_params(alpha, beta)
    
    @property
    def dist(self) -> rv_frozen:
        return gamma(a=self.alpha, scale=1.0/self.beta)

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

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scope.query):
            raise ValueError(
                f"Expected scope_data to be of shape (n,{len(self.scope.query)}), but was: {scope_data.shape}"
            )

        valid = np.ones(scope_data.shape, dtype=bool)

        # check for infinite values
        valid &= ~np.isinf(scope_data)

        # check if values are in valid range
        valid[valid] &= (scope_data[valid] > 0)

        return valid