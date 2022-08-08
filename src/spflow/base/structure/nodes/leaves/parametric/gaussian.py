"""
Created on November 6, 2021

@authors: Philipp Deibert, Bennet Wittelsbach
"""
from typing import Tuple, Optional
import numpy as np
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import norm
from scipy.stats.distributions import rv_frozen


class Gaussian(LeafNode):
    r"""(Univariate) Normal distribution.

    .. math::

        \text{PDF}(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})

    where
        - :math:`x` the observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

    Args:
        scope:
            Scope object specifying the variable scope.
        mean:
            mean (:math:`\mu`) of the distribution (default 0.0).
        std:
            standard deviation (:math:`\sigma`) of the distribution (must be greater than 0; default 0.0).
    """
    def __init__(
        self,
        scope: Scope,
        mean: float=0.0,
        std: float=1.0,
    ) -> None:

        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for Gaussian should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for Gaussian should be empty, but was {scope.evidence}.")

        super(Gaussian, self).__init__(scope=scope)
        self.set_params(mean, std)
    
    @property
    def dist(self) -> rv_frozen:
        return norm(loc=self.mean, scale=self.std)

    def set_params(self, mean: Optional[float]=0.0, std: Optional[float]=1.0) -> None:

        if not (np.isfinite(mean) and np.isfinite(std)):
            raise ValueError(
                f"Mean and standard deviation for Gaussian distribution must be finite, but were: {mean}, {std}"
            )
        if std <= 0.0:
            raise ValueError(
                f"Standard deviation for Gaussian distribution must be greater than 0.0, but was: {std}"
            )

        self.mean = mean
        self.std = std

    def get_params(self) -> Tuple[float, float]:
        return self.mean, self.std

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if instances are part of the support of the Gaussian distribution.

        .. math::

            \text{supp}(\text{Gaussian})=(-\infty,+\infty)

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

        valid = np.ones(scope_data.shape[0], dtype=bool)

        # check for infinite values
        valid &= ~np.isinf(scope_data).sum(axis=-1).astype(bool)

        return valid