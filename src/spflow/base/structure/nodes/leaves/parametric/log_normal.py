"""
Created on November 6, 2021

@authors: Philipp Deibert, Bennet Wittelsbach
"""
from typing import Optional, Tuple
import numpy as np
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import lognorm
from scipy.stats.distributions import rv_frozen


class LogNormal(LeafNode):
    r"""(Univariate) Log-Normal distribution.

    .. math::

        \text{PDF}(x) = \frac{1}{x\sigma\sqrt{2\pi}}\exp\left(-\frac{(\ln(x)-\mu)^2}{2\sigma^2}\right)

    where
        - :math:`x` is an observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

    Args:
        scope:
            Scope object specifying the variable scope.
        mean:
            mean (:math:`\mu`) of the distribution (default 0.0).
        std:
            standard deviation (:math:`\sigma`) of the distribution (must be greater than 0; default 1.0).
    """
    def __init__(self, scope: Scope, mean: Optional[float]=0.0, std: Optional[float]=1.0) -> None:

        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for LogNormal should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for LogNormal should be empty, but was {scope.evidence}.")

        super(LogNormal, self).__init__(scope=scope)
        self.set_params(mean, std)
    
    @property
    def dist(self) -> rv_frozen:
        return lognorm(loc=0.0, scale=np.exp(self.mean), s=self.std)

    def set_params(self, mean: float, std: float) -> None:

        if not (np.isfinite(mean) and np.isfinite(std)):
            raise ValueError(
                f"Mean and standard deviation for LogNormal distribution must be finite, but were: {mean}, {std}"
            )
        if std <= 0.0:
            raise ValueError(
                f"Standard deviation for LogNormal distribution must be greater than 0.0, but was: {std}"
            )

        self.mean = mean
        self.std = std

    def get_params(self) -> Tuple[float, float]:
        return self.mean, self.std

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if instances are part of the support of the LogNormal distribution.

        .. math::

            \text{supp}(\text{LogNormal})=(0,\infty)

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

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

        # nan entries (regarded as valid)
        nan_mask = np.isnan(scope_data)

        # check for infinite values
        valid[~nan_mask] &= ~np.isinf(scope_data[~nan_mask])

        # check if values are in valid range
        valid[valid & ~nan_mask] &= (scope_data[valid & ~nan_mask] > 0)

        return valid