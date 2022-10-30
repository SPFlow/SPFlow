# -*- coding: utf-8 -*-
"""Contains Log-Normal leaf node for SPFlow in the ``base`` backend.
"""
from typing import Optional, Tuple
import numpy as np
from spflow.meta.data.scope import Scope
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import lognorm  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore


class LogNormal(LeafNode):
    r"""(Univariate) Log-Normal distribution leaf node in the ``base`` backend.

    Represents an univariate Log-Normal distribution, with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{x\sigma\sqrt{2\pi}}\exp\left(-\frac{(\ln(x)-\mu)^2}{2\sigma^2}\right)

    where
        - :math:`x` is an observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

    Attributes:
        scope:
            Scope object specifying the variable scope.
        mean:
            Floating point value representing the mean (:math:`\mu`) of the distribution.
        std:
            Floating point values representing the standard deviation (:math:`\sigma`) of the distribution (must be greater than 0).
    """

    def __init__(
        self, scope: Scope, mean: float = 0.0, std: float = 1.0
    ) -> None:
        r"""Initializes ``LogNormal`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            mean:
                Floating point value representing the mean (:math:`\mu`) of the distribution.
                Defaults to 0.0.
            std:
                Floating point values representing the standard deviation (:math:`\sigma`) of the distribution (must be greater than 0).
                Defaults to 1.0.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'LogNormal' should be 1, but was: {len(scope.query)}."
            )
        if len(scope.evidence):
            raise ValueError(
                f"Evidence scope for 'LogNormal' should be empty, but was {scope.evidence}."
            )

        super(LogNormal, self).__init__(scope=scope)
        self.set_params(mean, std)

    @property
    def dist(self) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return lognorm(loc=0.0, scale=np.exp(self.mean), s=self.std)

    def set_params(self, mean: float, std: float) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            mean:
                Floating point value representing the mean (:math:`\mu`) of the distribution.
            std:
                Floating point values representing the standard deviation (:math:`\sigma`) of the distribution (must be greater than 0).
        """
        if not (np.isfinite(mean) and np.isfinite(std)):
            raise ValueError(
                f"Values for 'mean' and 'std' for 'LogNormal' must be finite, but were: {mean}, {std}"
            )
        if std <= 0.0:
            raise ValueError(
                f"Value for 'std' for 'LogNormal' must be greater than 0.0, but was: {std}"
            )

        self.mean = mean
        self.std = std

    def get_params(self) -> Tuple[float, float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of the floating point values representing the mean and standard deviation.
        """
        return self.mean, self.std

    def check_support(
        self, data: np.ndarray, is_scope_data: bool = False
    ) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Log-Normal distribution, which is:

        .. math::

            \text{supp}(\text{LogNormal})=(0,\infty)

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional NumPy array containing sample instances.
                Each row is regarded as a sample.
                Unless ``is_scope_data`` is set to True, it is assumed that the relevant data is located in the columns corresponding to the scope indices.
            is_scope_data:
                Boolean indicating if the given data already contains the relevant data for the leaf's scope in the correct order (True) or if it needs to be extracted from the full data set.
                Defaults to False.

        Returns:
            Two-dimensional NumPy array indicating for each instance, whether they are part of the support (True) or not (False).
        """
        if is_scope_data:
            scope_data = data
        else:
            # select relevant data for scope
            scope_data = data[:, self.scope.query]

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scope.query):
            raise ValueError(
                f"Expected 'scope_data' to be of shape (n,{len(self.scope.query)}), but was: {scope_data.shape}"
            )

        valid = np.ones(scope_data.shape, dtype=bool)

        # nan entries (regarded as valid)
        nan_mask = np.isnan(scope_data)

        # check for infinite values
        valid[~nan_mask] &= ~np.isinf(scope_data[~nan_mask])

        # check if values are in valid range
        valid[valid & ~nan_mask] &= scope_data[valid & ~nan_mask] > 0

        return valid
