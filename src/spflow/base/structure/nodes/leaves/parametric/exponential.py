# -*- coding: utf-8 -*-
"""Contains Exponential leaf node for SPFlow in the ``base`` backend.
"""
from typing import Tuple
import numpy as np
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import expon  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore


class Exponential(LeafNode):
    r"""(Univariate) Exponential distribution leaf node in the ``base`` backend.

    Represents an univariate Exponential distribution, with the following probability distribution function (PDF):

    .. math::
        
        \text{PDF}(x) = \begin{cases} \lambda e^{-\lambda x} & \text{if } x > 0\\
                                      0                      & \text{if } x <= 0\end{cases}
    
    where
        - :math:`x` is the input observation
        - :math:`\lambda` is the rate parameter
    
    Attributes:
        l:
            Floating point value representing the rate parameter (:math:`\lambda`) of the Exponential distribution (must be greater than 0; default 1.0).
    """

    def __init__(self, scope: Scope, l: float = 1.0) -> None:
        r"""Initializes ``Exponential`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            l:
                Floating point value representing the rate parameter (:math:`\lambda`) of the Exponential distribution (must be greater than 0).
                Defaults to 1.0.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'Exponential' should be 1, but was {len(scope.query)}."
            )
        if len(scope.evidence):
            raise ValueError(
                f"Evidence scope for 'Exponential' should be empty, but was {scope.evidence}."
            )

        super(Exponential, self).__init__(scope=scope)
        self.set_params(l)

    @property
    def dist(self) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return expon(scale=1.0 / self.l)

    def set_params(self, l: float) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            l:
                Floating point value representing the rate parameter (:math:`\lambda`) of the Exponential distribution (must be greater than 0).
        """
        if l <= 0.0 or not np.isfinite(l):
            raise ValueError(
                f"Value of 'l' for 'Exponential' must be greater than 0, but was: {l}"
            )

        self.l = l

    def get_params(self) -> Tuple[float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Floating point value representing the rate parameter.
        """
        return (self.l,)

    def check_support(self, data: np.ndarray, is_scope_data: bool=False) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Exponential distribution, which is:

        .. math::

            \text{supp}(\text{Exponential})=[0,+\infty)

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            scope_data:
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
        valid[valid & ~nan_mask] &= scope_data[valid & ~nan_mask] >= 0

        return valid
