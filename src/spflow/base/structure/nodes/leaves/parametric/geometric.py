# -*- coding: utf-8 -*-
"""Contains Geometric leaf node for SPFlow in the ``base`` backend.
"""
from typing import Tuple, Optional
import numpy as np
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import geom  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore


class Geometric(LeafNode):
    r"""(Univariate) Geometric distribution leaf node in the ``base`` backend.

    Represents an univariate Geometric distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) =  p(1-p)^{k-1}

    where
        - :math:`k` is the number of trials
        - :math:`p` is the success probability of each trial

    Attributes:
        p:
            Floating points representing the probability of success in the range :math:`(0,1]`.
    """

    def __init__(self, scope: Scope, p: Optional[float] = 0.5) -> None:
        r"""Initializes ``Geometric`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            p:
                Floating points representing the probability of success in the range :math:`(0,1]`.
                Defaults to 0.5.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'Geometric' should be 1, but was {len(scope.query)}."
            )
        if len(scope.evidence):
            raise ValueError(
                f"Evidence scope for 'Geometric' should be empty, but was {scope.evidence}."
            )

        super(Geometric, self).__init__(scope=scope)
        self.set_params(p)

    @property
    def dist(self) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return geom(p=self.p)

    def set_params(self, p: float) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            p:
                Floating points representing the probability of success in the range :math:`(0,1]`.
        """
        if p <= 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of 'p' for 'Geometric' must to be greater than 0.0 and less or equal to 1.0, but was: {p}"
            )

        self.p = p

    def get_params(self) -> Tuple[float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Floating point value representing the success probability.
        """
        return (self.p,)

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Geometric distribution, which is:

        .. math::

            \text{supp}(\text{Geometric})=\mathbb{N}\setminus\{0\}

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            scope_data:
                Two-dimensional NumPy array containing sample instances.
                Each row is regarded as a sample.
        Returns:
            Two dimensional NumPy array indicating for each instance, whether they are part of the support (True) or not (False).
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

        # check if all values are valid integers
        valid[valid & ~nan_mask] &= (
            np.remainder(scope_data[valid & ~nan_mask], 1) == 0
        )

        # check if values are in valid range
        valid[valid & ~nan_mask] &= scope_data[valid & ~nan_mask] >= 1

        return valid
