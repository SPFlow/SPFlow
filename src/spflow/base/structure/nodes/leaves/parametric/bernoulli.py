"""
Created on November 6, 2021

@authors: Philipp Deibert, Bennet Wittelsbach
"""
from typing import Tuple, Optional
import numpy as np
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import bernoulli
from scipy.stats.distributions import rv_frozen


class Bernoulli(LeafNode):
    r"""(Univariate) Bernoulli distribution.

    .. math::

        \text{PMF}(k)=\begin{cases} p   & \text{if } k=1\\
                                    1-p & \text{if } k=0\end{cases}
        
    where
        - :math:`p` is the success probability
        - :math:`k` is the outcome of the trial (0 or 1)

    Args:
        scope:
            Scope object specifying the variable scope.
        p:
            Probability of success in the range :math:`[0,1]` (default 0.5).
    """
    def __init__(self, scope: Scope, p: Optional[float]=0.5) -> None:

        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for Bernoulli should be 1, but was {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for Bernoulli should be empty, but was {scope.evidence}.")

        super(Bernoulli, self).__init__(scope=scope)
        self.set_params(p)
    
    @property
    def dist(self) -> rv_frozen:
        return bernoulli(p=self.p)

    def set_params(self, p: float) -> None:

        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for Bernoulli distribution must to be between 0.0 and 1.0, but was: {p}"
            )

        self.p = p

    def get_params(self) -> Tuple[float]:
        return (self.p,)

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if instances are part of the support of the Bernoulli distribution.

        .. math::

            \text{supp}(\text{Bernoulli})=\{0,1\}

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

        # check if all values are valid integers
        # TODO: runtime warning due to nan values
        valid[valid] &= np.remainder(scope_data[valid], 1) == 0

        # check if values are in valid range
        valid[valid] &= (scope_data[valid] >= 0) & (scope_data[valid] <= 1)

        return valid