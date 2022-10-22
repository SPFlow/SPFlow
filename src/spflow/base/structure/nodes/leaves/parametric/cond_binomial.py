"""
Created on October 18, 2022

@authors: Philipp Deibert
"""
from typing import Tuple, Optional, Callable, Union
import numpy as np
from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import binom
from scipy.stats.distributions import rv_frozen


class CondBinomial(LeafNode):
    r"""Conditional (univariate) Binomial distribution.

    .. math::

        \text{PMF}(k) = \binom{n}{k}p^k(1-p)^{n-k}

    where
        - :math:`p` is the success probability of each trial
        - :math:`n` is the number of total trials
        - :math:`k` is the number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Args:
        scope:
            Scope object specifying the variable scope.
        n:
            Number of i.i.d. Bernoulli trials (greater of equal to 0).
        cond_f:
            Callable that provides the conditional parameters (p) of this distribution. TODO
    """
    def __init__(self, scope: Scope, n: int, cond_f: Optional[Callable]=None) -> None:

        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for Binomial should be 1, but was {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for Binomial should be empty, but was {scope.evidence}.")

        super(CondBinomial, self).__init__(scope=scope)
        
        self.set_params(n)
        
        self.set_cond_f(cond_f)

    def set_cond_f(self, cond_f: Optional[Callable]=None) -> None:
        self.cond_f = cond_f

    def dist(self, p: float) -> rv_frozen:
        return binom(n=self.n, p=p)

    def set_params(self, n: int) -> None:

        if n < 0 or not np.isfinite(n):
            raise ValueError(
                f"Value of n for Binomial distribution must to greater of equal to 0, but was: {n}"
            )

        if not (np.remainder(n, 1.0) == 0.0):
            raise ValueError(
                f"Value of n for Binomial distribution must be (equal to) an integer value, but was: {n}"
            )

        self.n = n

    def retrieve_params(self, data: np.ndarray, dispatch_ctx: DispatchContext) -> Tuple[Union[np.ndarray, float]]:
        
        p, cond_f = None, None

        # check dispatch cache for required conditional parameter 'p'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if a value for 'p' is specified (highest priority)
            if "p" in args:
                p = args["p"]
            # check if alternative function to provide 'p' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'p' specified (lowest priority)
            cond_f = self.cond_f
        
        # if neither 'p' nor 'cond_f' is specified (via node or arguments)
        if p is None and cond_f is None:
            raise ValueError("'CondBinomial' requires either 'p' or 'cond_f' to retrieve 'p' to be specified.")

        # if 'p' was not already specified, retrieve it
        if p is None:
            p = cond_f(data)['p']

        # check if value for 'p' is valid
        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for conditional Binomial distribution must to be between 0.0 and 1.0, but was: {p}"
            )
        
        return p

    def get_params(self) -> Tuple[int]:
        return (self.n,)

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if instances are part of the support of the Binomial distribution.

        .. math::

            \text{supp}(\text{Binomial})=\{0,\hdots,n\}
        
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

        # check if all values are valid integers
        valid[valid & ~nan_mask] &= np.remainder(scope_data[valid & ~nan_mask], 1) == 0

        # check if values are in valid range
        valid[valid & ~nan_mask] &= (scope_data[valid & ~nan_mask] >= 0) & (scope_data[valid & ~nan_mask] <= self.n)

        return valid