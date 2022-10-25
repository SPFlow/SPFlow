# -*- coding: utf-8 -*-
"""Contains conditional Binomial leaf node for SPFlow in the 'base' backend.
"""
from typing import Tuple, Optional, Callable, Union
import numpy as np
from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import binom  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore


class CondBinomial(LeafNode):
    r"""Conditional (univariate) Binomial distribution leaf node in the 'base' backend.

    Represents a conditional univariate Binomial distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \binom{n}{k}p^k(1-p)^{n-k}

    where
        - :math:`p` is the success probability of each trial
        - :math:`n` is the number of total trials
        - :math:`k` is the number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Attributes:
        n:
            Integer representing the number of i.i.d. Bernoulli trials (greater or equal to 0).
        cond_f:
            Optional callable to retrieve the conditional parameter for the leaf node.
            Its output should be a dictionary containing ``p`` as a key, and the value should be
            a floating point value representing the success probability in :math:`[0,1]`.
    """
    def __init__(self, scope: Scope, n: int, cond_f: Optional[Callable]=None) -> None:
        r"""Initializes ``ConditionalBernoulli`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            n:
                Integer representing the number of i.i.d. Bernoulli trials (greater or equal to 0).
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``p`` as a key, and the value should be
                a floating point value representing the success probability in :math:`[0,1]`.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'CondBinomial' should be 1, but was {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for 'CondBinomial' should be empty, but was {scope.evidence}.")

        super(CondBinomial, self).__init__(scope=scope)
        
        self.set_params(n)
        
        self.set_cond_f(cond_f)

    def set_cond_f(self, cond_f: Optional[Callable]=None) -> None:
        r"""Sets the function to retrieve the node's conditonal parameter.

        Args:
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``p`` as a key, and the value should be
                a floating point value representing the success probability in :math:`[0,1]`.
        """
        self.cond_f = cond_f

    def dist(self, p: float) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Args:
            p:
                Floating point value representing the success probability of each trial between zero and one.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return binom(n=self.n, p=p)

    def set_params(self, n: int) -> None:
        """Sets the parameters for the represented distribution.

        Args:
            n:
                Integer representing the number of i.i.d. Bernoulli trials (greater or equal to 0).
        """
        if n < 0 or not np.isfinite(n):
            raise ValueError(
                f"Value of 'n' for 'Binomial' must to greater of equal to 0, but was: {n}"
            )

        if not (np.remainder(n, 1.0) == 0.0):
            raise ValueError(
                f"Value of 'n' for 'Binomial' must be (equal to) an integer value, but was: {n}"
            )

        self.n = n

    def retrieve_params(self, data: np.ndarray, dispatch_ctx: DispatchContext) -> Tuple[Union[np.ndarray, float]]:
        r"""Retrieves the conditional parameter of the leaf node.
    
        First, checks if conditional parameter (``p``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameter.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameter.

        Args:
            data:
                Two-dimensional NumPy array containing the data to compute the conditional parameters.
                Each row is regarded as a sample.
            dispatch_ctx:
                Dispatch context.

        Returns:
            One-dimensional NumPy array of non-zero weights
        
        Raises:
            ValueError: No way to retrieve conditional parameters or invalid conditional parameters.
        """
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
                f"Value of 'p' for 'CondBinomial' distribution must to be between 0.0 and 1.0, but was: {p}"
            )
        
        return p

    def get_params(self) -> Tuple[int]:
        """Returns the parameters of the represented distribution.

        Returns:
            Integer number representing the number of i.i.d. Bernoulli trials and the floating point value representing the success probability.
        """
        return (self.n,)

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Binomial distribution, which is:

        .. math::

            \text{supp}(\text{Binomial})=\{0,\hdots,n\}
        
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
        valid[valid & ~nan_mask] &= np.remainder(scope_data[valid & ~nan_mask], 1) == 0

        # check if values are in valid range
        valid[valid & ~nan_mask] &= (scope_data[valid & ~nan_mask] >= 0) & (scope_data[valid & ~nan_mask] <= self.n)

        return valid