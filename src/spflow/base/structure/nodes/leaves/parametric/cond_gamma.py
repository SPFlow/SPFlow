# -*- coding: utf-8 -*-
"""Contains conditional Gamma leaf node for SPFlow in the 'base' backend.
"""
from typing import Tuple, Optional, Callable, Union
import numpy as np
from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import gamma  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore


class CondGamma(LeafNode):
    r"""Conditional (univariate) Gamma distribution leaf node in the 'base' backend.

    Represents a conditional univariate Exponential distribution, with the following probability distribution function (PDF):

    .. math::
    
        \text{PDF}(x) = \begin{cases} \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x} & \text{if } x > 0\\
                                      0 & \text{if } x <= 0\end{cases}

    where
        - :math:`x` is the input observation
        - :math:`\Gamma` is the Gamma function
        - :math:`\alpha` is the shape parameter
        - :math:`\beta` is the rate parameter
 
    Attributes:
        cond_f:
            Optional callable to retrieve the conditional parameter for the leaf node.
            Its output should be a dictionary containing 'alpha','beta' as keys, and the values should be
            floating point values, greater than 0, representing the shape and rate parameters, respectively.
    """
    def __init__(self, scope: Scope, cond_f: Optional[Callable]=None) -> None:
        r"""Initializes ``CondExponential`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``alpha``,``beta`` as keys, and the values should be
                floating point values, greater than 0, representing the shape and rate parameters, respectively.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'CondGamma' should be 1, but was {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for 'CondGamma' should be empty, but was {scope.evidence}.")

        super(CondGamma, self).__init__(scope=scope)

        self.set_cond_f(cond_f)

    def set_cond_f(self, cond_f: Optional[Callable]=None) -> None:
        r"""Sets the function to retrieve the node's conditonal parameter.

        Args:
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``alpha``,``beta`` as keys, and the values should be
                floating point values, greater than 0, representing the shape and rate parameters, respectively.
        """
        self.cond_f = cond_f

    def retrieve_params(self, data: np.ndarray, dispatch_ctx: DispatchContext) -> Tuple[Union[np.ndarray, float],Union[np.ndarray, float]]:
        r"""Retrieves the conditional parameter of the leaf node.
    
        First, checks if conditional parameters (``alpha``,``beta``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameters.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameters.

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
        alpha, beta, cond_f = None, None, None

        # check dispatch cache for required conditional parameters 'alpha', 'beta'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if values for 'alpha', 'beta' are specified (highest priority)
            if "alpha" in args:
                alpha = args["alpha"]
            if "beta" in args:
                beta = args["beta"]
            # check if alternative function to provide 'alpha', 'beta' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'l' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'alpha' or 'beta' nor 'cond_f' is specified (via node or arguments)
        if (alpha is None or beta is None) and cond_f is None:
            raise ValueError("'CondGamma' requires either 'alpha' and 'beta' or 'cond_f' to retrieve 'alpha', 'beta' to be specified.")

        # if 'alpha' or 'beta' not already specified, retrieve them
        if alpha is None or beta is None:
            params = cond_f(data)
            alpha = params['alpha']
            beta = params['beta']

        # check if values for 'alpha', 'beta' are valid
        if alpha <= 0.0 or not np.isfinite(alpha):
            raise ValueError(
                f"Value of alpha for conditional Gamma distribution must be greater than 0, but was: {alpha}"
            )
        if beta <= 0.0 or not np.isfinite(beta):
            raise ValueError(
                f"Value of beta for conditional Gamma distribution must be greater than 0, but was: {beta}"
            )
        
        return alpha, beta

    def dist(self, alpha: float, beta: float) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.
        
        Args:
            alpha:
                Floating point value representing the shape parameter (:math:`\alpha`), greater than 0.
            beta:
                Floating point value representing the rate parameter (:math:`\beta`), greater than 0.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return gamma(a=alpha, scale=1.0/beta)

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Gamma distribution, which is:

        .. math::

            \text{supp}(\text{Gamma})=(0,+\infty)

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

        # check if values are in valid range
        valid[valid & ~nan_mask] &= (scope_data[valid & ~nan_mask] > 0)

        return valid