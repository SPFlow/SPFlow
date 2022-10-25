# -*- coding: utf-8 -*-
"""Contains conditional Log-Normal leaf node for SPFlow in the 'base' backend.
"""
from typing import Optional, Tuple, Callable, Union
import numpy as np
from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import lognorm  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore


class CondLogNormal(LeafNode):
    r"""Conditional (univariate) Log-Normal distribution leaf node in the 'base' backend.

    Represents a conditional univariate Log-Normal distribution, with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{x\sigma\sqrt{2\pi}}\exp\left(-\frac{(\ln(x)-\mu)^2}{2\sigma^2}\right)

    where
        - :math:`x` is an observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

    Attributes:
        cond_f:
            Optional callable to retrieve the conditional parameter for the leaf node.
            Its output should be a dictionary containing ``mean``,``std`` as keys, and the values should be
            floating point values, where the latter should be greater than 0.
    """
    def __init__(self, scope: Scope, cond_f: Optional[Callable]=None) -> None:
        r"""Initializes ``CondLogNormal`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``mean``,``std`` as keys, and the values should be
                floating point values, where the latter should be greater than 0.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for LogNormal should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for LogNormal should be empty, but was {scope.evidence}.")

        super(CondLogNormal, self).__init__(scope=scope)
        
        self.set_cond_f(cond_f)

    def set_cond_f(self, cond_f: Optional[Callable]=None) -> None:
        r"""Sets the function to retrieve the node's conditonal parameter.

        Args:
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``mean``,``std`` as keys, and the values should be
                floating point values, where the latter should be greater than 0.
        """
        self.cond_f = cond_f

    def retrieve_params(self, data: np.ndarray, dispatch_ctx: DispatchContext) -> Tuple[Union[np.ndarray, float],Union[np.ndarray, float]]:
        r"""Retrieves the conditional parameter of the leaf node.
    
        First, checks if conditional parameters (``mean``,``std``) is passed as an additional argument in the dispatch context.
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
        mean, std, cond_f = None, None, None

        # check dispatch cache for required conditional parameters 'mean', 'std'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if values for 'mean', 'std' are specified (highest priority)
            if "mean" in args:
                mean = args["mean"]
            if "std" in args:
                std = args["std"]
            # check if alternative function to provide 'mean', 'std' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'mean','std' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'mean' or 'std' nor 'cond_f' is specified (via node or arguments)
        if (mean is None or std is None) and cond_f is None:
            raise ValueError("'CondLogNormal' requires either 'alpha' and 'beta' or 'cond_f' to retrieve 'alpha', 'beta' to be specified.")

        # if 'mean' or 'std' not already specified, retrieve them
        if mean is None or std is None:
            params = cond_f(data)
            mean = params['mean']
            std = params['std']

        # check if values for 'mean', 'std' are valid
        if not (np.isfinite(mean) and np.isfinite(std)):
            raise ValueError(
                f"Values for 'mean' and 'std' for 'CondLogNormal' must be finite, but were: {mean}, {std}"
            )
        if std <= 0.0:
            raise ValueError(
                f"Value for 'std' for 'CondLogNormal' must be greater than 0.0, but was: {std}"
            )

        return mean, std

    def dist(self, mean: float, std: float) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.
        
        Args:
            mean:
                Floating point value representing the mean (:math:`\mu`) of the distribution.
            std:
                Floating point values representing the standard deviation (:math:`\sigma`) of the distribution (must be greater than 0).

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return lognorm(loc=0.0, scale=np.exp(mean), s=std)

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Log-Normal distribution, which is:

        .. math::

            \text{supp}(\text{LogNormal})=(0,\infty)

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