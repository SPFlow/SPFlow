"""
Created on October 18, 2022

@authors: Philipp Deibert
"""
from typing import Tuple, Optional, Callable, Union
import numpy as np
from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import norm
from scipy.stats.distributions import rv_frozen


class CondGaussian(LeafNode):
    r"""Conditional (univariate) Normal distribution.

    .. math::

        \text{PDF}(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})

    where
        - :math:`x` the observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

    Args:
        scope:
            Scope object specifying the variable scope.
        cond_f:
            Callable that provides the conditional parameters (mean, std) of this distribution. TODO
    """
    def __init__(
        self,
        scope: Scope,
        cond_f: Optional[Callable]=None,
    ) -> None:

        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for conditional Gaussian should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for conditional Gaussian should be empty, but was {scope.evidence}.")

        super(CondGaussian, self).__init__(scope=scope)

        self.set_cond_f(cond_f)

    def set_cond_f(self, cond_f: Optional[Callable]=None) -> None:
        self.cond_f = cond_f

    def retrieve_params(self, data: np.ndarray, dispatch_ctx: DispatchContext) -> Tuple[Union[np.ndarray, float],Union[np.ndarray, float]]:
        
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
            raise ValueError("'CondExponential' requires either 'mean' and 'std' or 'cond_f' to retrieve 'mean', 'std' to be specified.")

        # if 'mean' or 'std' not already specified, retrieve them
        if mean is None or std is None:
            params = cond_f(data)
            mean = params['mean']
            std = params['std']

        # check if values for 'mean', 'std' are valid
        if not (np.isfinite(mean) and np.isfinite(std)):
            raise ValueError(
                f"Mean and standard deviation for Gaussian distribution must be finite, but were: {mean}, {std}"
            )
        if std <= 0.0:
            raise ValueError(
                f"Standard deviation for Gaussian distribution must be greater than 0.0, but was: {std}"
            )

        return mean, std

    def dist(self, mean: float, std: float) -> rv_frozen:
        return norm(loc=mean, scale=std)

    def get_params(self) -> Tuple:
        return tuple([])

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if instances are part of the support of the Gaussian distribution.

        .. math::

            \text{supp}(\text{Gaussian})=(-\infty,+\infty)

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

        return valid