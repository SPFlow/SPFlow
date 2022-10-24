"""
Created on October 20, 2022

@authors: Philipp Deibert
"""
import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter
from typing import List, Tuple, Optional, Callable
from spflow.meta.scope.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.structure.nodes.node import LeafNode
from spflow.base.structure.nodes.leaves.parametric.cond_log_normal import CondLogNormal as BaseCondLogNormal


class CondLogNormal(LeafNode):
    r"""Conditional (univariate) Log-Normal distribution for Torch backend.

    .. math::

        \text{PDF}(x) = \frac{1}{x\sigma\sqrt{2\pi}}\exp\left(-\frac{(\ln(x)-\mu)^2}{2\sigma^2}\right)

    where
        - :math:`x` is an observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

    Args:
        scope:
            List of integers specifying the variable scope.
        cond_f:
            TODO
    """
    def __init__(self, scope: Scope, cond_f: Optional[Callable]=None) -> None:

        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for CondLogNormal should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for CondLogNormal should be empty, but was {scope.evidence}.")

        super(CondLogNormal, self).__init__(scope=scope)

        self.set_cond_f(cond_f)

    def set_cond_f(self, cond_f: Optional[Callable]=None) -> None:
        self.cond_f = cond_f

    def dist(self, mean: torch.Tensor, std: torch.Tensor) -> D.Distribution:
        return D.LogNormal(loc=mean, scale=std)

    def retrieve_params(self, data: torch.Tensor, dispatch_ctx: DispatchContext) -> Tuple[torch.Tensor, torch.Tensor]:

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
        
        if isinstance(mean, float):
            mean = torch.tensor(mean)
        if isinstance(std, float):
            std = torch.tensor(std)

        # check if values for 'mean', 'std' are valid
        if not (torch.isfinite(mean) and torch.isfinite(std)):
            raise ValueError(
                f"Mean and standard deviation for Log Normal distribution must be finite, but were: {mean}, {std}"
            )
        if std <= 0.0:
            raise ValueError(
                f"Standard deviation for conditional Log Normal distribution must be greater than 0.0, but was: {std}"
            )

        return mean, std
    
    def get_params(self) -> Tuple:
        return tuple([])

    def check_support(self, scope_data: torch.Tensor) -> torch.Tensor:
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

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        valid = torch.ones(scope_data.shape[0], 1, dtype=torch.bool)
        valid[~nan_mask] = self.dist(torch.tensor(0.0), torch.tensor(1.0)).support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf().squeeze(-1)

        return valid


@dispatch(memoize=True)
def toTorch(node: BaseCondLogNormal, dispatch_ctx: Optional[DispatchContext]=None) -> CondLogNormal:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CondLogNormal(node.scope, *node.get_params())


@dispatch(memoize=True)
def toBase(torch_node: CondLogNormal, dispatch_ctx: Optional[DispatchContext]=None) -> BaseCondLogNormal:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCondLogNormal(torch_node.scope, *torch_node.get_params())