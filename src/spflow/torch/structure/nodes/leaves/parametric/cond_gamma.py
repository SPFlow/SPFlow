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
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.nodes.node import LeafNode
from spflow.base.structure.nodes.leaves.parametric.cond_gamma import CondGamma as BaseCondGamma


class CondGamma(LeafNode):
    r"""Conditional (univariate) Gamma distribution for Torch backend.
    
    .. math::
    
        \text{PDF}(x) = \begin{cases} \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x} & \text{if } x > 0\\
                                      0 & \text{if } x <= 0\end{cases}

    where
        - :math:`x` is the input observation
        - :math:`\Gamma` is the Gamma function
        - :math:`\alpha` is the shape parameter
        - :math:`\beta` is the rate parameter

    TODO: check
    
    Args:
        scope:
            List of integers specifying the variable scope.
        cond_f:
            TODO
    """
    def __init__(self, scope: Scope, cond_f: Optional[Callable]=None) -> None:

        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for CondGamma should be 1, but was {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for CondGamma should be empty, but was {scope.evidence}.")

        super(CondGamma, self).__init__(scope=scope)

        self.set_cond_f(cond_f)

    def set_cond_f(self, cond_f: Optional[Callable]=None) -> None:
        self.cond_f = cond_f

    def dist(self, alpha: torch.Tensor, beta: torch.Tensor) -> D.Distribution:
        return D.Gamma(concentration=alpha, rate=beta)
    
    def retrieve_params(self, data: torch.Tensor, dispatch_ctx: DispatchContext) -> Tuple[torch.Tensor,torch.Tensor]:
        
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
        
        if isinstance(alpha, float):
            alpha = torch.tensor(alpha)
        if isinstance(beta, float):
            beta = torch.tensor(beta)

        # check if values for 'alpha', 'beta' are valid
        if alpha <= 0.0 or not torch.isfinite(alpha):
            raise ValueError(
                f"Value of alpha for conditional Gamma distribution must be greater than 0, but was: {alpha}"
            )
        if beta <= 0.0 or not torch.isfinite(beta):
            raise ValueError(
                f"Value of beta for conditional Gamma distribution must be greater than 0, but was: {beta}"
            )
        
        return alpha, beta

    def get_params(self) -> Tuple:
        return tuple([])

    def check_support(self, scope_data: torch.Tensor) -> torch.Tensor:
        r"""Checks if instances are part of the support of the Gamma distribution.

        .. math::

            \text{supp}(\text{Gamma})=(0,+\infty)
        
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
        valid[~nan_mask] = self.dist(alpha=torch.tensor(1.0), beta=torch.tensor(1.0)).support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf().squeeze(-1)

        return valid


@dispatch(memoize=True)
def toTorch(node: BaseCondGamma, dispatch_ctx: Optional[DispatchContext]=None) -> CondGamma:
    return CondGamma(node.scope)


@dispatch(memoize=True)
def toBase(torch_node: CondGamma, dispatch_ctx: Optional[DispatchContext]=None) -> BaseCondGamma:
    return BaseCondGamma(torch_node.scope)