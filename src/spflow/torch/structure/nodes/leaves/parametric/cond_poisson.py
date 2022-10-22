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
from spflow.base.structure.nodes.leaves.parametric.cond_poisson import CondPoisson as BaseCondPoisson


class CondPoisson(LeafNode):
    r"""Conditional (univariate) Poisson distribution for Torch backend.

    .. math::

        \text{PMF}(k) = \lambda^k\frac{e^{-\lambda}}{k!}

    where
        - :math:`k` is the number of occurrences
        - :math:`\lambda` is the rate parameter

    Args:
        scope:
            List of integers specifying the variable scope.
        cond_f:
            TODO
    """
    def __init__(self, scope: Scope, cond_f: Optional[Callable]=None) -> None:

        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for Poisson should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for Poisson should be empty, but was {scope.evidence}.")

        super(CondPoisson, self).__init__(scope=scope)

        self.set_cond_f(cond_f)

    def set_cond_f(self, cond_f: Optional[Callable]=None) -> None:
        self.cond_f = cond_f

    def dist(self, l: torch.Tensor) -> D.Distribution:
        return D.Poisson(rate=l)
    
    def retrieve_params(self, data: torch.Tensor, dispatch_ctx: DispatchContext) -> torch.Tensor:
        
        l, cond_f = None, None

        # check dispatch cache for required conditional parameter 'l'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if a value for 'l' is specified (highest priority)
            if "l" in args:
                l = args["l"]
            # check if alternative function to provide 'l' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'l' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'l' nor 'cond_f' is specified (via node or arguments)
        if l is None and cond_f is None:
            raise ValueError("'CondExponential' requires either 'l' or 'cond_f' to retrieve 'l' to be specified.")

        # if 'l' was not already specified, retrieve it
        if l is None:
            l = cond_f(data)['l']
    
        if isinstance(l, int) or isinstance(l, float):
            l = torch.tensor(l)

        # check if value for 'l' is valid
        if not torch.isfinite(l):
            raise ValueError(f"Value of l for conditional Poisson distribution must be finite, but was: {l}")

        if l < 0:
            raise ValueError(
                f"Value of l for conditional Poisson distribution must be non-negative, but was: {l}"
            )
        
        return l

    def get_params(self) -> Tuple:
        return tuple([])

    def check_support(self, scope_data: torch.Tensor) -> torch.Tensor:
        r"""Checks if instances are part of the support of the Poisson distribution.

        .. math::

            \text{supp}(\text{Poisson})=\mathbb{N}\cup\{0\}
        
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
        valid[~nan_mask] = self.dist(torch.tensor(1.0)).support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore

        # check if all values are valid integers
        valid[~nan_mask & valid] &= torch.remainder(scope_data[~nan_mask & valid], torch.tensor(1)).squeeze(-1) == 0

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf().squeeze(-1)

        return valid


@dispatch(memoize=True)
def toTorch(node: BaseCondPoisson, dispatch_ctx: Optional[DispatchContext]=None) -> CondPoisson:
    return CondPoisson(node.scope)


@dispatch(memoize=True)
def toBase(torch_node: CondPoisson, dispatch_ctx: Optional[DispatchContext]=None) -> BaseCondPoisson:
    return BaseCondPoisson(torch_node.scope)