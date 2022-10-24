"""
Created on October 20, 2022

@authors: Philipp Deibert
"""
import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter
from typing import List, Tuple, Optional, Callable
from .projections import proj_bounded_to_real, proj_real_to_bounded
from spflow.meta.scope.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.structure.nodes.node import LeafNode
from spflow.base.structure.nodes.leaves.parametric.cond_binomial import CondBinomial as BaseCondBinomial


class CondBinomial(LeafNode):
    r"""Conditional (univariate) Binomial distribution for Torch backend.

    .. math::

        \text{PMF}(k) = \binom{n}{k}p^k(1-p)^{n-k}

    where
        - :math:`p` is the success probability of each trial
        - :math:`n` is the number of total trials
        - :math:`k` is the number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Args:
        scope:
            List of integers specifying the variable scope.
        n:
            Number of i.i.d. Bernoulli trials (greater of equal to 0).
        cond_f:
            TODO
    """
    def __init__(self, scope: Scope, n: int, cond_f: Optional[Callable]=None) -> None:

        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for CondBinomial should be 1, but was {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for CondBinomial should be empty, but was {scope.evidence}.")

        super(CondBinomial, self).__init__(scope=scope)

        # register number of trials n as torch buffer (should not be changed)
        self.register_buffer("n", torch.empty(size=[]))

        # set parameters
        self.set_params(n)
        
        self.set_cond_f(cond_f)

    def set_cond_f(self, cond_f: Optional[Callable]=None) -> None:
        self.cond_f = cond_f

    def retrieve_params(self, data: torch.Tensor, dispatch_ctx: DispatchContext) -> Tuple[torch.Tensor]:
        
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
            raise ValueError("'CondBernoulli' requires either 'p' or 'cond_f' to retrieve 'p' to be specified.")

        # if 'p' was not already specified, retrieve it
        if p is None:
            p = cond_f(data)['p']

        if isinstance(p, float):
            p = torch.tensor(p)

        # check if value for 'p' is valid
        if p < 0.0 or p > 1.0 or not torch.isfinite(p):
            raise ValueError(
                f"Value of p for CondBinomial distribution must to be between 0.0 and 1.0, but was: {p}"
            )
        
        return p

    def get_params(self) -> Tuple[int]:
        return (self.n.data.cpu().numpy(),)  # type: ignore

    def dist(self, p: torch.Tensor) -> D.Distribution:
        return D.Binomial(total_count=self.n, probs=p)

    def set_params(self, n: int) -> None:
    
        if isinstance(n, float):
            if not n.is_integer():
                raise ValueError(
                    f"Value of n for CondBinomial distribution must be (equal to) an integer value, but was: {n}"
                )
            n = torch.tensor(int(n))
        elif isinstance(n, int):
            n = torch.tensor(n)
        if n < 0 or not torch.isfinite(n):
            raise ValueError(
                f"Value of n for CondBinomial distribution must to greater of equal to 0, but was: {n}"
            )

        self.n.data = torch.tensor(int(n))  # type: ignore

    def check_support(self, scope_data: torch.Tensor) -> torch.Tensor:
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
        
        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        valid = torch.ones(scope_data.shape[0], 1, dtype=torch.bool)
        valid[~nan_mask] = self.dist(p=torch.tensor(0.0)).support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf().squeeze(-1)

        return valid


@dispatch(memoize=True)
def toTorch(node: BaseCondBinomial, dispatch_ctx: Optional[DispatchContext]=None) -> CondBinomial:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CondBinomial(node.scope, *node.get_params())


@dispatch(memoize=True)
def toBase(torch_node: CondBinomial, dispatch_ctx: Optional[DispatchContext]=None) -> BaseCondBinomial:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCondBinomial(torch_node.scope, *torch_node.get_params())