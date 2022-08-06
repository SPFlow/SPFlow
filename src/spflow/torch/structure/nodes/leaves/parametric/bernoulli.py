"""
Created on November 06, 2021

@authors: Philipp Deibert
"""

import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter
from typing import List, Tuple, Optional
from .projections import proj_bounded_to_real, proj_real_to_bounded
from spflow.meta.scope.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.torch.structure.nodes.node import LeafNode
from spflow.base.structure.nodes.leaves.parametric.bernoulli import Bernoulli as BaseBernoulli


class Bernoulli(LeafNode):
    r"""(Univariate) Bernoulli distribution for Torch backend.

    .. math::

        \text{PMF}(k)=\begin{cases} p   & \text{if } k=1\\
                                    1-p & \text{if } k=0\end{cases}
        
    where
        - :math:`p` is the success probability
        - :math:`k` is the outcome of the trial (0 or 1)

    Args:
        scope:
            List of integers specifying the variable scope.
        p:
            Probability of success in the range :math:`[0,1]` (default 0.5).
    """
    def __init__(self, scope: Scope, p: Optional[float]=0.5) -> None:

        if len(scope.query) != 1:
            raise ValueError(f"Scope size for Bernoulli should be 1, but was: {len(scope.query)}")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for Bernoulli should be empty, but was {scope.evidence}.")

        super(Bernoulli, self).__init__(scope=scope)

        # register auxiliary torch paramter for the success probability p
        self.p_aux = Parameter()

        # set parameters
        self.set_params(p)

    @property
    def p(self) -> torch.Tensor:
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.p_aux, lb=0.0, ub=1.0)  # type: ignore

    @p.setter
    def p(self, p: float) -> None:

        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for Bernoulli distribution must to be between 0.0 and 1.0, but was: {p}"
            )

        self.p_aux.data = proj_bounded_to_real(torch.tensor(float(p)), lb=0.0, ub=1.0)

    @property
    def dist(self) -> D.Distribution:
        return D.Bernoulli(probs=self.p)

    def set_params(self, p: float) -> None:
        self.p = torch.tensor(float(p))

    def get_params(self) -> Tuple[float]:
        return (self.p.data.cpu().numpy(),)  # type: ignore

    def check_support(self, scope_data: torch.Tensor) -> torch.Tensor:
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

        valid = self.dist.support.check(scope_data)  # type: ignore

        # check for infinite values
        mask = valid.clone()
        valid[mask] &= ~scope_data[mask].isinf().sum(dim=-1).bool()

        return valid


@dispatch(memoize=True)
def toTorch(node: BaseBernoulli) -> Bernoulli:
    return Bernoulli(node.scope, *node.get_params())


@dispatch(memoize=True)
def toBase(torch_node: Bernoulli) -> BaseBernoulli:
    return BaseBernoulli(torch_node.scope, *torch_node.get_params())