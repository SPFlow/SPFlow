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
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.structure.nodes.node import LeafNode
from spflow.base.structure.nodes.leaves.parametric.exponential import Exponential as BaseExponential


class Exponential(LeafNode):
    r"""(Univariate) Exponential distribution for Torch backend.

    .. math::
        
        \text{PDF}(x) = \begin{cases} \lambda e^{-\lambda x} & \text{if } x > 0\\
                                      0                      & \text{if } x <= 0\end{cases}
    
    where
        - :math:`x` is the input observation
        - :math:`\lambda` is the rate parameter
    
    Args:
        scope:
            List of integers specifying the variable scope.
        l:
            Rate parameter (:math:`\lambda`) of the Exponential distribution (must be greater than 0; default 1.5).
    """
    def __init__(self, scope: Scope, l: Optional[float]=1.0) -> None:

        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for Exponential should be 1, but was {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for Exponential should be empty, but was {scope.evidence}.")

        super(Exponential, self).__init__(scope=scope)

        # register auxiliary torch parameter for parameter l
        self.l_aux = Parameter()

        # set parameters
        self.set_params(l)

    @property
    def l(self) -> torch.Tensor:
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.l_aux, lb=0.0)  # type: ignore

    @property
    def dist(self) -> D.Distribution:
        return D.Exponential(rate=self.l)

    def set_params(self, l: float) -> None:

        if l <= 0.0 or not np.isfinite(l):
            raise ValueError(
                f"Value of l for Exponential distribution must be greater than 0, but was: {l}"
            )

        self.l_aux.data = proj_bounded_to_real(torch.tensor(float(l)), lb=0.0)

    def get_params(self) -> Tuple[float]:
        return (self.l.data.cpu().numpy(),)  # type: ignore

    def check_support(self, scope_data: torch.Tensor) -> torch.Tensor:
        r"""Checks if instances are part of the support of the Exponential distribution.

        .. math::

            \text{supp}(\text{Exponential})=(0,+\infty)

        Note: for PyTorch version < 1.11.0 zero is not part of the support Exponential, even though it is for Exponential.

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
        valid[~nan_mask] = self.dist.support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf().squeeze(-1)

        return valid


@dispatch(memoize=True)
def toTorch(node: BaseExponential, dispatch_ctx: Optional[DispatchContext]=None) -> Exponential:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return Exponential(node.scope, *node.get_params())


@dispatch(memoize=True)
def toBase(torch_node: Exponential, dispatch_ctx: Optional[DispatchContext]=None) -> BaseExponential:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseExponential(torch_node.scope, *torch_node.get_params())