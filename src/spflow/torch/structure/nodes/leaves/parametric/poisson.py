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
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.nodes.node import LeafNode
from spflow.base.structure.nodes.leaves.parametric.poisson import Poisson as BasePoisson


class Poisson(LeafNode):
    r"""(Univariate) Poisson distribution for Torch backend.

    .. math::

        \text{PMF}(k) = \lambda^k\frac{e^{-\lambda}}{k!}

    where
        - :math:`k` is the number of occurrences
        - :math:`\lambda` is the rate parameter

    Args:
        scope:
            List of integers specifying the variable scope.
        l:
            Rate parameter (:math:`\lambda`), expected value and variance of the Poisson distribution (must be greater than or equal to 0; default 1.0).
    """
    def __init__(self, scope: Scope, l: Optional[float]=1.0) -> None:

        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for Poisson should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for Poisson should be empty, but was {scope.evidence}.")

        super(Poisson, self).__init__(scope=scope)

        # register auxiliary torch parameter for lambda l
        self.l_aux = Parameter()

        # set parameters
        self.set_params(l)

    @property
    def l(self) -> torch.Tensor:
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.l_aux, lb=0.0)  # type: ignore

    @property
    def dist(self) -> D.Distribution:
        return D.Poisson(rate=self.l)

    def set_params(self, l: float) -> None:

        if not np.isfinite(l):
            raise ValueError(
                f"Value of l for Poisson distribution must be finite, but was: {l}"
            )

        if l < 0:
            raise ValueError(
                f"Value of l for Poisson distribution must be non-negative, but was: {l}"
            )

        self.l_aux.data = proj_bounded_to_real(torch.tensor(float(l)), lb=0.0)

    def get_params(self) -> Tuple[float]:
        return (self.l.data.cpu().numpy(),)  # type: ignore

    def check_support(self, scope_data: torch.Tensor) -> torch.Tensor:
        r"""Checks if instances are part of the support of the Poisson distribution.

        .. math::

            \text{supp}(\text{Poisson})=\mathbb{N}\cup\{0\}

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

        # check if all values are valid integers
        # TODO: runtime warning due to nan values
        mask = valid.clone()
        valid[mask] &= np.remainder(scope_data[mask], 1) == 0

        # check for infinite values
        mask = valid.clone()
        valid[mask] &= ~scope_data[mask].isinf()

        return valid


@dispatch(memoize=True)
def toTorch(node: BasePoisson, dispatch_ctx: Optional[DispatchContext]=None) -> Poisson:
    return Poisson(node.scope, *node.get_params())


@dispatch(memoize=True)
def toBase(torch_node: Poisson, dispatch_ctx: Optional[DispatchContext]=None) -> BasePoisson:
    return BasePoisson(torch_node.scope, *torch_node.get_params())