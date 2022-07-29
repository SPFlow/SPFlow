"""
Created on November 06, 2021

@authors: Philipp Deibert
"""

import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter
from typing import List, Tuple, Optional
from .parametric import TorchParametricLeaf, proj_bounded_to_real, proj_real_to_bounded
from spflow.base.structure.nodes.leaves.parametric.statistical_types import ParametricType
from spflow.base.structure.nodes.leaves.parametric import Poisson

from multipledispatch import dispatch  # type: ignore


class TorchPoisson(TorchParametricLeaf):
    r"""(Univariate) Poisson distribution.

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

    ptype = ParametricType.COUNT

    def __init__(self, scope: List[int], l: Optional[float]=1.0) -> None:

        if len(scope) != 1:
            raise ValueError(f"Scope size for TorchPoisson should be 1, but was: {len(scope)}")

        super(TorchPoisson, self).__init__(scope)

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
                f"Value of l for TorchPoisson distribution must be finite, but was: {l}"
            )

        if l < 0:
            raise ValueError(
                f"Value of l for TorchPoisson distribution must be non-negative, but was: {l}"
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

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scope):
            raise ValueError(
                f"Expected scope_data to be of shape (n,{len(self.scope)}), but was: {scope_data.shape}"
            )

        valid = self.dist.support.check(scope_data)  # type: ignore

        # check if all values are valid integers
        # TODO: runtime warning due to nan values
        mask = valid.clone()
        valid[mask] &= np.remainder(scope_data[mask], 1) == 0

        # check for infinite values
        mask = valid.clone()
        valid[mask] &= ~scope_data[mask].isinf().sum(dim=-1).bool()

        return valid


@dispatch(Poisson)  # type: ignore[no-redef]
def toTorch(node: Poisson) -> TorchPoisson:
    return TorchPoisson(node.scope, *node.get_params())


@dispatch(TorchPoisson)  # type: ignore[no-redef]
def toNodes(torch_node: TorchPoisson) -> Poisson:
    return Poisson(torch_node.scope, *torch_node.get_params())
