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
from spflow.base.structure.nodes.leaves.parametric import Exponential

from multipledispatch import dispatch  # type: ignore


class TorchExponential(TorchParametricLeaf):
    r"""(Univariate) Exponential distribution.

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

    ptype = ParametricType.POSITIVE

    def __init__(self, scope: List[int], l: Optional[float]=1.0) -> None:

        if len(scope) != 1:
            raise ValueError(f"Scope size for TorchExponential should be 1, but was: {len(scope)}")

        super(TorchExponential, self).__init__(scope)

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
                f"Value of l for TorchExponential distribution must be greater than 0, but was: {l}"
            )

        self.l_aux.data = proj_bounded_to_real(torch.tensor(float(l)), lb=0.0)

    def get_params(self) -> Tuple[float]:
        return (self.l.data.cpu().numpy(),)  # type: ignore

    def check_support(self, scope_data: torch.Tensor) -> torch.Tensor:
        r"""Checks if instances are part of the support of the Exponential distribution.

        .. math::

            \text{supp}(\text{Exponential})=(0,+\infty)

        Note: for PyTorch version < 1.11.0 zero is not part of the support TorchExponential, even though it is for Exponential.

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

        # check for infinite values
        mask = valid.clone()
        valid[mask] &= ~scope_data[mask].isinf().sum(dim=-1).bool()

        return valid


@dispatch(Exponential)  # type: ignore[no-redef]
def toTorch(node: Exponential) -> TorchExponential:
    return TorchExponential(node.scope, *node.get_params())


@dispatch(TorchExponential)  # type: ignore[no-redef]
def toNodes(torch_node: TorchExponential) -> Exponential:
    return Exponential(torch_node.scope, *torch_node.get_params())
