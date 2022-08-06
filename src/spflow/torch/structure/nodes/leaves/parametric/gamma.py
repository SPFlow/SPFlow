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
from spflow.base.structure.nodes.leaves.parametric.gamma import Gamma as BaseGamma


class Gamma(LeafNode):
    r"""(Univariate) Gamma distribution for Torch backend.
    
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
        alpha:
            Shape parameter (:math:`\alpha`), greater than 0 (default 1.0).
        beta:
            Rate parameter (:math:`\beta`), greater than 0 (default 1.0).
    """
    def __init__(self, scope: Scope, alpha: Optional[float]=1.0, beta: Optional[float]=1.0) -> None:

        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for Gamma should be 1, but was {len(scope.query)}.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for Gamma should be empty, but was {scope.evidence}.")

        super(Gamma, self).__init__(scope=scope)

        # register auxiliary torch parameters for alpha and beta
        self.alpha_aux = Parameter()
        self.beta_aux = Parameter()

        # set parameters
        self.set_params(alpha, beta)

    @property
    def alpha(self) -> torch.Tensor:
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.alpha_aux, lb=0.0)  # type: ignore

    @property
    def beta(self) -> torch.Tensor:
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.beta_aux, lb=0.0)  # type: ignore

    @property
    def dist(self) -> D.Distribution:
        return D.Gamma(concentration=self.alpha, rate=self.beta)

    def set_params(self, alpha: float, beta: float) -> None:

        if alpha <= 0.0 or not np.isfinite(alpha):
            raise ValueError(
                f"Value of alpha for Gamma distribution must be greater than 0, but was: {alpha}"
            )
        if beta <= 0.0 or not np.isfinite(beta):
            raise ValueError(
                f"Value of beta for Gamma distribution must be greater than 0, but was: {beta}"
            )

        self.alpha_aux.data = proj_bounded_to_real(torch.tensor(float(alpha)), lb=0.0)
        self.beta_aux.data = proj_bounded_to_real(torch.tensor(float(beta)), lb=0.0)

    def get_params(self) -> Tuple[float, float]:
        return self.alpha.data.cpu().numpy(), self.beta.data.cpu().numpy()  # type: ignore

    def check_support(self, scope_data: torch.Tensor) -> torch.Tensor:
        r"""Checks if instances are part of the support of the Gamma distribution.

        .. math::

            \text{supp}(\text{Gamma})=(0,+\infty)

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
def toTorch(node: BaseGamma) -> Gamma:
    return Gamma(node.scope, *node.get_params())


@dispatch(memoize=True)
def toBase(torch_node: Gamma) -> BaseGamma:
    return BaseGamma(torch_node.scope, *torch_node.get_params())