"""
Created on November 06, 2021

@authors: Philipp Deibert
"""

import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter
from typing import List, Tuple
from .parametric import TorchParametricLeaf, proj_bounded_to_real, proj_real_to_bounded
from spflow.base.structure.nodes.leaves.parametric.statistical_types import ParametricType
from spflow.base.structure.nodes.leaves.parametric import Bernoulli

from multipledispatch import dispatch  # type: ignore


class TorchBernoulli(TorchParametricLeaf):
    r"""(Univariate) Bernoulli distribution.

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
            Probability of success in the range :math:`[0,1]`.
    """

    ptype = ParametricType.BINARY

    def __init__(self, scope: List[int], p: float) -> None:

        if len(scope) != 1:
            raise ValueError(f"Scope size for TorchBernoulli should be 1, but was: {len(scope)}")

        super(TorchBernoulli, self).__init__(scope)

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
                f"Value of p for TorchBernoulli distribution must to be between 0.0 and 1.0, but was: {p}"
            )

        self.p_aux.data = proj_bounded_to_real(torch.tensor(float(p)), lb=0.0, ub=1.0)

    @property
    def dist(self) -> D.Distribution:
        return D.Bernoulli(probs=self.p)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1)

        # ----- marginalization -----

        marg_ids = torch.isnan(scope_data).sum(dim=1) == len(self.scope)

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[marg_ids] = 0.0

        # ----- log probabilities -----

        # create masked based on distribution's support
        valid_ids = self.check_support(scope_data[~marg_ids])

        if not all(valid_ids):
            raise ValueError(
                f"Encountered data instances that are not in the support of the TorchBernoulli distribution."
            )

        # compute probabilities for values inside distribution support
        log_prob[~marg_ids] = self.dist.log_prob(
            scope_data[~marg_ids].type(torch.get_default_dtype())
        )

        return log_prob

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

        valid = self.dist.support.check(scope_data)  # type: ignore

        # check for infinite values
        mask = valid.clone()
        valid[mask] &= ~scope_data[mask].isinf().sum(dim=-1).bool()

        return valid


@dispatch(Bernoulli)  # type: ignore[no-redef]
def toTorch(node: Bernoulli) -> TorchBernoulli:
    return TorchBernoulli(node.scope, *node.get_params())


@dispatch(TorchBernoulli)  # type: ignore[no-redef]
def toNodes(torch_node: TorchBernoulli) -> Bernoulli:
    return Bernoulli(torch_node.scope, *torch_node.get_params())
