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
from spflow.base.structure.nodes.leaves.parametric import NegativeBinomial

from multipledispatch import dispatch  # type: ignore


class TorchNegativeBinomial(TorchParametricLeaf):
    r"""(Univariate) Negative Binomial distribution.

    .. math::

        \text{PMF}(k) = \binom{k+n-1}{k}(1-p)^n p^k

    where
        - :math:`k` is the number of successes
        - :math:`n` is the maximum number of failures
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Args:
        scope:
            List of integers specifying the variable scope.
        n:
            Number of i.i.d. trials (greater or equal to 0).
        p:
            Probability of success for each trial in the range :math:`(0,1]`.
    """

    ptype = ParametricType.COUNT

    def __init__(self, scope: List[int], n: int, p: float) -> None:

        if len(scope) != 1:
            raise ValueError(
                f"Scope size for TorchNegativeBinomial should be 1, but was: {len(scope)}"
            )

        super(TorchNegativeBinomial, self).__init__(scope)

        # register number of trials n as torch buffer (should not be changed)
        self.register_buffer("n", torch.empty(size=[]))

        # register auxiliary torch parameter for the success probability p
        self.p_aux = Parameter()

        # set parameters
        self.set_params(n, p)

    @property
    def p(self) -> torch.Tensor:
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.p_aux, lb=0.0, ub=1.0)  # type: ignore

    @property
    def dist(self) -> D.Distribution:
        # note: the distribution is not stored as an attribute due to mismatching parameters after gradient updates (gradients don't flow back to p when initializing with 1.0-p)
        return D.NegativeBinomial(total_count=self.n, probs=torch.ones(1) - self.p)

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
                f"Encountered data instances that are not in the support of the TorchNegativeBinomial distribution."
            )

        # compute probabilities for values inside distribution support
        log_prob[~marg_ids] = self.dist.log_prob(
            scope_data[~marg_ids].type(torch.get_default_dtype())
        )

        return log_prob

    def set_params(self, n: int, p: float) -> None:

        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for TorchNegativeBinomial distribution must to be between 0.0 and 1.0, but was: {p}"
            )
        if n < 0 or not np.isfinite(n):
            raise ValueError(
                f"Value of n for TorchNegativeBinomial distribution must to greater of equal to 0, but was: {n}"
            )

        self.n.data = torch.tensor(int(n))  # type: ignore
        self.p_aux.data = proj_bounded_to_real(torch.tensor(float(p)), lb=0.0, ub=1.0)  # type: ignore

    def get_params(self) -> Tuple[int, float]:
        return self.n.data.cpu().numpy(), self.p.data.cpu().numpy()  # type: ignore

    def check_support(self, scope_data: torch.Tensor) -> torch.Tensor:
        return self.dist.support.check(scope_data)  # type: ignore


@dispatch(NegativeBinomial)  # type: ignore[no-redef]
def toTorch(node: NegativeBinomial) -> TorchNegativeBinomial:
    return TorchNegativeBinomial(node.scope, *node.get_params())


@dispatch(TorchNegativeBinomial)  # type: ignore[no-redef]
def toNodes(torch_node: TorchNegativeBinomial) -> NegativeBinomial:
    return NegativeBinomial(torch_node.scope, *torch_node.get_params())
