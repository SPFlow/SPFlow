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
    """(Univariate) Negative Binomial distribution.
    PMF(k) =
        (k+n-1)C(k) * (1-p)^n * p^k, where
            - (n)C(k) is the binomial coefficient (n choose k)
    Attributes:
        n:
            Number of i.i.d. trials (greater of equal to 0).
        p:
            Probability of success of each trial in the range (0,1].
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

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0.0

        # ----- log probabilities -----

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        # set probabilities of values outside of distribution support to 0 (-inf in log space)
        support_mask = (scope_data >= 0).sum(dim=1).bool()
        log_prob[prob_mask & (~support_mask)] = -float("Inf")
        # compute probabilities for values inside distribution support
        log_prob[prob_mask & support_mask] = self.dist.log_prob(
            scope_data[prob_mask & support_mask]
        )

        return log_prob

    def set_params(self, n: int, p: float) -> None:

        if p <= 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for NegativeBinomial distribution must to be between 0.0 and 1.0, but was: {p}"
            )
        if n < 0 or not np.isfinite(n):
            raise ValueError(
                f"Value of n for NegativeBinomial distribution must to greater of equal to 0, but was: {n}"
            )

        self.n.data = torch.tensor(int(n))  # type: ignore
        self.p_aux.data = proj_bounded_to_real(torch.tensor(float(p)), lb=0.0, ub=1.0)  # type: ignore

    def get_params(self) -> Tuple[int, float]:
        return self.n.data.cpu().numpy(), self.p.data.cpu().numpy()  # type: ignore


@dispatch(NegativeBinomial)  # type: ignore[no-redef]
def toTorch(node: NegativeBinomial) -> TorchNegativeBinomial:
    return TorchNegativeBinomial(node.scope, *node.get_params())


@dispatch(TorchNegativeBinomial)  # type: ignore[no-redef]
def toNodes(torch_node: TorchNegativeBinomial) -> NegativeBinomial:
    return NegativeBinomial(torch_node.scope, *torch_node.get_params())
