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
    """(Univariate) Binomial distribution.
    PMF(k) =
        p   , if k=1
        1-p , if k=0
    Attributes:
        p:
            Probability of success in the range [0,1].
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
                f"Value of p for Bernoulli distribution must to be between 0.0 and 1.0, but was: {p}"
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

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0.0

        # ----- log probabilities -----

        # create Torch distribution with specified parameters
        dist = D.Bernoulli(probs=self.p)

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        # set probabilities of values outside of distribution support to 0 (-inf in log space)
        support_mask = ((scope_data == 1) | (scope_data == 0)).sum(dim=1).bool()
        log_prob[prob_mask & (~support_mask)] = -float("Inf")
        # compute probabilities for values inside distribution support
        log_prob[prob_mask & support_mask] = dist.log_prob(
            scope_data[prob_mask & support_mask].type(torch.get_default_dtype())
        )

        return log_prob

    def set_params(self, p: float) -> None:
        self.p = p

    def get_params(self) -> Tuple[float]:
        return (self.p.data.cpu().numpy(),)  # type: ignore


@dispatch(Bernoulli)  # type: ignore[no-redef]
def toTorch(node: Bernoulli) -> TorchBernoulli:
    return TorchBernoulli(node.scope, *node.get_params())


@dispatch(TorchBernoulli)  # type: ignore[no-redef]
def toNodes(torch_node: TorchBernoulli) -> Bernoulli:
    return Bernoulli(torch_node.scope, *torch_node.get_params())
