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
from spflow.base.structure.nodes.leaves.parametric import Binomial

from multipledispatch import dispatch  # type: ignore


class TorchBinomial(TorchParametricLeaf):
    r"""(Univariate) Binomial distribution.

    .. math::

        \text{PMF}(k) = \binom{n}{k}p^k(1-p)^{n-k}

    where
        - :math:`p` is the success probability of each trial
        - :math:`n` is the number of total trials
        - :math:`k` is the number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Args:
        scope:
            List of integers specifying the variable scope.
        n:
            Number of i.i.d. Bernoulli trials (greater of equal to 0).
        p:
            Probability of success of each trial in the range :math:`[0,1]` (default 0.5).
    """

    ptype = ParametricType.COUNT

    def __init__(self, scope: List[int], n: int, p: Optional[float]=0.5) -> None:

        if len(scope) != 1:
            raise ValueError(f"Scope size for TorchBinomial should be 1, but was: {len(scope)}")

        super(TorchBinomial, self).__init__(scope)

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

    @p.setter
    def p(self, p: float) -> None:

        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for TorchBinomial distribution must to be between 0.0 and 1.0, but was: {p}"
            )

        self.p_aux.data = proj_bounded_to_real(torch.tensor(float(p)), lb=0.0, ub=1.0)  # type: ignore

    @property
    def dist(self) -> D.Distribution:
        return D.Binomial(total_count=self.n, probs=self.p)

    def set_params(self, n: int, p: float) -> None:

        if n < 0 or not np.isfinite(n):
            raise ValueError(
                f"Value of n for TorchBinomial distribution must to greater of equal to 0, but was: {n}"
            )

        if not (torch.remainder(torch.tensor(n), 1.0) == torch.tensor(0.0)):
            raise ValueError(
                f"Value of n for TorchBinomial distribution must be (equal to) an integer value, but was: {n}"
            )

        self.p = torch.tensor(float(p))
        self.n.data = torch.tensor(int(n))  # type: ignore

    def get_params(self) -> Tuple[int, float]:
        return self.n.data.cpu().numpy(), self.p.data.cpu().numpy()  # type: ignore

    def check_support(self, scope_data: torch.Tensor) -> torch.Tensor:
        r"""Checks if instances are part of the support of the Binomial distribution.

        .. math::

            \text{supp}(\text{Binomial})=\{0,\hdots,n\}

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


@dispatch(Binomial)  # type: ignore[no-redef]
def toTorch(node: Binomial) -> TorchBinomial:
    return TorchBinomial(node.scope, *node.get_params())


@dispatch(TorchBinomial)  # type: ignore[no-redef]
def toNodes(torch_node: TorchBinomial) -> Binomial:
    return Binomial(torch_node.scope, *torch_node.get_params())