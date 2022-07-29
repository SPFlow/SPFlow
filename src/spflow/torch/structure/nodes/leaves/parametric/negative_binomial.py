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
            Probability of success for each trial in the range :math:`[0,1]` (default 0.5).
    """

    ptype = ParametricType.COUNT

    def __init__(self, scope: List[int], n: int, p: Optional[float]=0.5) -> None:

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

    def set_params(self, n: int, p: float) -> None:

        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for TorchNegativeBinomial distribution must to be between 0.0 and 1.0, but was: {p}"
            )
        if n < 0 or not np.isfinite(n):
            raise ValueError(
                f"Value of n for TorchNegativeBinomial distribution must to greater of equal to 0, but was: {n}"
            )

        if not (torch.remainder(torch.tensor(n), 1.0) == 0.0):
            raise ValueError(
                f"Value of n for TorchNegativeBinomial distribution must be (equal to) an integer value, but was: {n}"
            )

        self.n.data = torch.tensor(int(n))  # type: ignore
        self.p_aux.data = proj_bounded_to_real(torch.tensor(float(p)), lb=0.0, ub=1.0)  # type: ignore

    def get_params(self) -> Tuple[int, float]:
        return self.n.data.cpu().numpy(), self.p.data.cpu().numpy()  # type: ignore

    def check_support(self, scope_data: torch.Tensor) -> torch.Tensor:
        r"""Checks if instances are part of the support of the NegativeBinomial distribution.

        .. math::

            \text{supp}(\text{NegativeBinomial})=\mathbb{N}\cup\{0\}

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


@dispatch(NegativeBinomial)  # type: ignore[no-redef]
def toTorch(node: NegativeBinomial) -> TorchNegativeBinomial:
    return TorchNegativeBinomial(node.scope, *node.get_params())


@dispatch(TorchNegativeBinomial)  # type: ignore[no-redef]
def toNodes(torch_node: TorchNegativeBinomial) -> NegativeBinomial:
    return NegativeBinomial(torch_node.scope, *torch_node.get_params())
