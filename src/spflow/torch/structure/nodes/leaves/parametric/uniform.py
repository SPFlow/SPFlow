"""
Created on November 06, 2021

@authors: Philipp Deibert
"""

import numpy as np
import torch
import torch.distributions as D
from typing import List, Tuple
from .parametric import TorchParametricLeaf
from spflow.base.structure.nodes.leaves.parametric.statistical_types import ParametricType
from spflow.base.structure.nodes.leaves.parametric import Uniform

from multipledispatch import dispatch  # type: ignore


class TorchUniform(TorchParametricLeaf):
    """(Univariate) continuous Uniform distribution.
    PDF(x) =
        1 / (end - start) * 1_[start, end], where
            - 1_[start, end] is the indicator function of the given interval (evaluating to 0 if x is not in the interval)
    Attributes:
        start:
            Start of the interval.
        end:
            End of interval (must be larger the interval start).
    """

    ptype = ParametricType.CONTINUOUS

    def __init__(self, scope: List[int], start: float, end: float) -> None:

        if len(scope) != 1:
            raise ValueError(f"Scope size for TorchUniform should be 1, but was: {len(scope)}")

        super(TorchUniform, self).__init__(scope)

        # register interval bounds as torch buffers (should not be changed)
        self.register_buffer("start", torch.empty(size=[]))
        self.register_buffer("end", torch.empty(size=[]))

        # set parameters
        self.set_params(start, end)

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
        support_mask = ((scope_data >= self.start) & (scope_data <= self.end)).sum(dim=1).bool()
        log_prob[prob_mask & (~support_mask)] = -float("Inf")
        # compute probabilities for values inside distribution support
        log_prob[prob_mask & support_mask] = self.dist.log_prob(
            scope_data[prob_mask & support_mask]
        )

        return log_prob

    def set_params(self, start: float, end: float) -> None:

        if not start < end:
            raise ValueError(
                f"Lower bound for Uniform distribution must be less than upper bound, but were: {start}, {end}"
            )
        if not (np.isfinite(start) and np.isfinite(end)):
            raise ValueError(f"Lower and upper bound must be finite, but were: {start}, {end}")

        # since torch Uniform distribution excludes the upper bound, compute next largest number
        end_next = torch.nextafter(torch.tensor(end), torch.tensor(float("Inf")))  # type: ignore

        self.start.data = torch.tensor(float(start))  # type: ignore
        self.end.data = torch.tensor(float(end_next))  # type: ignore

        # create Torch distribution with specified parameters
        self.dist = D.Uniform(low=self.start, high=end_next)

    def get_params(self) -> Tuple[float, float]:
        return self.start.cpu().numpy(), self.end.cpu().numpy()  # type: ignore


@dispatch(Uniform)  # type: ignore[no-redef]
def toTorch(node: Uniform) -> TorchUniform:
    return TorchUniform(node.scope, node.start, node.end)


@dispatch(TorchUniform)  # type: ignore[no-redef]
def toNodes(torch_node: TorchUniform) -> Uniform:
    return Uniform(torch_node.scope, torch_node.start.cpu().numpy(), torch_node.end.cpu().numpy())  # type: ignore
