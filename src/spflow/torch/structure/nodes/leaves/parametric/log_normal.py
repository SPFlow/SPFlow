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
from spflow.base.structure.nodes.leaves.parametric import LogNormal

from multipledispatch import dispatch  # type: ignore


class TorchLogNormal(TorchParametricLeaf):
    r"""(Univariate) Log-Normal distribution.

    .. math::

        \text{PDF}(x) = \frac{1}{x\sigma\sqrt{2\pi}}\exp\left(-\frac{(\ln(x)-\mu)^2}{2\sigma^2}\right)

    where
        - :math:`x` is an observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

    Args:
        scope:
            List of integers specifying the variable scope.
        mean:
            mean (:math:`\mu`) of the distribution (default 0.0).
        stdev:
            standard deviation (:math:`\sigma`) of the distribution (must be greater than 0; default 1.0).
    """

    ptype = ParametricType.POSITIVE

    def __init__(self, scope: List[int], mean: Optional[float]=0.0, stdev: Optional[float]=1.0) -> None:

        if len(scope) != 1:
            raise ValueError(f"Scope size for TorchLogNormal should be 1, but was: {len(scope)}")

        super(TorchLogNormal, self).__init__(scope)

        # register mean as torch parameter
        self.mean = Parameter()
        # register auxiliary torch paramter for standard deviation
        self.stdev_aux = Parameter()

        # set parameters
        self.set_params(mean, stdev)

    @property
    def stdev(self) -> torch.Tensor:
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.stdev_aux, lb=0.0)  # type: ignore

    @property
    def dist(self) -> D.Distribution:
        return D.LogNormal(loc=self.mean, scale=self.stdev)

    def set_params(self, mean: float, stdev: float) -> None:

        if not (np.isfinite(mean) and np.isfinite(stdev)):
            raise ValueError(
                f"Mean and standard deviation for TorchGaussian distribution must be finite, but were: {mean}, {stdev}"
            )
        if stdev <= 0.0:
            raise ValueError(
                f"Standard deviation for TorchGaussian distribution must be greater than 0.0, but was: {stdev}"
            )

        self.mean.data = torch.tensor(float(mean))
        self.stdev_aux.data = proj_bounded_to_real(torch.tensor(float(stdev)), lb=0.0)

    def get_params(self) -> Tuple[float, float]:
        return self.mean.data.cpu().numpy(), self.stdev.data.cpu().numpy()  # type: ignore

    def check_support(self, scope_data: torch.Tensor) -> torch.Tensor:
        r"""Checks if instances are part of the support of the LogNormal distribution.

        .. math::

            \text{supp}(\text{LogNormal})=(0,\infty)

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


@dispatch(LogNormal)  # type: ignore[no-redef]
def toTorch(node: LogNormal) -> TorchLogNormal:
    return TorchLogNormal(node.scope, *node.get_params())


@dispatch(TorchLogNormal)  # type: ignore[no-redef]
def toNodes(torch_node: TorchLogNormal) -> LogNormal:
    return LogNormal(torch_node.scope, *torch_node.get_params())
