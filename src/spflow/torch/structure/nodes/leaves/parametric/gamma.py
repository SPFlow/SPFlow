import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter
from typing import List, Tuple
from .parametric import TorchParametricLeaf, proj_bounded_to_real, proj_real_to_bounded
from spflow.base.structure.nodes.leaves.parametric.statistical_types import ParametricType
from spflow.base.structure.nodes.leaves.parametric import Gamma

from multipledispatch import dispatch  # type: ignore


class TorchGamma(TorchParametricLeaf):
    """(Univariate) Gamma distribution.
    PDF(x) =
        1/(G(beta) * alpha^beta) * x^(beta-1) * exp(-x/alpha)   , if x > 0
        0                                                       , if x <= 0, where
            - G(beta) is the Gamma function
    Attributes:
        alpha:
            Shape parameter, greater than 0.
        beta:
            Scale parameter, greater than 0.
    """

    ptype = ParametricType.POSITIVE

    def __init__(self, scope: List[int], alpha: float, beta: float) -> None:
        super(TorchGamma, self).__init__(scope)

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
        dist = D.Gamma(concentration=self.alpha, rate=self.beta)

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        log_prob[prob_mask] = dist.log_prob(scope_data[prob_mask])

        return log_prob

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


@dispatch(Gamma)  # type: ignore[no-redef]
def toTorch(node: Gamma) -> TorchGamma:
    return TorchGamma(node.scope, *node.get_params())


@dispatch(TorchGamma)  # type: ignore[no-redef]
def toNodes(torch_node: TorchGamma) -> Gamma:
    return Gamma(torch_node.scope, *torch_node.get_params())
