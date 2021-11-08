import numpy as np
import torch
from typing import List, Tuple
from .parametric import TorchParametricLeaf
from spflow.base.structure.nodes.leaves.parametric.statistical_types import ParametricType
from spflow.base.structure.nodes.leaves.parametric import Hypergeometric

from multipledispatch import dispatch  # type: ignore


class TorchHypergeometric(TorchParametricLeaf):
    """(Univariate) Hypergeometric distribution.
    PMF(k) =
        (M)C(k) * (N-M)C(n-k) / (N)C(n), where
            - (n)C(k) is the binomial coefficient (n choose k)
    Attributes:
        N:
            Total number of entities (in the population), greater or equal to 0.
        M:
            Number of entities with property of interest (in the population), greater or equal to zero and less than or equal to N.
        n:
            Number of observed entities (sample size), greater or equal to zero and less than or equal to N.
    """

    ptype = ParametricType.COUNT

    def __init__(self, scope: List[int], N: int, M: int, n: int) -> None:

        if len(scope) != 1:
            raise ValueError(
                f"Scope size for TorchHypergeometric should be 1, but was: {len(scope)}"
            )

        super(TorchHypergeometric, self).__init__(scope)

        # register parameters as torch buffers (should not be changed)
        self.register_buffer("N", torch.empty(size=[]))
        self.register_buffer("M", torch.empty(size=[]))
        self.register_buffer("n", torch.empty(size=[]))

        # set parameters
        self.set_params(N, M, n)

    def log_prob(self, k: torch.Tensor):

        N_minus_M = self.N - self.M  # type: ignore
        n_minus_k = self.n - k  # type: ignore

        # ----- (M over m) * (N-M over n-k) / (N over n) -----

        # log_M_over_k = torch.lgamma(self.M+1) - torch.lgamma(self.M-k+1) - torch.lgamma(k+1)
        # log_NM_over_nk = torch.lgamma(N_minus_M+1) - torch.lgamma(N_minus_M-n_minus_k+1) - torch.lgamma(n_minus_k+1)
        # log_N_over_n = torch.lgamma(self.N+1) - torch.lgamma(self.N-self.n+1) - torch.lgamma(self.n+1)
        # result = log_M_over_k + log_NM_over_nk - log_N_over_n

        # ---- alternatively (more precise according to SciPy) -----
        # betaln(good+1, 1) + betaln(bad+1,1) + betaln(total-draws+1, draws+1) - betaln(k+1, good-k+1) - betaln(draws-k+1, bad-draws+k+1) - betaln(total+1, 1)

        # TODO: avoid recomputation of terms
        result = (
            torch.lgamma(self.M + 1)  # type: ignore
            + torch.lgamma(torch.tensor(1.0))
            - torch.lgamma(self.M + 2)  # type: ignore
            + torch.lgamma(N_minus_M + 1)  # type: ignore
            + torch.lgamma(torch.tensor(1.0))
            - torch.lgamma(N_minus_M + 2)  # type: ignore
            + torch.lgamma(self.N - self.n + 1)  # type: ignore
            + torch.lgamma(self.n + 1)  # type: ignore
            - torch.lgamma(self.N + 2)  # type: ignore
            - torch.lgamma(k + 1)
            - torch.lgamma(self.M - k + 1)
            + torch.lgamma(self.M + 2)  # type: ignore
            - torch.lgamma(n_minus_k + 1)
            - torch.lgamma(N_minus_M - self.n + k + 1)
            + torch.lgamma(N_minus_M + 2)  # type: ignore
            - torch.lgamma(self.N + 1)  # type: ignore
            - torch.lgamma(torch.tensor(1.0))
            + torch.lgamma(self.N + 2)  # type: ignore
        )

        return result

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
        support_mask = (
            ((scope_data >= max(0, self.n + self.M - self.N)) & (scope_data <= min(self.n, self.M)))  # type: ignore
            .sum(dim=1)
            .bool()
        )
        log_prob[prob_mask & (~support_mask)] = -float("Inf")
        # compute probabilities for values inside distribution support
        log_prob[prob_mask & support_mask] = self.log_prob(scope_data[prob_mask & support_mask])

        return log_prob

    def set_params(self, N: int, M: int, n: int) -> None:

        if N < 0 or not np.isfinite(N):
            raise ValueError(
                f"Value of N for Hypergeometric distribution must be greater of equal to 0, but was: {N}"
            )
        if M < 0 or M > N or not np.isfinite(M):
            raise ValueError(
                f"Value of M for Hypergeometric distribution must be greater of equal to 0 and less or equal to N, but was: {M}"
            )
        if n < 0 or n > N or not np.isfinite(n):
            raise ValueError(
                f"Value of n for Hypergeometric distribution must be greater of equal to 0 and less or equal to N, but was: {n}"
            )

        self.M.data = torch.tensor(int(M))
        self.N.data = torch.tensor(int(N))
        self.n.data = torch.tensor(int(n))

    def get_params(self) -> Tuple[int, int, int]:
        return self.N.data.cpu().numpy(), self.M.data.cpu().numpy(), self.n.data.cpu().numpy()  # type: ignore


@dispatch(Hypergeometric)  # type: ignore[no-redef]
def toTorch(node: Hypergeometric) -> TorchHypergeometric:
    return TorchHypergeometric(node.scope, *node.get_params())


@dispatch(TorchHypergeometric)  # type: ignore[no-redef]
def toNodes(torch_node: TorchHypergeometric) -> Hypergeometric:
    return Hypergeometric(torch_node.scope, *torch_node.get_params())
