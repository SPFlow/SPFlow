"""Contains Hypergeometric leaf node for SPFlow in the ``torch`` backend.
"""
from typing import List, Optional, Tuple

import numpy as np
import torch

from spflow.base.structure.general.nodes.leaves.parametric.hypergeometric import (
    Hypergeometric as BaseHypergeometric,
)
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.nodes.leaf_node import LeafNode


class Hypergeometric(LeafNode):
    r"""(Univariate) Hypergeometric distribution leaf node in the 'base' backend.

    Represents an univariate Hypergeometric distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \frac{\binom{M}{k}\binom{N-M}{n-k}}{\binom{N}{n}}

    where
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)
        - :math:`N` is the total number of entities
        - :math:`M` is the number of entities with property of interest
        - :math:`n` is the number of draws
        - :math:`k` s the number of observed entities

    Attributes:
        N:
            Scalar PyTorch tensor specifying the total number of entities (in the population), greater or equal to 0.
        M:
            Scalar PyTorch tensor specifying the number of entities with property of interest (in the population), greater or equal to zero and less than or equal to N.
        n:
            Scalar PyTorch tensor specifying the number of draws, greater of equal to zero and less than or equal to N.
    """

    def __init__(self, scope: Scope, N: int, M: int, n: int) -> None:
        r"""Initializes 'Hypergeometric' leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            N:
                Integer specifying the total number of entities (in the population), greater or equal to 0.
            M:
                Integer specifying the number of entities with property of interest (in the population), greater or equal to zero and less than or equal to N.
            n:
                Integer specifying the number of draws, greater of equal to zero and less than or equal to N.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'Hypergeometric' should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'Hypergeometric' should be empty, but was {scope.evidence}.")

        super().__init__(scope=scope)

        # register parameters as torch buffers (should not be changed)
        self.register_buffer("N", torch.empty(size=[]))
        self.register_buffer("M", torch.empty(size=[]))
        self.register_buffer("n", torch.empty(size=[]))

        # set parameters
        self.set_params(N, M, n)

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Hypergeometric`` can represent a single univariate node with ``HypergeometricType`` domain.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf only has one output
        if len(signatures) != 1:
            return False

        # get single output signature
        feature_ctx = signatures[0]
        domains = feature_ctx.get_domains()

        # leaf is a single non-conditional univariate node
        if len(domains) != 1 or len(feature_ctx.scope.query) != len(domains) or len(feature_ctx.scope.evidence) != 0:
            return False

        # leaf is a discrete Hypergeometric distribution
        # NOTE: only accept instances of 'FeatureTypes.Hypergeometric', otherwise required parameters 'N','M','n' are not specified. Reject 'FeatureTypes.Discrete' for the same reason.
        if not isinstance(domains[0], FeatureTypes.Hypergeometric):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "Hypergeometric":
        """Creates an instance from a specified signature.

        Returns:
            ``Hypergeometric`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'Hypergeometric' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if isinstance(domain, FeatureTypes.Hypergeometric):
            N, M, n = domain.N, domain.M, domain.n
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Hypergeometric' that was not caught during acception checking."
            )

        return Hypergeometric(feature_ctx.scope, N=N, M=M, n=n)

    def log_prob(self, k: torch.Tensor) -> torch.Tensor:
        """Computes the log-likelihood for specified input data.

        The log-likelihoods of the Hypergeometric distribution are computed according to the logarithm of its probability mass function (PMF).

        Args:
            k:
                Two-dimensional PyTorch tensor containing sample instances.
                Each row is regarded as a sample.

        Returns:
            Two-dimensional PyTorch tensor containing the log-likelihoods of the corresponding input samples.
        """
        N_minus_M = self.N - self.M  # type: ignore
        n_minus_k = self.n - k  # type: ignore

        # ----- (M over m) * (N-M over n-k) / (N over n) -----

        # log_M_over_k = torch.lgamma(self.M+1) - torch.lgamma(self.M-k+1) - torch.lgamma(k+1)
        # log_NM_over_nk = torch.lgamma(N_minus_M+1) - torch.lgamma(N_minus_M-n_minus_k+1) - torch.lgamma(n_minus_k+1)
        # log_N_over_n = torch.lgamma(self.N+1) - torch.lgamma(self.N-self.n+1) - torch.lgamma(self.n+1)
        # result = log_M_over_k + log_NM_over_nk - log_N_over_n

        # ---- alternatively (more precise according to SciPy) -----
        # betaln(good+1, 1) + betaln(bad+1,1) + betaln(total-draws+1, draws+1) - betaln(k+1, good-k+1) - betaln(draws-k+1, bad-draws+k+1) - betaln(total+1, 1)

        # reuse terms that occur multiple times
        lgamma_1 = torch.lgamma(torch.tensor(1))
        lgamma_M_p_2 = torch.lgamma(self.M + 2)
        lgamma_N_p_2 = torch.lgamma(self.N + 2)
        lgamma_N_m_M_p_2 = torch.lgamma(N_minus_M + 2)

        result = (
            torch.lgamma(self.M + 1)  # type: ignore
            + lgamma_1
            - lgamma_M_p_2  # type: ignore
            + torch.lgamma(N_minus_M + 1)  # type: ignore
            + lgamma_1
            - lgamma_N_m_M_p_2  # type: ignore
            + torch.lgamma(self.N - self.n + 1)  # type: ignore
            + torch.lgamma(self.n + 1)  # type: ignore
            - lgamma_N_p_2  # type: ignore
            - torch.lgamma(k + 1)  # .float()
            - torch.lgamma(self.M - k + 1)
            + lgamma_M_p_2  # type: ignore
            - torch.lgamma(n_minus_k + 1)
            - torch.lgamma(N_minus_M - self.n + k + 1)
            + lgamma_N_m_M_p_2  # type: ignore
            - torch.lgamma(self.N + 1)  # type: ignore
            - lgamma_1
            + lgamma_N_p_2  # type: ignore
        )

        return result

    def set_params(self, N: int, M: int, n: int) -> None:
        """Sets the parameters for the represented distribution.

        Args:
            N:
                Integer specifying the total number of entities (in the population), greater or equal to 0.
            M:
                Integer specifying the number of entities with property of interest (in the population), greater or equal to zero and less than or equal to N.
            n:
                Integer specifying the number of draws, greater of equal to zero and less than or equal to N.
        """
        if N < 0 or not np.isfinite(N):
            raise ValueError(f"Value of 'N' for 'Hypergeometric' must be greater of equal to 0, but was: {N}")
        if not (torch.remainder(torch.tensor(N), 1.0) == torch.tensor(0.0)):
            raise ValueError(f"Value of 'N' for 'Hypergeometric' must be (equal to) an integer value, but was: {N}")

        if M < 0 or M > N or not np.isfinite(M):
            raise ValueError(
                f"Value of 'M' for 'Hypergeometric' must be greater of equal to 0 and less or equal to N, but was: {M}"
            )
        if not (torch.remainder(torch.tensor(M), 1.0) == torch.tensor(0.0)):
            raise ValueError(f"Value of 'M' for 'Hypergeometric' must be (equal to) an integer value, but was: {M}")

        if n < 0 or n > N or not np.isfinite(n):
            raise ValueError(
                f"Value of 'n' for 'Hypergeometric' must be greater of equal to 0 and less or equal to N, but was: {n}"
            )
        if not (torch.remainder(torch.tensor(n), 1.0) == torch.tensor(0.0)):
            raise ValueError(f"Value of 'n' for 'Hypergeometric' must be (equal to) an integer value, but was: {n}")

        self.M.data = torch.tensor(int(M))
        self.N.data = torch.tensor(int(N))
        self.n.data = torch.tensor(int(n))

    def get_params(self) -> Tuple[int, int, int]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of integer values representing the size of the total population, the size of the population of interest and the number of draws.
        """
        return self.N.data.cpu().numpy(), self.M.data.cpu().numpy(), self.n.data.cpu().numpy()  # type: ignore

    def check_support(self, data: torch.Tensor, is_scope_data: bool = False) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Hypergeometric distribution, which is:

        .. math::

            \text{supp}(\text{Hypergeometric})={\max(0,n+M-N),...,\min(n,M)}

        where
            - :math:`N` is the total number of entities
            - :math:`M` is the number of entities with property of interest
            - :math:`n` is the number of draws

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional PyTorch tensor containing sample instances.
                Each row is regarded as a sample.
                Unless ``is_scope_data`` is set to True, it is assumed that the relevant data is located in the columns corresponding to the scope indices.
            is_scope_data:
                Boolean indicating if the given data already contains the relevant data for the leaf's scope in the correct order (True) or if it needs to be extracted from the full data set.
                Defaults to False.

        Returns:
            Two-dimensional PyTorch tensor indicating for each instance, whether they are part of the support (True) or not (False).
        """
        if is_scope_data:
            scope_data = data
        else:
            # select relevant data for scope
            scope_data = data[:, self.scope.query]

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scope.query):
            raise ValueError(
                f"Expected 'scope_data' to be of shape (n,{len(self.scope.query)}), but was: {scope_data.shape}"
            )

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        valid = torch.ones(scope_data.shape[0], 1, dtype=torch.bool)

        # check for infinite values
        valid[~nan_mask] &= ~scope_data[~nan_mask & valid].isinf().squeeze(-1)

        # check if all values are valid integers
        valid[~nan_mask & valid] &= torch.remainder(scope_data[~nan_mask & valid], 1).squeeze(-1) == 0

        # check if values are in valid range
        valid[~nan_mask & valid] &= (scope_data[~nan_mask & valid] >= max(0, self.n + self.M - self.N)) & (  # type: ignore
            scope_data[~nan_mask & valid] <= min(self.n, self.M)  # type: ignore
        ).squeeze(
            -1
        )

        return valid


@dispatch(memoize=True)  # type: ignore
def toTorch(node: BaseHypergeometric, dispatch_ctx: Optional[DispatchContext] = None) -> Hypergeometric:
    """Conversion for ``Hypergeometric`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return Hypergeometric(node.scope, *node.get_params())


@dispatch(memoize=True)  # type: ignore
def toBase(node: Hypergeometric, dispatch_ctx: Optional[DispatchContext] = None) -> BaseHypergeometric:
    """Conversion for ``Hypergeometric`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseHypergeometric(node.scope, *node.get_params())
