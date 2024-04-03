"""Contains Hypergeometric leaf node for SPFlow in the ``torch`` backend.
"""
from spflow.modules.node.leaf_node import LeafNode
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)

from typing import Callable, Optional, Union

import numpy as np

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


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
        super().__init__(scope=scope)
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'Hypergeometric' should be 1, but was: {len(scope.query)}."
            )
        if len(scope.evidence) != 0:
            raise ValueError(
                f"Evidence scope for 'Hypergeometric' should be empty, but was {scope.evidence}."
            )

        if N < 0 or not np.isfinite(N):
            raise ValueError(f"Value of 'N' for 'Hypergeometric' must be greater of equal to 0, but was: {N}")
        if not (torch.remainder(torch.tensor(N), 1.0) == torch.tensor(0.0)):
            raise ValueError(
                f"Value of 'N' for 'Hypergeometric' must be (equal to) an integer value, but was: {N}"
            )

        if M < 0 or M > N or not np.isfinite(M):
            raise ValueError(
                f"Value of 'M' for 'Hypergeometric' must be greater of equal to 0 and less or equal to N, but was: {M}"
            )
        if not (torch.remainder(torch.tensor(M), 1.0) == torch.tensor(0.0)):
            raise ValueError(
                f"Value of 'M' for 'Hypergeometric' must be (equal to) an integer value, but was: {M}"
            )

        if n < 0 or n > N or not np.isfinite(n):
            raise ValueError(
                f"Value of 'n' for 'Hypergeometric' must be greater of equal to 0 and less or equal to N, but was: {n}"
            )
        if not (torch.remainder(torch.tensor(n), 1.0) == torch.tensor(0.0)):
            raise ValueError(
                f"Value of 'n' for 'Hypergeometric' must be (equal to) an integer value, but was: {n}"
            )

        # register parameters as torch buffers (should not be changed)
        self.register_buffer("N", torch.tensor(N))
        self.register_buffer("M", torch.tensor(M))
        self.register_buffer("n", torch.tensor(n))

    @property
    def device(self) -> torch.device:
        """Overwrite since this leaf has no parameters and the Module device method iterates over the parameters."""
        return self.N.device

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
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
        if (
            len(domains) != 1
            or len(feature_ctx.scope.query) != len(domains)
            or len(feature_ctx.scope.evidence) != 0
        ):
            return False

        # leaf is a discrete Hypergeometric distribution
        # NOTE: only accept instances of 'FeatureTypes.Hypergeometric', otherwise required parameters 'N','M','n' are not specified. Reject 'FeatureTypes.Discrete' for the same reason.
        if not isinstance(domains[0], FeatureTypes.Hypergeometric):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "Hypergeometric":
        """Creates an instance from a specified signature.

        Returns:
            ``Hypergeometric`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'Hypergeometric' cannot be instantiated from the following signatures: {signatures}."
            )

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

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return _HypegeometricDistribution(self.N, self.M, self.n)


class _HypegeometricDistribution:
    def __init__(self, N, M, n):
        self.N = N
        self.M = M
        self.n = n
        self.support = HypergeometricSupport(N, M, n)

    def sample(self, size=None):
        # TODO: may be inefficient
        # create random permutations of N elements
        rand_perm = torch.argsort(torch.rand(size + (self.N,)), dim=1)
        # assuming that first M indices are the M objects of interest, count how many of these indices were "drawn" in the first n draws (with replacement since all indices are unique per row)
        return (rand_perm[:, : self.n] < self.M).sum(dim=1)

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

        # reuse terms that occur multiple times
        lgamma_1 = torch.lgamma(torch.tensor(1))
        lgamma_M_p_2 = torch.lgamma(self.M + 2)
        lgamma_N_p_2 = torch.lgamma(self.N + 2)
        lgamma_N_m_M_p_2 = torch.lgamma(N_minus_M + 2)

        result = (
            torch.lgamma(self.M + 1)
            + lgamma_1
            - lgamma_M_p_2
            + torch.lgamma(N_minus_M + 1)
            + lgamma_1
            - lgamma_N_m_M_p_2
            + torch.lgamma(self.N - self.n + 1)
            + torch.lgamma(self.n + 1)
            - lgamma_N_p_2
            - torch.lgamma(k + 1)
            - torch.lgamma(self.M - k + 1)
            + lgamma_M_p_2
            - torch.lgamma(n_minus_k + 1)
            - torch.lgamma(N_minus_M - self.n + k + 1)
            + lgamma_N_m_M_p_2
            - torch.lgamma(self.N + 1)
            - lgamma_1
            + lgamma_N_p_2
        )

        return result


class HypergeometricSupport:
    def __init__(self, N, M, n):
        self.N = N
        self.M = M
        self.n = n

    def check(self, data: Tensor):
        # check if all values are valid integers
        valid = torch.remainder(data, 1).squeeze(-1) == 0

        # check if values are in valid range
        valid &= data[valid] >= max(0, self.n + self.M - self.N)
        valid &= data[valid] <= min(self.n, self.M)
        return valid


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: Hypergeometric,
    data: Tensor,
    weights: Optional[Tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``Hypergeometric`` node parameters in the ``base`` backend.

    All parameters of the Hypergeometric distribution are regarded as fixed and will not be estimated.
    Therefore, this method does nothing, but check for the validity of the data.

    Args:
        leaf:
            Leaf node to estimate parameters of.
        data:
            Two-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        weights:
            Optional one-dimensional NumPy array containing non-negative weights for all data samples.
            Must match number of samples in ``data``.
            Defaults to None in which case all weights are initialized to ones.
        bias_corrections:
            Boolen indicating whether or not to correct possible biases.
            Has no effects for ``Hypergeometric`` nodes.
            Defaults to True.
        nan_strategy:
            Optional string or callable specifying how to handle missing data.
            If 'ignore', missing values (i.e., NaN entries) are ignored.
            If a callable, it is called using ``data`` and should return another NumPy array of same size.
            Defaults to None.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Raises:
        ValueError: Invalid arguments.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    if check_support:
        if torch.any(~leaf.check_support(data[:, leaf.scope.query])):
            raise ValueError("Encountered values outside of the support for 'Hypergeometric'.")

    # do nothing since there are no learnable parameters
