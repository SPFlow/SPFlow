"""Contains Negative Binomial leaf node for SPFlow in the ``torch`` backend.
"""
from spflow.modules.node.leaf.utils import apply_nan_strategy
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as D
from torch import Tensor, nn

from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.modules.node.leaf_node import LeafNode
from spflow.utils.projections import proj_bounded_to_real, proj_real_to_bounded


class NegativeBinomial(LeafNode):
    r"""(Univariate) Negative Binomial distribution leaf node in the 'base' backend.

    Represents an univariate Negative Binomial distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \binom{k+n-1}{n-1}p^n(1-p)^k

    where
        - :math:`k` is the number of failures
        - :math:`n` is the maximum number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Internally :math:`p` is represented as an unbounded parameter that is projected onto the bounded range :math:`[0,1]` for representing the actual success probability.

    Attributes:
        n:
            Scalar PyTorch tensor representing the number of successes (greater or equal to 0).
        p_aux:
            Unbounded scalar PyTorch parameter that is projected to yield the actual success probability.
        p:
            Scalar PyTorch tensor representing the success probability (projected from ``p_aux``).
    """

    def __init__(self, scope: Scope, n: int, p: Optional[float] = 0.5) -> None:
        r"""Initializes ``NegativeBinomial`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            n:
                Integer representing the number of successes (greater or equal to 0).
            p:
                Floating point value representing the success probability of each trial in the range :math:`(0,1]`.
                Defaults to 0.5.
        """
        super().__init__(scope=scope)
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'NegativeBinomial' should be 1, but was: {len(scope.query)}."
            )
        if len(scope.evidence) != 0:
            raise ValueError(
                f"Evidence scope for 'NegativeBinomial' should be empty, but was {scope.evidence}."
            )
        if isinstance(n, float):
            if not n.is_integer():
                raise ValueError(
                    f"Value of 'n' for 'Binomial' must be (equal to) an integer value, but was: {n}"
                )
            n = torch.tensor(int(n))
        elif isinstance(n, int):
            n = torch.tensor(n)

        if not torch.is_tensor(p):
            p = torch.tensor(p)

        if p <= 0.0 or p > 1.0 or not torch.isfinite(p):
            raise ValueError(
                f"Value of 'p' for 'NegativeBinomial' must to be between 0.0 and 1.0, but was: {p}"
            )
        if n < 0 or not np.isfinite(n):
            raise ValueError(
                f"Value of 'n' for 'NegativeBinomial' must to greater of equal to 0, but was: {n}"
            )

        if not (np.remainder(n, 1.0) == 0.0):
            raise ValueError(
                f"Value of 'n' for 'NegativeBinomial' must be (equal to) an integer value, but was: {n}"
            )

        # register number of trials n as torch buffer (not trainable)
        self.register_buffer("n", n)

        # register auxiliary torch parameter for the success probability p
        self.p_aux = nn.Parameter(torch.empty(1))
        self.p = p

    @property
    def p(self) -> Tensor:
        """Returns the success proability."""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.p_aux, 0.0, 1.0)

    @p.setter
    def p(self, p: Tensor) -> None:
        """Sets the success probability."""
        # project parameter onto auxiliary parameter range
        if p < 0.0 or p > 1.0 or not torch.isfinite(p):
            raise ValueError(f"Value of 'p' for 'Binomial' must to be between 0.0 and 1.0, but was: {p}")

        self.p_aux.data = proj_bounded_to_real(p, 0.0, 1.0)

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``NegativeBinomial`` can represent a single univariate node with ``NegativeBinomialType`` domain.

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

        # leaf is a discrete Negative Binomial distribution
        # NOTE: only accept instances of 'FeatureTypes.NegativeBinomial', otherwise required parameter 'n' is not specified. Reject 'FeatureTypes.Discrete' for the same reason.
        if not isinstance(domains[0], FeatureTypes.NegativeBinomial):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "NegativeBinomial":
        """Creates an instance from a specified signature.

        Returns:
            ``NegativeBinomial`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'NegativeBinomial' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if isinstance(domain, FeatureTypes.NegativeBinomial):
            n, p = domain.n, domain.p
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'NegativeBinomial' that was not caught during acception checking."
            )

        return NegativeBinomial(feature_ctx.scope, n=n, p=p)

    @property
    def distribution(self) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Returns:
            ``torch.distributions.NegativeBinomial`` instance.
        """
        return D.NegativeBinomial(total_count=self.n, probs=self.p)


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: NegativeBinomial,
    data: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``NegativeBinomial`` node parameters in the ``torch`` backend.

    Estimates the success probability :math:`p` of a Negative Binomial distribution from data, as follows:

    .. math::

        p^{\*}=\frac{n\sum_{i=1}^N w_i}{\sum_{i=1}^{N}w_i(x_i+n)}

    where
        - :math:`n` is the number of successes
        - :math:`N` is the number of samples in the data set
        - :math:`x_i` is the data of the relevant scope for the `i`-th sample of the data set
        - :math:`w_i` is the weight for the `i`-th sample of the data set

    The number of successes is fixed and will not be estimated.
    Weights are normalized to sum up to :math:`N`.

    Args:
        leaf:
            Leaf node to estimate parameters of.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        weights:
            Optional one-dimensional PyTorch tensor containing non-negative weights for all data samples.
            Must match number of samples in ``data``.
            Defaults to None in which case all weights are initialized to ones.
        bias_corrections:
            Boolen indicating whether or not to correct possible biases.
            Has no effect for ``NegativeBinomial`` nodes.
            Defaults to True.
        nan_strategy:
            Optional string or callable specifying how to handle missing data.
            If 'ignore', missing values (i.e., NaN entries) are ignored.
            If a callable, it is called using ``data`` and should return another PyTorch tensor of same size.
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

    # select relevant data for scope
    scope_data = data[:, leaf.scope.query]

    # handle nans
    scope_data, weights = apply_nan_strategy(nan_strategy, scope_data, leaf, weights, check_support)

    # total (weighted) number of successes
    n_success = weights.sum() * leaf.n

    # count (weighted) number of trials
    n_total = (weights * (scope_data + leaf.n)).sum()

    # estimate (weighted) success probability
    p_est = n_success / n_total

    # Convert to failure probability
    p_est = 1 - p_est

    # edge case: if prob. 1 (or 0), set to smaller (or larger) value
    if torch.isclose(p_est, torch.tensor(0.0)):
        p_est = torch.tensor(1e-8)
    elif torch.isclose(p_est, torch.tensor(1.0)):
        p_est = torch.tensor(1 - 1e-8)

    # set parameters of leaf node
    leaf.p = p_est
