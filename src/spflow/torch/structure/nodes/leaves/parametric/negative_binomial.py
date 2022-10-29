# -*- coding: utf-8 -*-
"""Contains Negative Binomial leaf node for SPFlow in the ``torch`` backend.
"""
import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter
from typing import Tuple, Optional
from .projections import proj_bounded_to_real, proj_real_to_bounded
from spflow.meta.scope.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.nodes.node import LeafNode
from spflow.base.structure.nodes.leaves.parametric.negative_binomial import (
    NegativeBinomial as BaseNegativeBinomial,
)


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
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'NegativeBinomial' should be 1, but was: {len(scope.query)}."
            )
        if len(scope.evidence):
            raise ValueError(
                f"Evidence scope for 'NegativeBinomial' should be empty, but was {scope.evidence}."
            )

        super(NegativeBinomial, self).__init__(scope=scope)

        # register number of trials n as torch buffer (should not be changed)
        self.register_buffer("n", torch.empty(size=[]))

        # register auxiliary torch parameter for the success probability p
        self.p_aux = Parameter()

        # set parameters
        self.set_params(n, p)

    @property
    def p(self) -> torch.Tensor:
        """TODO"""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.p_aux, lb=0.0, ub=1.0)  # type: ignore

    @property
    def dist(self) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Returns:
            ``torch.distributions.NegativeBinomial`` instance.
        """
        # note: the distribution is not stored as an attribute due to mismatching parameters after gradient updates (gradients don't flow back to p when initializing with 1.0-p)
        return D.NegativeBinomial(
            total_count=self.n, probs=torch.ones(1) - self.p
        )

    def set_params(self, n: int, p: float) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            n:
               Integer representing the number of successes (greater or equal to 0).
            p:
                Floating point value representing the success probability of each trial in the range :math:`(0,1]`.
        """
        if p <= 0.0 or p > 1.0 or not np.isfinite(p):
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

        self.n.data = torch.tensor(int(n))  # type: ignore
        self.p_aux.data = proj_bounded_to_real(torch.tensor(float(p)), lb=0.0, ub=1.0)  # type: ignore

    def get_params(self) -> Tuple[int, float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of the number of successes and the floating point value representing the success probability.
        """
        return self.n.data.cpu().numpy(), self.p.data.cpu().numpy()  # type: ignore

    def check_support(self, data: torch.Tensor, is_scope_data: bool=False) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Negative Binomial distribution, which is:

        .. math::

            \text{supp}(\text{NegativeBinomial})=\mathbb{N}\cup\{0\}

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
        valid[~nan_mask] = self.dist.support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore

        # check if all values are valid integers
        valid[~nan_mask & valid] &= (
            torch.remainder(
                scope_data[~nan_mask & valid], torch.tensor(1)
            ).squeeze(-1)
            == 0
        )

        # check for infinite values
        valid[~nan_mask & valid] &= (
            ~scope_data[~nan_mask & valid].isinf().squeeze(-1)
        )

        return valid


@dispatch(memoize=True)  # type: ignore
def toTorch(
    node: BaseNegativeBinomial, dispatch_ctx: Optional[DispatchContext] = None
) -> NegativeBinomial:
    """Conversion for ``NegativeBinomial`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return NegativeBinomial(node.scope, *node.get_params())


@dispatch(memoize=True)  # type: ignore
def toBase(
    node: NegativeBinomial, dispatch_ctx: Optional[DispatchContext] = None
) -> BaseNegativeBinomial:
    """Conversion for ``NegativeBinomial`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseNegativeBinomial(node.scope, *node.get_params())
