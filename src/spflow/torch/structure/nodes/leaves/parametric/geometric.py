# -*- coding: utf-8 -*-
"""Contains Geometric leaf node for SPFlow in the ``torch`` backend.
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
from spflow.base.structure.nodes.leaves.parametric.geometric import (
    Geometric as BaseGeometric,
)


class Geometric(LeafNode):
    r"""(Univariate) Geometric distribution leaf node in the ``torch`` backend.

    Represents an univariate Geometric distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) =  p(1-p)^{k-1}

    where
        - :math:`k` is the number of trials
        - :math:`p` is the success probability of each trial

    Internally :math:`p` is represented as an unbounded parameter that is projected onto the bounded range :math:`(0,1]` for representing the actual success probability.

    Attributes:
        p_aux:
            Unbounded scalar PyTorch parameter that is projected to yield the actual success probability.
        p:
            Scalar PyTorch tensor representing the success probability in the range :math:`(0,1]` (projected from ``p_aux``).
    """

    def __init__(self, scope: Scope, p: float = 0.5) -> None:
        r"""Initializes ``Geometric`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            p:
                Floating points representing the probability of success in the range :math:`(0,1]`.
                Defaults to 0.5.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'Geometric' should be 1, but was {len(scope.query)}."
            )
        if len(scope.evidence):
            raise ValueError(
                f"Evidence scope for 'Geometric' should be empty, but was {scope.evidence}."
            )

        super(Geometric, self).__init__(scope=scope)

        # register auxiliary torch parameter for the success probability p
        self.p_aux = Parameter()

        # set parameters
        self.set_params(p)

    @property
    def p(self) -> torch.Tensor:
        """TODO"""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.p_aux, lb=0.0, ub=1.0)  # type: ignore

    @property
    def dist(self) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Note, that the Geometric distribution as implemented in PyTorch uses :math:`k-1` as input.
        Therefore values are offset by 1 if used directly.

        Returns:
            ``torch.distributions.Geometric`` instance.
        """
        return D.Geometric(probs=self.p)

    def set_params(self, p: float) -> None:

        if p <= 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for Geometric distribution must to be greater than 0.0 and less or equal to 1.0, but was: {p}"
            )

        self.p_aux.data = proj_bounded_to_real(
            torch.tensor(float(p)), lb=0.0, ub=1.0
        )

    def get_params(self) -> Tuple[float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Floating point value representing the success probability.
        """
        return (self.p.data.cpu().numpy(),)  # type: ignore

    def check_support(self, scope_data: torch.Tensor) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Geometric distribution, which is:

        .. math::

            \text{supp}(\text{Geometric})=\mathbb{N}\setminus\{0\}

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            scope_data:
                Two-dimensional PyTorch tensor containing sample instances.
                Each row is regarded as a sample.
        Returns:
            Two-dimensional PyTorch tensor indicating for each instance, whether they are part of the support (True) or not (False).
        """
        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scope.query):
            raise ValueError(
                f"Expected scope_data to be of shape (n,{len(self.scope.query)}), but was: {scope_data.shape}"
            )

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        valid = torch.ones(scope_data.shape[0], 1, dtype=torch.bool)
        # data needs to be offset by -1 due to the different definitions between SciPy and PyTorch
        valid[~nan_mask] = self.dist.support.check(scope_data[~nan_mask] - 1).squeeze(-1)  # type: ignore

        # check for infinite values
        valid[~nan_mask & valid] &= (
            ~scope_data[~nan_mask & valid].isinf().squeeze(-1)
        )

        return valid


@dispatch(memoize=True)  # type: ignore
def toTorch(
    node: BaseGeometric, dispatch_ctx: Optional[DispatchContext] = None
) -> Geometric:
    """Conversion for ``Geometric`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return Geometric(node.scope, *node.get_params())


@dispatch(memoize=True)  # type: ignore
def toBase(
    node: Geometric, dispatch_ctx: Optional[DispatchContext] = None
) -> BaseGeometric:
    """Conversion for ``Geometric`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseGeometric(node.scope, *node.get_params())
