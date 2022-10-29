# -*- coding: utf-8 -*-
"""Contains Poisson leaf node for SPFlow in the ``torch`` backend.
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
from spflow.base.structure.nodes.leaves.parametric.poisson import (
    Poisson as BasePoisson,
)


class Poisson(LeafNode):
    r"""(Univariate) Poisson distribution leaf node in the ``torch`` backend.

    Represents a univariate Poisson distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \lambda^k\frac{e^{-\lambda}}{k!}

    where
        - :math:`k` is the number of occurrences
        - :math:`\lambda` is the rate parameter

    Internally :math:`l` is represented as an unbounded parameter that is projected onto the bounded range :math:`[0,\infty)` for representing the actual rate probability.

    Attributes:
        l_aux:
            Unbounded scalar PyTorch parameter that is projected to yield the actual rate parameter.
        l:
            Scalar PyTorch tensor representing the rate parameters (:math:`\lambda`) of the Poisson distribution (projected from ``l_aux``).
    """

    def __init__(self, scope: Scope, l: Optional[float] = 1.0) -> None:
        r"""Initializes ``Poisson`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            l:
                Floating point value representing the rate parameter (:math:`\lambda`), expected value and variance of the Poisson distribution (must be greater than or equal to 0).
                Defaults to 1.0.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'Poisson' should be 1, but was: {len(scope.query)}."
            )
        if len(scope.evidence):
            raise ValueError(
                f"Evidence scope for 'Poisson' should be empty, but was {scope.evidence}."
            )

        super(Poisson, self).__init__(scope=scope)

        # register auxiliary torch parameter for lambda l
        self.l_aux = Parameter()

        # set parameters
        self.set_params(l)

    @property
    def l(self) -> torch.Tensor:
        """TODO"""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.l_aux, lb=0.0)  # type: ignore

    @property
    def dist(self) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Returns:
            ``torch.distributions.Poisson`` instance.
        """
        return D.Poisson(rate=self.l)

    def set_params(self, l: float) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            l:
                Floating point value representing the rate parameter (:math:`\lambda`), expected value and variance of the Poisson distribution (must be greater than or equal to 0).
        """
        if not np.isfinite(l):
            raise ValueError(
                f"Value of l for Poisson distribution must be finite, but was: {l}"
            )

        if l < 0:
            raise ValueError(
                f"Value of l for Poisson distribution must be non-negative, but was: {l}"
            )

        self.l_aux.data = proj_bounded_to_real(torch.tensor(float(l)), lb=0.0)

    def get_params(self) -> Tuple[float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Floating point value representing the rate parameter, expected value and variance.
        """
        return (self.l.data.cpu().numpy(),)  # type: ignore

    def check_support(self, data: torch.Tensor, is_scope_data: bool=False) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Poisson distribution, which is:

        .. math::

            \text{supp}(\text{Poisson})=\mathbb{N}\cup\{0\}

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
    node: BasePoisson, dispatch_ctx: Optional[DispatchContext] = None
) -> Poisson:
    """Conversion for ``Poisson`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return Poisson(node.scope, *node.get_params())


@dispatch(memoize=True)  # type: ignore
def toBase(
    node: Poisson, dispatch_ctx: Optional[DispatchContext] = None
) -> BasePoisson:
    """Conversion for ``Poisson`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BasePoisson(node.scope, *node.get_params())
