# -*- coding: utf-8 -*-
"""Contains conditional Poisson leaf node for SPFlow in the ``torch`` backend.
"""
import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter
from typing import List, Tuple, Optional, Callable
from spflow.meta.scope.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.nodes.node import LeafNode
from spflow.base.structure.nodes.leaves.parametric.cond_poisson import (
    CondPoisson as BaseCondPoisson,
)


class CondPoisson(LeafNode):
    r"""Conditional (univariate) Poisson distribution leaf node in the ``torch`` backend.

    Represents a conditional univariate Poisson distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \lambda^k\frac{e^{-\lambda}}{k!}

    where
        - :math:`k` is the number of occurrences
        - :math:`\lambda` is the rate parameter

    Attributes:
        cond_f:
            Optional callable to retrieve the conditional parameter for the leaf node.
            Its output should be a dictionary containing ``l`` as a key, and the value should be
            a floating point, scalar NumPy array or scalar PyTorch tensor representing the rate parameter, greater than or equal to 0.
    """

    def __init__(self, scope: Scope, cond_f: Optional[Callable] = None) -> None:
        r"""Initializes ``CondPoisson`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``l`` as a key, and the value should be
                a floating point, scalar NumPy array or scalar PyTorch tensor representing the rate parameter, greater than or equal to 0.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'CondPoisson' should be 1, but was: {len(scope.query)}."
            )
        if len(scope.evidence):
            raise ValueError(
                f"Evidence scope for 'CondPoisson' should be empty, but was {scope.evidence}."
            )

        super(CondPoisson, self).__init__(scope=scope)

        self.set_cond_f(cond_f)

    def set_cond_f(self, cond_f: Optional[Callable] = None) -> None:
        r"""Sets the function to retrieve the node's conditonal parameter.

        Args:
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``l`` as a key, and the value should be
                a floating point, scalar NumPy array or scalar PyTorch tensor representing the rate parameter, greater than or equal to 0.
        """
        self.cond_f = cond_f

    def dist(self, l: torch.Tensor) -> D.Distribution:
        return D.Poisson(rate=l)

    def retrieve_params(
        self, data: torch.Tensor, dispatch_ctx: DispatchContext
    ) -> torch.Tensor:
        r"""Retrieves the conditional parameter of the leaf node.

        First, checks if conditional parameters (``l``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameters.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameters.

        Args:
            data:
                Two-dimensional PyTorch tensor containing the data to compute the conditional parameters.
                Each row is regarded as a sample.
            dispatch_ctx:
                Dispatch context.

        Returns:
            Scalar PyTorch tensor representing the rate parameter.

        Raises:
            ValueError: No way to retrieve conditional parameters or invalid conditional parameters.
        """
        l, cond_f = None, None

        # check dispatch cache for required conditional parameter 'l'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if a value for 'l' is specified (highest priority)
            if "l" in args:
                l = args["l"]
            # check if alternative function to provide 'l' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'l' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'l' nor 'cond_f' is specified (via node or arguments)
        if l is None and cond_f is None:
            raise ValueError(
                "'CondPoisson' requires either 'l' or 'cond_f' to retrieve 'l' to be specified."
            )

        # if 'l' was not already specified, retrieve it
        if l is None:
            l = cond_f(data)["l"]

        if isinstance(l, int) or isinstance(l, float):
            l = torch.tensor(l)

        # check if value for 'l' is valid
        if not torch.isfinite(l):
            raise ValueError(
                f"Value of 'l' for 'CondPoisson' must be finite, but was: {l}"
            )

        if l < 0:
            raise ValueError(
                f"Value of 'l' for 'CondPoisson' must be non-negative, but was: {l}"
            )

        return l

    def check_support(self, scope_data: torch.Tensor) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Poisson distribution, which is:

        .. math::

            \text{supp}(\text{Poisson})=\mathbb{N}\cup\{0\}

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
        valid[~nan_mask] = self.dist(torch.tensor(1.0)).support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore

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
    node: BaseCondPoisson, dispatch_ctx: Optional[DispatchContext] = None
) -> CondPoisson:
    """Conversion for ``CondPoisson`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CondPoisson(node.scope)


@dispatch(memoize=True)  # type: ignore
def toBase(
    node: CondPoisson, dispatch_ctx: Optional[DispatchContext] = None
) -> BaseCondPoisson:
    """Conversion for ``CondPoisson`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCondPoisson(node.scope)
