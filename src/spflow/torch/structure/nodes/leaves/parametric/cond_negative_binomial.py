# -*- coding: utf-8 -*-
"""Contains conditional Negative Binomial leaf node for SPFlow in the ``torch`` backend.
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
from spflow.base.structure.nodes.leaves.parametric.cond_negative_binomial import (
    CondNegativeBinomial as BaseCondNegativeBinomial,
)


class CondNegativeBinomial(LeafNode):
    r"""Conditional (univariate) Negative Binomial distribution leaf node in the ``torch`` backend.

    Represents a conditional univariate Negative Binomial distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \binom{k+n-1}{n-1}p^n(1-p)^k

    where
        - :math:`k` is the number of failures
        - :math:`n` is the maximum number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Attributes:
        n:
            Integer representing the number of successes (greater or equal to 0).
        cond_f:
            Optional callable to retrieve the conditional parameter for the leaf node.
            Its output should be a dictionary containing ``p`` as a key, and the value should be
            a floating point, scalar NumPy array or scalar PyTorch tensor representing the success probability in :math:`(0,1]`.
    """

    def __init__(
        self, scope: Scope, n: int, cond_f: Optional[Callable] = None
    ) -> None:
        r"""Initializes ``CondBernoulli`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            n:
                Integer representing the number of successes (greater or equal to 0).
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``p`` as a key, and the value should be
                a floating point, scalar NumPy array or scalar PyTorch tensor representing the success probability in :math:`(0,1]`.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'CondNegativeBinomial' should be 1, but was: {len(scope.query)}."
            )
        if len(scope.evidence):
            raise ValueError(
                f"Evidence scope for 'CondNegativeBinomial' should be empty, but was {scope.evidence}."
            )

        super(CondNegativeBinomial, self).__init__(scope=scope)

        # register number of trials n as torch buffer (should not be changed)
        self.register_buffer("n", torch.empty(size=[]))

        # set parameters
        self.set_params(n)

        self.set_cond_f(cond_f)

    def set_cond_f(self, cond_f: Optional[Callable] = None) -> None:
        r"""Sets the function to retrieve the node's conditonal parameter.

        Args:
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``p`` as a key, and the value should be
                a floating point, scalar NumPy array or scalar PyTorch tensor representing the success probability in :math:`(0,1]`.
        """
        self.cond_f = cond_f

    def dist(self, p: torch.Tensor) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Args:
            p:
                Scalar PyTorch tensor representing the success probability of each trial in the range :math:`(0,1]`.

        Returns:
            ``torch.distributions.NegativeBinomial`` instance.
        """
        return D.NegativeBinomial(total_count=self.n, probs=torch.ones(1) - p)

    def set_params(self, n: int) -> None:
        """Sets the parameters for the represented distribution.

        Args:
            n:
                Integer representing the number of successes (greater or equal to 0).
        """
        if n < 0 or not np.isfinite(n):
            raise ValueError(
                f"Value of n for NegativeBinomial distribution must to greater of equal to 0, but was: {n}"
            )

        if not (np.remainder(n, 1.0) == 0.0):
            raise ValueError(
                f"Value of n for NegativeBinomial distribution must be (equal to) an integer value, but was: {n}"
            )

        self.n.data = torch.tensor(int(n))  # type: ignore

    def retrieve_params(
        self, data: torch.Tensor, dispatch_ctx: DispatchContext
    ) -> torch.Tensor:
        r"""Retrieves the conditional parameter of the leaf node.

        First, checks if conditional parameter (``p``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameter.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameter.

        Args:
            data:
                Two-dimensional PyTorch tensor containing the data to compute the conditional parameters.
                Each row is regarded as a sample.
            dispatch_ctx:
                Dispatch context.

        Returns:
            Scalar PyTorch tensor representing the success probability.

        Raises:
            ValueError: No way to retrieve conditional parameters or invalid conditional parameters.
        """
        p, cond_f = None, None

        # check dispatch cache for required conditional parameter 'p'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if a value for 'p' is specified (highest priority)
            if "p" in args:
                p = args["p"]
            # check if alternative function to provide 'p' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'p' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'p' nor 'cond_f' is specified (via node or arguments)
        if p is None and cond_f is None:
            raise ValueError(
                "'CondBinomial' requires either 'p' or 'cond_f' to retrieve 'p' to be specified."
            )

        # if 'p' was not already specified, retrieve it
        if p is None:
            p = cond_f(data)["p"]

        if isinstance(p, float):
            p = torch.tensor(p)

        # check if value for 'p' is valid
        if p <= 0.0 or p > 1.0 or not torch.isfinite(p):
            raise ValueError(
                f"Value of 'p' for 'CondNegativeBinomial' must to be between 0.0 (excluding) and 1.0 (including), but was: {p}"
            )

        return p

    def get_params(self) -> Tuple[int]:
        """Returns the parameters of the represented distribution.

        Returns:
            Integer representing the number of successes (greater or equal to 0).
        """
        return (self.n.data.cpu().numpy(),)

    def check_support(
        self, data: torch.Tensor, is_scope_data: bool = False
    ) -> torch.Tensor:
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
        valid[~nan_mask] = self.dist(torch.tensor(0.5)).support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore

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
    node: BaseCondNegativeBinomial,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> CondNegativeBinomial:
    """Conversion for ``CondNegativeBinomial`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CondNegativeBinomial(node.scope, *node.get_params())


@dispatch(memoize=True)  # type: ignore
def toBase(
    node: CondNegativeBinomial, dispatch_ctx: Optional[DispatchContext] = None
) -> BaseCondNegativeBinomial:
    """Conversion for ``CondNegativeBinomial`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCondNegativeBinomial(node.scope, *node.get_params())
