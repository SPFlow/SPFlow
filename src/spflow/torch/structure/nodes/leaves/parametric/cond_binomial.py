# -*- coding: utf-8 -*-
"""Contains conditional Binomial leaf node for SPFlow in the ``torch`` backend.
"""
import torch
import torch.distributions as D
from typing import Tuple, Optional, Callable, List, Union, Type
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import MetaType, FeatureType, FeatureTypes
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.nodes.node import LeafNode
from spflow.base.structure.nodes.leaves.parametric.cond_binomial import (
    CondBinomial as BaseCondBinomial,
)


class CondBinomial(LeafNode):
    r"""Conditional (univariate) Binomial distribution leaf node in the ``torch`` backend.

    Represents a conditional univariate Binomial distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \binom{n}{k}p^k(1-p)^{n-k}

    where
        - :math:`p` is the success probability of each trial
        - :math:`n` is the number of total trials
        - :math:`k` is the number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Attributes:
        n:
            Scalar PyTorch tensor representing the number of i.i.d. Bernoulli trials (greater or equal to 0).
        cond_f:
            cond_f:
            Optional callable to retrieve the conditional parameter for the leaf node.
            Its output should be a dictionary containing ``p`` as a key, and the value should be
            a floating point, scalar NumPy array or scalar PyTorch tensor representing the success probability in :math:`[0,1]`.
    """

    def __init__(
        self, scope: Scope, n: int, cond_f: Optional[Callable] = None
    ) -> None:
        r"""Initializes ``ConditionalBernoulli`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            n:
                Integer representing the number of i.i.d. Bernoulli trials (greater or equal to 0).
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``p`` as a key, and the value should be
                a floating point, scalar NumPy array or scalar PyTorch tensor representing the success probability in :math:`[0,1]`.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'CondBinomial' should be 1, but was {len(scope.query)}."
            )
        if len(scope.evidence) == 0:
            raise ValueError(
                f"Evidence scope for 'CondBinomial' should not be empty."
            )

        super(CondBinomial, self).__init__(scope=scope)

        # register number of trials n as torch buffer (should not be changed)
        self.register_buffer("n", torch.empty(size=[]))

        # set parameters
        self.set_params(n)

        self.set_cond_f(cond_f)

    @classmethod
    def accepts(self, signatures: List[Tuple[List[Union[MetaType, FeatureType, Type[FeatureType]]], Scope]]) -> bool:
        """TODO"""
        # leaf only has one output
        if len(signatures) != 1:
            return False

        # get single output signature
        types, scope = signatures[0]

        # leaf is a single non-conditional univariate node
        if len(types) != 1 or len(scope.query) != len(types) or len(scope.evidence) == 0:
            return False

        # leaf is a discrete Binomial distribution
        # NOTE: only accept instances of 'FeatureTypes.Binomial', otherwise required parameter 'n' is not specified. Reject 'FeatureTypes.Discrete' for the same reason.
        if not isinstance(types[0], FeatureTypes.Binomial):
            return False

        return True

    @classmethod
    def from_signatures(self, signatures: List[Tuple[List[Union[MetaType, FeatureType, Type[FeatureType]]], Scope]]) -> "CondBinomial":
        """TODO"""
        if not self.accepts(signatures):
            raise ValueError(f"'CondBinomial' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        types, scope = signatures[0]
        type = types[0]

        # read or initialize parameters
        if isinstance(type, FeatureTypes.Binomial):
            n = type.n
        else:
            raise ValueError(f"Unknown signature type {type} for 'CondBinomial' that was not caught during acception checking.")

        return CondBinomial(scope, n=n)

    def set_cond_f(self, cond_f: Optional[Callable] = None) -> None:
        r"""Sets the function to retrieve the node's conditonal parameter.

        Args:
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``p`` as a key, and the value should be
                a floating point, scalar NumPy array or scalar PyTorch tensor representing the success probability in :math:`[0,1]`.
        """
        self.cond_f = cond_f

    def retrieve_params(
        self, data: torch.Tensor, dispatch_ctx: DispatchContext
    ) -> Tuple[torch.Tensor]:
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
                "'CondBernoulli' requires either 'p' or 'cond_f' to retrieve 'p' to be specified."
            )

        # if 'p' was not already specified, retrieve it
        if p is None:
            p = cond_f(data)["p"]

        if isinstance(p, float):
            p = torch.tensor(p)

        # check if value for 'p' is valid
        if p < 0.0 or p > 1.0 or not torch.isfinite(p):
            raise ValueError(
                f"Value of p for CondBinomial distribution must to be between 0.0 and 1.0, but was: {p}"
            )

        return p

    def get_params(self) -> Tuple[int]:
        """Returns the parameters of the represented distribution.

        Returns:
            Integer number representing the number of i.i.d. Bernoulli trials and the floating point value representing the success probability.
        """
        return (self.n.data.cpu().numpy(),)  # type: ignore

    def dist(self, p: torch.Tensor) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Args:
            p:
                Scalar PyTorch tensor representing the success probability of each trial between zero and one.

        Returns:
            ``torch.distributions.Binomial`` instance.
        """
        return D.Binomial(total_count=self.n, probs=p)

    def set_params(self, n: int) -> None:
        """Sets the parameters for the represented distribution.

        Args:
            n:
                Integer representing the number of i.i.d. Bernoulli trials (greater or equal to 0).
        """
        if isinstance(n, float):
            if not n.is_integer():
                raise ValueError(
                    f"Value of n for CondBinomial distribution must be (equal to) an integer value, but was: {n}"
                )
            n = torch.tensor(int(n))
        elif isinstance(n, int):
            n = torch.tensor(n)
        if n < 0 or not torch.isfinite(n):
            raise ValueError(
                f"Value of n for CondBinomial distribution must to greater of equal to 0, but was: {n}"
            )

        self.n.data = torch.tensor(int(n))  # type: ignore

    def check_support(
        self, data: torch.Tensor, is_scope_data: bool = False
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Binomial distribution, which is:

        .. math::

            \text{supp}(\text{Binomial})=\{0,\hdots,n\}

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
        valid[~nan_mask] = self.dist(p=torch.tensor(0.0)).support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore

        # check for infinite values
        valid[~nan_mask & valid] &= (
            ~scope_data[~nan_mask & valid].isinf().squeeze(-1)
        )

        return valid


@dispatch(memoize=True)  # type: ignore
def toTorch(
    node: BaseCondBinomial, dispatch_ctx: Optional[DispatchContext] = None
) -> CondBinomial:
    """Conversion for ``CondBinomial`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CondBinomial(node.scope, *node.get_params())


@dispatch(memoize=True)  # type: ignore
def toBase(
    node: CondBinomial, dispatch_ctx: Optional[DispatchContext] = None
) -> BaseCondBinomial:
    """Conversion for ``CondBinomial`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCondBinomial(node.scope, *node.get_params())
