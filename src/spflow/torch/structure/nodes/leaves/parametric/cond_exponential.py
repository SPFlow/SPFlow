# -*- coding: utf-8 -*-
"""Contains conditional Binomial leaf node for SPFlow in the ``torch`` backend.
"""
import torch
import torch.distributions as D
from typing import List, Tuple, Optional, Callable, Union, Type
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import MetaType, FeatureType, FeatureTypes
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.nodes.node import LeafNode
from spflow.base.structure.nodes.leaves.parametric.cond_exponential import (
    CondExponential as BaseCondExponential,
)


class CondExponential(LeafNode):
    r"""Conditional (univariate) Exponential distribution leaf node in the ``torch`` backend.

    Represents a conditional univariate Exponential distribution, with the following probability distribution function (PDF):

    .. math::
        
        \text{PDF}(x) = \begin{cases} \lambda e^{-\lambda x} & \text{if } x > 0\\
                                      0                      & \text{if } x <= 0\end{cases}
    
    where
        - :math:`x` is the input observation
        - :math:`\lambda` is the rate parameter
    
    Attributes:
        cond_f:
            Optional callable to retrieve the conditional parameter for the leaf node.
            Its output should be a dictionary containing ``l`` as a key, and the value should be
            a floating point, scalar NumPy array or scalar PyTorch tensor representing the rate parameter.
    """

    def __init__(self, scope: Scope, cond_f: Optional[Callable] = None) -> None:
        r"""Initializes ``CondExponential`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``l`` as a key, and the value should be
                a floating point, scalar NumPy array or scalar PyTorch tensor representing the rate parameter.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'CondExponential' should be 1, but was {len(scope.query)}."
            )
        if len(scope.evidence) == 0:
            raise ValueError(
                f"Evidence scope for 'CondExponential' should not be empty."
            )

        super(CondExponential, self).__init__(scope=scope)

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

        # leaf is a discrete Exponential distribution
        if not (types[0] == FeatureTypes.Continuous or types[0] == FeatureTypes.Exponential or isinstance(types[0], FeatureTypes.Exponential)):
            return False

        return True

    @classmethod
    def from_signatures(self, signatures: List[Tuple[List[Union[MetaType, FeatureType, Type[FeatureType]]], Scope]]) -> "CondExponential":
        """TODO"""
        if not self.accepts(signatures):
            raise ValueError(f"'CondExponential' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        types, scope = signatures[0]
        type = types[0]

        # read or initialize parameters
        if type == MetaType.Continuous or type == FeatureTypes.Exponential or isinstance(type, FeatureTypes.Exponential):
            pass
        else:
            raise ValueError(f"Unknown signature type {type} for 'CondExponential' that was not caught during acception checking.")

        return CondExponential(scope)

    def set_cond_f(self, cond_f: Callable) -> None:
        r"""Sets the function to retrieve the node's conditonal parameter.

        Args:
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``l`` as a key, and the value should be
                a floating point, scalar NumPy array or scalar PyTorch tensor representing the rate parameter.
        """
        self.cond_f = cond_f

    def retrieve_params(
        self, data: torch.Tensor, dispatch_ctx: DispatchContext
    ) -> Tuple[torch.Tensor]:
        r"""Retrieves the conditional parameter of the leaf node.

        First, checks if conditional parameter (``l``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameter.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameter.

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
                "'CondExponential' requires either 'l' or 'cond_f' to retrieve 'l' to be specified."
            )

        # if 'l' was not already specified, retrieve it
        if l is None:
            l = cond_f(data)["l"]

        if isinstance(l, float):
            l = torch.tensor(l)

        # check if value for 'l' is valid
        if l <= 0.0 or not torch.isfinite(l):
            raise ValueError(
                f"Value of 'l' for conditional Exponential distribution must be greater than 0, but was: {l}"
            )

        return l

    def dist(self, l: torch.Tensor) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Args:
            l:
                Scalar PyTorch tensor representing the rate parameter (:math:`\lambda`) of the Exponential distribution (must be greater than 0).

        Returns:
            ``torch.distributions.Exponential`` instance.
        """
        return D.Exponential(rate=l)

    def check_support(
        self, data: torch.Tensor, is_scope_data: bool = False
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Exponential distribution, which is:

        .. math::

            \text{supp}(\text{Exponential})=[0,+\infty)

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
        valid[~nan_mask] = self.dist(l=torch.tensor(0.5)).support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore

        # check for infinite values
        valid[~nan_mask & valid] &= (
            ~scope_data[~nan_mask & valid].isinf().squeeze(-1)
        )

        return valid


@dispatch(memoize=True)  # type: ignore
def toTorch(
    node: BaseCondExponential, dispatch_ctx: Optional[DispatchContext] = None
) -> CondExponential:
    """Conversion for ``CondExponential`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CondExponential(node.scope)


@dispatch(memoize=True)  # type: ignore
def toBase(
    node: CondExponential, dispatch_ctx: Optional[DispatchContext] = None
) -> BaseCondExponential:
    """Conversion for ``CondExponential`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCondExponential(node.scope)
