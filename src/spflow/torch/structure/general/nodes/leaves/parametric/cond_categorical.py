"""Contains conditional Categorical leaf node for SPFlow in the ``torch`` backend.
"""
from typing import Callable, List, Optional, Tuple, Type, Union
import numpy as np

import torch
import torch.distributions as D

from spflow.base.structure.general.nodes.leaves.parametric.cond_categorical import (
    CondCategorical as BaseCondCategorical,
)
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureType, FeatureTypes, MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.nodes.leaf_node import LeafNode


class CondCategorical(LeafNode):
    r"""Conditional (univariate) Categorical distribution leaf node in the ``torch`` backend.

    Represents a conditional univariate Categorical distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k)= p_k  
        
    where
        - :math:`k` is a positive integer associated with a category
        - :math:`p_k` is the success probability of the associated category k in :math:`[0,1]`

    Attributes:
        cond_f:
            Optional callable to retrieve the conditional parameter for the leaf node.
            Its output should be a dictionary containing ``p`` as a key, and the value should be
            a floating point, scalar NumPy array or scalar PyTorch tensor representing the success probability in :math:`[0,1]`.
    """

    def __init__(self, scope: Scope, cond_f: Optional[Callable] = None) -> None:
        r"""Initializes ``ConditionalCategorical`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``p`` as a key, and the value should be
                a floating point, scalar NumPy array or scalar PyTorch tensor representing the success probability in :math:`[0,1]`.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'CondCategorical' should be 1, but was: {len(scope.query)}")
        if len(scope.evidence) == 0:
            raise ValueError(f"Evidence scope for 'CondCategorical' should not be empty.")

        super().__init__(scope=scope)

        self.set_cond_f(cond_f)

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``CondCategorical`` can represent a single univariate node with ``MetaType.Discrete`` or ``CategoricalType`` domain.

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
        if len(domains) != 1 or len(feature_ctx.scope.query) != len(domains) or len(feature_ctx.scope.evidence) == 0:
            return False

        # leaf is a discrete Categorical distribution
        if not (
            domains[0] == FeatureTypes.Discrete
            or domains[0] == FeatureTypes.Categorical
            or isinstance(domains[0], FeatureTypes.Categorical)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "CondCategorical":
        """Creates an instance from a specified signature.

        Returns:
            ``CondCategorical`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'CondCategorical' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if (
            domain == MetaType.Discrete
            or domain == FeatureTypes.Categorical
            or isinstance(domain, FeatureTypes.Categorical)
        ):
            pass
        else:
            raise ValueError(
                f"Unknown signature type {type} for 'CondCategorical' that was not caught during acception checking."
            )

        return CondCategorical(feature_ctx.scope)

    def set_cond_f(self, cond_f: Optional[Callable] = None) -> None:
        r"""Sets the function to retrieve the node's conditonal parameter.

        Args:
            cond_f:
            Optional callable to retrieve the conditional parameter for the leaf node.
            Its output should be a dictionary containing ``p`` as a key, and the value should be
            a floating point, scalar NumPy array or scalar PyTorch tensor representing the success probability in :math:`[0,1]`.
        """
        self.cond_f = cond_f

    def retrieve_params(self, data: torch.Tensor, dispatch_ctx: DispatchContext) -> Tuple[torch.Tensor]:
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
        k, p, cond_f = None, None, None

        # check dispatch cache for required conditional parameter 'p'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if values for 'k' and 'p' are specified (highest priority)
            if "k" in args:
                k = args["k"]
            if "p" in args:
                p = args["p"]
            # check if alternative function to provide 'p' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'p' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'p' nor 'cond_f' is specified (via node or arguments)
        if k is None and p is None and cond_f is None:
            raise ValueError("'CondCategorical' requires either 'k' and 'p', or 'cond_f' to retrieve 'k' 'p' to be specified.")

        # if 'p' was not already specified, retrieve it
        if k is None:
            k = cond_f(data)["k"]
        if p is None:
            p = cond_f(data)["p"]

        if isinstance(k, int):
            k = torch.tensor(k)
        if isinstance(p, (List, np.ndarray)):
            p = torch.tensor(p)

        # check if values for 'k' and 'p' are valid
        if k < 1 or not np.isfinite(k):
            raise ValueError(f"Value of k for CondCategorical distribution must be positive integer, but was: {k}")
        if torch.any(p < 0.0) or torch.any(p > 1.0) or not torch.all(torch.isfinite(p)):
            raise ValueError(
                f"Value of 'p' for 'CondCategorical' distribution must to be between 0.0 and 1.0, but was: {p}"
            )
        if not torch.isclose(torch.sum(p), torch.tensor(1.0)):
            raise ValueError(f"The sum of all values in p needs to be 1.0, but was: {torch.sum(p)}") 
        if not len(p) == k:
            raise ValueError(f"k and the length of p need to match, but were ({k}, {len(p)})")


        return k, p

    def dist(self, p: torch.Tensor) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Args:
            p:
                PyTorch tensor representing the selection probabilities of the Categorical distribution between zero and one.

        Returns:
            ``torch.distributions.Categorical`` instance.
        """
        return D.Categorical(probs=p)

    def check_support(self, data: torch.Tensor, is_scope_data: bool = False, dispatch_ctx: DispatchContext = DispatchContext()) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Categorical distribution, which is:

        .. math::

            \text{supp}(\text{Categorical})=\{0, 1, ..., k-1\}

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            scope_data:
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

        k, p = self.retrieve_params(torch.tensor(1.0), dispatch_ctx=dispatch_ctx)

        valid = torch.ones(scope_data.shape[0], 1, dtype=torch.bool)
        # valid[~nan_mask] = self.dist(p=torch.zeros_like(p)).support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore
        valid[~nan_mask] = scope_data[~nan_mask] < k

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf().squeeze(-1)

        return valid


@dispatch(memoize=True)  # type: ignore
def toTorch(node: BaseCondCategorical, dispatch_ctx: Optional[DispatchContext] = None) -> CondCategorical:
    """Conversion for ``CondCategorical`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CondCategorical(node.scope)


@dispatch(memoize=True)  # type: ignore
def toBase(node: CondCategorical, dispatch_ctx: Optional[DispatchContext] = None) -> BaseCondCategorical:
    """Conversion for ``CondCategorical`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCondCategorical(node.scope)
