"""Contains conditional Bernoulli leaf node for SPFlow in the ``torch`` backend.
"""
from typing import Callable, List, Optional, Tuple, Type, Union

import torch
import torch.distributions as D
import tensorly as tl

from spflow.base.structure.general.nodes.leaves.parametric.cond_bernoulli import (
    CondBernoulli as BaseCondBernoulli,
)
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_bernoulli import CondBernoulli as GeneralCondBernoulli
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureType, FeatureTypes, MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.nodes.leaf_node import LeafNode


class CondBernoulli(LeafNode):
    r"""Conditional (univariate) Bernoulli distribution leaf node in the ``torch`` backend.

    Represents a conditional univariate Bernoulli distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k)=\begin{cases} p   & \text{if } k=1\\
                                    1-p & \text{if } k=0\end{cases}
        
    where
        - :math:`p` is the success probability
        - :math:`k` is the outcome of the trial (0 or 1)
    
    Attributes:
        cond_f:
            Optional callable to retrieve the conditional parameter for the leaf node.
            Its output should be a dictionary containing ``p`` as a key, and the value should be
            a floating point, scalar NumPy array or scalar PyTorch tensor representing the success probability in :math:`[0,1]`.
    """

    def __init__(self, scope: Scope, cond_f: Optional[Callable] = None) -> None:
        r"""Initializes ``ConditionalBernoulli`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``p`` as a key, and the value should be
                a floating point, scalar NumPy array or scalar PyTorch tensor representing the success probability in :math:`[0,1]`.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'CondBernoulli' should be 1, but was: {len(scope.query)}")
        if len(scope.evidence) == 0:
            raise ValueError(f"Evidence scope for 'CondBernoulli' should not be empty.")

        super().__init__(scope=scope)

        self.set_cond_f(cond_f)
        self.backend = "pytorch"

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``CondBernoulli`` can represent a single univariate node with ``MetaType.Discrete`` or ``BernoulliType`` domain.

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

        # leaf is a discrete Bernoulli distribution
        if not (
            domains[0] == FeatureTypes.Discrete
            or domains[0] == FeatureTypes.Bernoulli
            or isinstance(domains[0], FeatureTypes.Bernoulli)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "CondBernoulli":
        """Creates an instance from a specified signature.

        Returns:
            ``CondBernoulli`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'CondBernoulli' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if (
            domain == MetaType.Discrete
            or domain == FeatureTypes.Bernoulli
            or isinstance(domain, FeatureTypes.Bernoulli)
        ):
            pass
        else:
            raise ValueError(
                f"Unknown signature type {type} for 'CondBernoulli' that was not caught during acception checking."
            )

        return CondBernoulli(feature_ctx.scope)

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
            raise ValueError("'CondBernoulli' requires either 'p' or 'cond_f' to retrieve 'p' to be specified.")

        # if 'p' was not already specified, retrieve it
        if p is None:
            p = cond_f(data)["p"]

        if isinstance(p, float):
            p = torch.tensor(p)

        # check if value for 'p' is valid
        if p < 0.0 or p > 1.0 or not torch.isfinite(p):
            raise ValueError(
                f"Value of 'p' for 'CondBernoulli' distribution must to be between 0.0 and 1.0, but was: {p}"
            )

        return p

    def dist(self, p: torch.Tensor) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Args:
            p:
                Scalar PyTorch tensor representing the success probability of the Bernoulli distribution between zero and one.

        Returns:
            ``torch.distributions.Bernoulli`` instance.
        """
        return D.Bernoulli(probs=p)

    def check_support(self, data: torch.Tensor, is_scope_data: bool = False) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Bernoulli distribution, which is:

        .. math::

            \text{supp}(\text{Bernoulli})=\{0,1\}

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

        valid = torch.ones(scope_data.shape[0], 1, dtype=torch.bool)
        valid[~nan_mask] = self.dist(p=torch.tensor(0.0)).support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf().squeeze(-1)

        return valid


@dispatch(memoize=True)  # type: ignore
def toTorch(node: BaseCondBernoulli, dispatch_ctx: Optional[DispatchContext] = None) -> CondBernoulli:
    """Conversion for ``CondBernoulli`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CondBernoulli(node.scope)


@dispatch(memoize=True)  # type: ignore
def toBase(node: CondBernoulli, dispatch_ctx: Optional[DispatchContext] = None) -> BaseCondBernoulli:
    """Conversion for ``CondBernoulli`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCondBernoulli(node.scope)

@dispatch(memoize=True)  # type: ignore
def updateBackend(leaf_node: CondBernoulli, dispatch_ctx: Optional[DispatchContext] = None):
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    data = tl.tensor([])
    cond_f = None
    if leaf_node.cond_f != None:
        params = leaf_node.cond_f(data)

        for key in leaf_node.cond_f(params):
            # Update the value for each key
            params[key] = tl.tensor(params[key])
        cond_f = lambda data: params
    return GeneralCondBernoulli(scope=leaf_node.scope, cond_f=cond_f)