"""Contains conditional Gamma leaf node for SPFlow in the ``torch`` backend.
"""
from typing import Callable, List, Optional, Tuple, Type, Union

import torch
import torch.distributions as D
import tensorly as tl

from spflow.base.structure.general.nodes.leaves.parametric.cond_gamma import (
    CondGamma as BaseCondGamma,
)
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_gamma import CondGamma as GeneralCondGamma
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureType, FeatureTypes, MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.nodes.leaf_node import LeafNode


class CondGamma(LeafNode):
    r"""Conditional (univariate) Gamma distribution leaf node in the ``torch`` backend.

    Represents a conditional univariate Exponential distribution, with the following probability distribution function (PDF):

    .. math::
    
        \text{PDF}(x) = \begin{cases} \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x} & \text{if } x > 0\\
                                      0 & \text{if } x <= 0\end{cases}

    where
        - :math:`x` is the input observation
        - :math:`\Gamma` is the Gamma function
        - :math:`\alpha` is the shape parameter
        - :math:`\beta` is the rate parameter
 
    Attributes:
        cond_f:
            Optional callable to retrieve the conditional parameter for the leaf node.
            Its output should be a dictionary containing ``alpha``,``beta`` as a key, and the value should be
            a floating points, scalar NumPy arrays or scalar PyTorch tensors representing the shape and rate parameters, respectively.
    """

    def __init__(self, scope: Scope, cond_f: Optional[Callable] = None) -> None:
        r"""Initializes ``CondExponential`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``alpha``,``beta`` as a key, and the value should be
                a floating points, scalar NumPy arrays or scalar PyTorch tensors representing the shape and rate parameters, respectively.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for CondGamma should be 1, but was {len(scope.query)}.")
        if len(scope.evidence) == 0:
            raise ValueError(f"Evidence scope for CondGamma should not be empty.")

        super().__init__(scope=scope)

        self.set_cond_f(cond_f)
        self.backend = "pytorch"

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``CondGamma`` can represent a single univariate node with ``MetaType.Continuous`` or ``GammaType`` domain.

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

        # leaf is a continuous Gamma distribution
        if not (
            domains[0] == FeatureTypes.Continuous
            or domains[0] == FeatureTypes.Gamma
            or isinstance(domains[0], FeatureTypes.Gamma)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "CondGamma":
        """Creates an instance from a specified signature.

        Returns:
            ``CondGamma`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'CondGamma' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Continuous or domain == FeatureTypes.Gamma or isinstance(domain, FeatureTypes.Gamma):
            pass
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'CondGamma' that was not caught during acception checking."
            )

        return CondGamma(feature_ctx.scope)

    def set_cond_f(self, cond_f: Optional[Callable] = None) -> None:
        r"""Sets the function to retrieve the node's conditonal parameter.

        Args:
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``alpha``,``beta`` as a key, and the value should be
                a floating points, scalar NumPy arrays or scalar PyTorch tensors representing the shape and rate parameters, respectively.
        """
        self.cond_f = cond_f

    def dist(self, alpha: torch.Tensor, beta: torch.Tensor) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Args:
            alpha:
                Scalar PyTorch tensor representing the shape parameter (:math:`\alpha`), greater than 0.
            beta:
                Scalar PyTorch tensor representing the rate parameter (:math:`\beta`), greater than 0.

        Returns:
            ``torch.distributions.Gamma`` instance.
        """
        return D.Gamma(concentration=alpha, rate=beta)

    def retrieve_params(self, data: torch.Tensor, dispatch_ctx: DispatchContext) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Retrieves the conditional parameter of the leaf node.

        First, checks if conditional parameters (``alpha``,``beta``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameters.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameters.

        Args:
            data:
                Two-dimensional PyTorch tensor containing the data to compute the conditional parameters.
                Each row is regarded as a sample.
            dispatch_ctx:
                Dispatch context.

        Returns:
            Tuple of scalar PyTorch tensor representing the shape and rate parameters, respectively.

        Raises:
            ValueError: No way to retrieve conditional parameters or invalid conditional parameters.
        """
        alpha, beta, cond_f = None, None, None

        # check dispatch cache for required conditional parameters 'alpha', 'beta'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if values for 'alpha', 'beta' are specified (highest priority)
            if "alpha" in args:
                alpha = args["alpha"]
            if "beta" in args:
                beta = args["beta"]
            # check if alternative function to provide 'alpha', 'beta' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'l' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'alpha' or 'beta' nor 'cond_f' is specified (via node or arguments)
        if (alpha is None or beta is None) and cond_f is None:
            raise ValueError(
                "'CondGamma' requires either 'alpha' and 'beta' or 'cond_f' to retrieve 'alpha', 'beta' to be specified."
            )

        # if 'alpha' or 'beta' not already specified, retrieve them
        if alpha is None or beta is None:
            params = cond_f(data)
            alpha = params["alpha"]
            beta = params["beta"]

        if isinstance(alpha, float):
            alpha = torch.tensor(alpha)
        if isinstance(beta, float):
            beta = torch.tensor(beta)

        # check if values for 'alpha', 'beta' are valid
        if alpha <= 0.0 or not torch.isfinite(alpha):
            raise ValueError(f"Value of 'alpha' for 'CondGamma' must be greater than 0, but was: {alpha}")
        if beta <= 0.0 or not torch.isfinite(beta):
            raise ValueError(f"Value of 'beta' for 'CondGamma' must be greater than 0, but was: {beta}")

        return alpha, beta

    def check_support(self, data: torch.Tensor, is_scope_data: bool = False) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Gamma distribution, which is:

        .. math::

            \text{supp}(\text{Gamma})=(0,+\infty)

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
        valid[~nan_mask] = self.dist(alpha=torch.tensor(1.0), beta=torch.tensor(1.0)).support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf().squeeze(-1)

        return valid


@dispatch(memoize=True)  # type: ignore
def toTorch(node: BaseCondGamma, dispatch_ctx: Optional[DispatchContext] = None) -> CondGamma:
    """Conversion for ``CondGamma`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CondGamma(node.scope)


@dispatch(memoize=True)  # type: ignore
def toBase(node: CondGamma, dispatch_ctx: Optional[DispatchContext] = None) -> BaseCondGamma:
    """Conversion for ``CondGamma`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCondGamma(node.scope)

@dispatch(memoize=True)  # type: ignore
def updateBackend(leaf_node: CondGamma, dispatch_ctx: Optional[DispatchContext] = None):
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    data = tl.tensor([])
    params = leaf_node.cond_f(data)

    for key in leaf_node.cond_f(params):
        # Update the value for each key
        params[key] = tl.tensor(params[key])
    cond_f = lambda data: params
    return GeneralCondGamma(scope=leaf_node.scope, cond_f=cond_f)