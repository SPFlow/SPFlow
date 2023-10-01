"""Contains conditional Poisson leaf node for SPFlow in the ``base`` backend.
"""
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import tensorly as tl
from scipy.stats import poisson  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.tensorly.structure.spn.nodes.leaves.parametric import CondPoisson as GeneralCondPoisson
from spflow.base.structure.general.nodes.leaf_node import LeafNode
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes, MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.dispatch import dispatch

class CondPoisson(LeafNode):
    r"""Conditional (univariate) Poisson distribution leaf node in the ``base`` backend.

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
            a floating point value representing the rate parameter, greater than or equal to 0.
    """

    def __init__(self, scope: Scope, cond_f: Optional[Callable] = None) -> None:
        r"""Initializes ``CondPoisson`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``l`` as a key, and the value should be
                a floating point value representing the rate parameter, greater than or equal to 0.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'CondPoisson' should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence) == 0:
            raise ValueError(f"Evidence scope for 'CondPoisson' should be empty.")

        super().__init__(scope=scope)

        # set optional conditional function
        self.set_cond_f(cond_f)

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``CondPoisson`` can represent a single univariate node with ``MetaType.Discrete`` or ``PoissonType`` domain.

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

        # leaf is a discrete Poisson distribution
        if not (
            domains[0] == FeatureTypes.Discrete
            or domains[0] == FeatureTypes.Poisson
            or isinstance(domains[0], FeatureTypes.Poisson)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "CondPoisson":
        """Creates an instance from a specified signature.

        Returns:
            ``CondPoisson`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'CondPoisson' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Discrete or domain == FeatureTypes.Poisson or isinstance(domain, FeatureTypes.Poisson):
            pass
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'CondPoisson' that was not caught during acception checking."
            )

        return CondPoisson(feature_ctx.scope)

    def set_cond_f(self, cond_f: Optional[Callable] = None) -> None:
        r"""Sets the function to retrieve the node's conditonal parameter.

        Args:
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``l`` as a key, and the value should be
                a floating point value representing the rate parameter, greater than or equal to 0.
        """
        self.cond_f = cond_f

    def retrieve_params(self, data: np.ndarray, dispatch_ctx: DispatchContext) -> Tuple[Union[np.ndarray, int, float]]:
        r"""Retrieves the conditional parameter of the leaf node.

        First, checks if conditional parameters (``l``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameters.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameters.

        Args:
            data:
                Two-dimensional NumPy array containing the data to compute the conditional parameters.
                Each row is regarded as a sample.
            dispatch_ctx:
                Dispatch context.

        Returns:
            Floating point value or scalar NumPy array representing the rate parameter.

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
            raise ValueError("'CondPoisson' requires either 'l' or 'cond_f' to retrieve 'l' to be specified.")

        # if 'l' was not already specified, retrieve it
        if l is None:
            l = cond_f(data)["l"]

        # check if value for 'l' is valid
        if not np.isfinite(l):
            raise ValueError(f"Value of 'l' for 'CondPoisson' must be finite, but was: {l}")

        if l < 0:
            raise ValueError(f"Value of 'l' for 'CondPoisson' must be non-negative, but was: {l}")

        return l

    def dist(self, l: float) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Args:
            l:
                Floating point value representing the rate parameter (:math:`\lambda`), expected value and variance of the Poisson distribution (must be greater than or equal to 0).

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return poisson(mu=l)

    def check_support(self, data: np.ndarray, is_scope_data: bool = False) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Poisson distribution, which is:

        .. math::

            \text{supp}(\text{Poisson})=\mathbb{N}\cup\{0\}

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional NumPy array containing sample instances.
                Each row is regarded as a sample.
                Unless ``is_scope_data`` is set to True, it is assumed that the relevant data is located in the columns corresponding to the scope indices.
            is_scope_data:
                Boolean indicating if the given data already contains the relevant data for the leaf's scope in the correct order (True) or if it needs to be extracted from the full data set.
                Defaults to False.

        Returns:
            Two-dimensional NumPy array indicating for each instance, whether they are part of the support (True) or not (False).
        """
        if is_scope_data:
            scope_data = data
        else:
            # select relevant data for scope
            scope_data = data[:, self.scope.query]

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scopes_out[0].query):
            raise ValueError(
                f"Expected 'scope_data' to be of shape (n,{len(self.scopes_out[0].query)}), but was: {scope_data.shape}"
            )

        valid = np.ones(scope_data.shape, dtype=bool)

        # nan entries (regarded as valid)
        nan_mask = np.isnan(scope_data)

        # check for infinite values
        valid[~nan_mask] &= ~np.isinf(scope_data[~nan_mask])

        # check if all values are valid integers
        valid[valid & ~nan_mask] &= np.remainder(scope_data[valid & ~nan_mask], 1) == 0

        # check if values are in valid range
        valid[valid & ~nan_mask] &= scope_data[valid & ~nan_mask] >= 0

        return valid

@dispatch(memoize=True)  # type: ignore
def updateBackend(leaf_node: CondPoisson, dispatch_ctx: Optional[DispatchContext] = None):
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
    return GeneralCondPoisson(scope=leaf_node.scope, cond_f=cond_f)
