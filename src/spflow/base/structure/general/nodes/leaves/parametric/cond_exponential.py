"""Contains conditional Exponential leaf node for SPFlow in the ``base`` backend.
"""
from typing import Tuple, Optional, Callable, Union, List
import numpy as np
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import MetaType, FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.general.nodes.leaf_node import LeafNode

from scipy.stats import expon  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore


class CondExponential(LeafNode):
    r"""Conditional (univariate) Exponential distribution leaf node in the ``base`` backend.

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
            a floating point value, greater than 0, representing the rate parameter.
    """

    def __init__(self, scope: Scope, cond_f: Optional[Callable] = None) -> None:
        r"""Initializes ``CondExponential`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``l`` as a key, and the value should be
                a floating point value, greater than 0, representing the rate parameter.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'CondExponential' should be 1, but was {len(scope.query)}."
            )
        if len(scope.evidence) == 0:
            raise ValueError(
                f"Evidence scope for 'CondExponential' should not be empty."
            )

        super().__init__(scope=scope)

        # set optional conditional function
        self.set_cond_f(cond_f)

    @classmethod
    def accepts(self, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``CondExponential`` can represent a single univariate node with ``MetaType.Continuous`` or ``ExponentialType`` domain.

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
        if (
            len(domains) != 1
            or len(feature_ctx.scope.query) != len(domains)
            or len(feature_ctx.scope.evidence) == 0
        ):
            return False

        # leaf is a discrete Exponential distribution
        if not (
            domains[0] == FeatureTypes.Continuous
            or domains[0] == FeatureTypes.Exponential
            or isinstance(domains[0], FeatureTypes.Exponential)
        ):
            return False

        return True

    @classmethod
    def from_signatures(
        self, signatures: List[FeatureContext]
    ) -> "CondExponential":
        """Creates an instance from a specified signature.

        Returns:
            ``CondExponential`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not self.accepts(signatures):
            raise ValueError(
                f"'CondExponential' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if (
            domain == MetaType.Continuous
            or domain == FeatureTypes.Exponential
            or isinstance(domain, FeatureTypes.Exponential)
        ):
            pass
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'CondExponential' that was not caught during acception checking."
            )

        return CondExponential(feature_ctx.scope)

    def set_cond_f(self, cond_f: Optional[Callable] = None) -> None:
        r"""Sets the function to retrieve the node's conditonal parameter.

        Args:
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``l`` as a key, and the value should be
                a floating point value, greater than 0, representing the rate parameter.
        """
        self.cond_f = cond_f

    def retrieve_params(
        self, data: np.ndarray, dispatch_ctx: DispatchContext
    ) -> Tuple[Union[np.ndarray, float]]:
        r"""Retrieves the conditional parameter of the leaf node.

        First, checks if conditional parameter (``l``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameter.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameter.

        Args:
            data:
                Two-dimensional NumPy array containing the data to compute the conditional parameters.
                Each row is regarded as a sample.
            dispatch_ctx:
                Dispatch context.

        Returns:
            Floating point or scalar NumPy array representing the rate parameter.

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

        # check if value for 'l' is valid
        if l <= 0.0 or not np.isfinite(l):
            raise ValueError(
                f"Value of 'l' for conditional Exponential distribution must be greater than 0, but was: {l}"
            )

        return l

    def dist(self, l: float) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Args:
            l:
                Floating point value representing the rate parameter (:math:`\lambda`) of the Exponential distribution (must be greater than 0).

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return expon(scale=1.0 / l)

    def check_support(
        self, data: np.ndarray, is_scope_data: bool = False
    ) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Exponential distribution, which is:

        .. math::

            \text{supp}(\text{Exponential})=[0,+\infty)

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

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scope.query):
            raise ValueError(
                f"Expected 'scope_data' to be of shape (n,{len(self.scope.query)}), but was: {scope_data.shape}"
            )

        valid = np.ones(scope_data.shape, dtype=bool)

        # nan entries (regarded as valid)
        nan_mask = np.isnan(scope_data)

        # check for infinite values
        valid[~nan_mask] &= ~np.isinf(scope_data[~nan_mask])

        # check if values are in valid range
        valid[valid & ~nan_mask] &= scope_data[valid & ~nan_mask] >= 0

        return valid
