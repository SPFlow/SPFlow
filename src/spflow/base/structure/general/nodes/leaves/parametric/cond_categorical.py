"""Contains conditional Categorical leaf node for SPFlow in the ``base`` backend.
"""
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import multinomial  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.base.structure.general.nodes.leaf_node import LeafNode
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes, MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext


class CondCategorical(LeafNode):
    r"""Conditional (univariate) Categorical distribution leaf node in the ``base`` backend.

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
            a floating point value representing the success probability in :math:`[0,1]`.
    """

    def __init__(self, scope: Scope, cond_f: Optional[Callable] = None) -> None:
        r"""Initializes ``ConditionalCategorical`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing the keys ``k``, which evaluates to a positive integer, and ``p``, which evaluates to
                a list of floating point values representing the selection probabilities in :math:`[0,1]`.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'CondCategorical' should be 1, but was {len(scope.query)}.")
        if len(scope.evidence) == 0:
            raise ValueError(f"Evidence scope for 'CondCategorical' should not be empty.")

        super().__init__(scope=scope)

        # set optional conditional function
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
                f"Unknown signature type {domain} for 'CondCategorical' that was not caught during acception checking."
            )

        return CondCategorical(feature_ctx.scope)

    def set_cond_f(self, cond_f: Optional[Callable] = None) -> None:
        r"""Sets the function to retrieve the node's conditonal parameter.

        Args:
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``p`` as a key, and the value should be
                a floating point value representing the success probability in :math:`[0,1]`.
        """
        self.cond_f = cond_f

    def retrieve_params(self, data: np.ndarray, dispatch_ctx: DispatchContext) -> Tuple[int, Union[np.ndarray, List[float]]]:
        r"""Retrieves the conditional parameter of the leaf node.

        First, checks if conditional parameters (``k``, ``p``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameter.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameter.

        Args:
            data:
                Two-dimensional NumPy array containing the data to compute the conditional parameters.
                Each row is regarded as a sample.
            dispatch_ctx:
                Dispatch context.

        Returns:
            Tuple of a positive integer representing the number of categories and a list/NumPy array of floating points representing the selection probabilities.

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
            if "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'p' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'p' nor 'cond_f' is specified (via node or arguments)
        if k is None and p is None and cond_f is None:
            raise ValueError("'CondCategorical' requires either 'k' and 'p', or 'cond_f' to retrieve 'k' and 'p' to be specified.")

        # if 'k' and 'p' were not already specified, retrieve them
        if k is None:
            k = cond_f(data)["k"]
        if p is None:
            p = cond_f(data)["p"]

        # check if values for k and 'p' are valid
        if not type(p) in [list, np.ndarray] or len(p) < 1 or not isinstance(p[0], float):
            raise ValueError(f"p needs to be a list or numpy array of at least size 1, but was {p}")
        if k < 1 or not np.isfinite(k):
            raise ValueError(f"Value of k for CondCategorical distribution must be positive integer, but was: {k}")
        if not len(p) == k:
            raise ValueError(f"p needs to be of length k, but were {(len(p), k)}")
        if np.any(np.array(p) < 0.0) or np.any(np.array(p) > 1.0) or not np.any(np.isfinite(np.array(p))):
            raise ValueError(
                f"Value of 'p' for 'CondCategorical' distribution must to be between 0.0 and 1.0, but was: {p}"
            )
        if not np.isclose(np.sum(p), 1.0):
            raise ValueError(f"Probabilities of categories must sum up to 1.0, but were {p}")

        return k, p

    def dist(self, p: List[float]) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Args:
            p:
                List of floating point values representing the selection probabilities of the Categorical distribution between zero and one.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return multinomial(n=1, p=p)

    def check_support(self, data: np.ndarray, is_scope_data: bool = False, dispatch_ctx: DispatchContext = DispatchContext()) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Categorical distribution, which is:

        .. math::

            \text{supp}(\text{Categorical})=\{0, 1, ..., k-1\}

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

        # initialize mask for valid entries
        valid = np.ones(scope_data.shape, dtype=bool)

        # nan entries (regarded as valid)
        nan_mask = np.isnan(scope_data)

        # check for infinite values
        valid[~nan_mask] &= ~np.isinf(scope_data[~nan_mask])

        # check if all values are valid integers
        valid[valid & ~nan_mask] &= np.remainder(scope_data[valid & ~nan_mask], 1) == 0

        # check if values are in valid range
        k, p = self.retrieve_params(np.array([1.]), dispatch_ctx=dispatch_ctx)
        valid[valid & ~nan_mask] &= scope_data[valid & ~nan_mask] < k 
        valid[valid & ~nan_mask] &= scope_data[valid & ~nan_mask] >= 0

        return valid
