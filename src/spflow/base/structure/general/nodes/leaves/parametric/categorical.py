"""Contains Categorical leaf node for SPFlow in the ``base`` backend.
"""
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.stats import multinomial  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.base.structure.general.nodes.leaf_node import LeafNode
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes, MetaType
from spflow.meta.data.scope import Scope


class Categorical(LeafNode):
    r"""(Univariate) Categorical distribution leaf node in the ``base`` backend.

    Represents an univariate Categorical distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k)= p_k  
        
    where
        - :math:`k` is a positive integer associated with a category
        - :math:`p_k` is the success probability of the associated category k in :math:`[0,1]`
        
    Attributes:
        k:
            The number of categories
        p:
            A list of floating point values representing the probability of each category
    """

    def __init__(self, scope: Scope, k: int = 2, p: Optional[Union[List[float], np.ndarray]] = None) -> None:
        r"""Initializes ``Categorical`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            k: 
                A positive integer representing the number of categories.
                Defaults to 2.
            p:
                A list of floating point values representing the probability that the k-th category is selected, each in the range [0, 1].
                Defaults to uniformly distributed over k categories.

        Raises:
            ValueError: Invalid arguments.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'Categorical' should be 1, but was {len(scope.query)}.")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'Categorical' should be empty, but was {scope.evidence}.")
        if k is None or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError(f"Number of categories needs to a positive integer, but was: {(type(k), k)}")
        super().__init__(scope=scope)

        if p is None: 
            p = [1.0/k for i in range(k)]
        if len(p) != k:
            raise ValueError(f"p needs to be the length of k, but len(p) and k were: ({len(p)}, {k})")

        # set parameters
        self.set_params(k, p)

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Categorical`` can represent a single univariate node with ``MetaType.Discrete`` or ``CategoricalType`` domain.

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
        if len(domains) != 1 or len(feature_ctx.scope.query) != len(domains) or len(feature_ctx.scope.evidence) != 0:
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
    def from_signatures(cls, signatures: List[FeatureContext]) -> "Categorical":
        """Creates an instance from a specified signature.

        Returns:
            ``Categorical`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'Categorical' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Discrete:
            k = 2
            p = [0.5, 0.5]
        elif domain == FeatureTypes.Categorical:
            # instantiate object
            k = domain().k
            p = domain().p
        elif isinstance(domain, FeatureTypes.Categorical):
            k = domain.k
            p = domain.p
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Categorical' that was not caught during acception checking."
            )

        return Categorical(feature_ctx.scope, k=k, p=p)

    @property
    def dist(self) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        # the categorical distribution is a special case of the multinomial distribution with n=1
        return multinomial(n=1, p=self.p)

    def set_params(self, k: int, p: List[float]) -> None:
        """Sets the parameters for the represented distribution.

        Args:
            k:
                A positive integer representing the number of categories.
            p:
                A list of floating point values representing the probability choosing one of the categories of the Categorical distribution between zero and one.
        """
        p = np.array(p)
        if (not all(p >= 0.0)) or (not all(p <= 1.0)) or (not all(np.isfinite(p))):
            raise ValueError(f"All values of 'p' for 'Categorical' must to be between 0.0 and 1.0, but were: {p}")
        if not np.isclose(sum(p), 1.0):
            raise ValueError(f"The sum of all values in p needs to be 1.0, but was: {sum(p)}") 
        if (not isinstance(k, (int, np.integer))) or k < 1 or not np.isfinite(k):
            raise ValueError(f"k needs to be a positive integer, but was {k}, {type(k)}")
        if not len(p) == k:
            raise ValueError(f"k and the length of p need to match, but were ({k}, {len(p)})")

        self.k = k
        self.p = p

    def get_params(self) -> Tuple[int, float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Floating point value representing the success probability.
        """
        return (self.k, self.p,)

    def check_support(self, data: np.ndarray, is_scope_data: bool = False) -> np.ndarray:
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
            Two dimensional NumPy array indicating for each instance, whether they are part of the support (True) or not (False).
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
        valid[valid & ~nan_mask] &= (scope_data[valid & ~nan_mask] >= 0) & (scope_data[valid & ~nan_mask] < self.k)

        return valid
