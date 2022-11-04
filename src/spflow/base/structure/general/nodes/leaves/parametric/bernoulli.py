# -*- coding: utf-8 -*-
"""Contains Bernoulli leaf node for SPFlow in the ``base`` backend.
"""
from typing import Tuple, List
import numpy as np
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import MetaType, FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.base.structure.general.nodes.leaf_node import LeafNode

from scipy.stats import bernoulli  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore


class Bernoulli(LeafNode):
    r"""(Univariate) Bernoulli distribution leaf node in the ``base`` backend.

    Represents an univariate Bernoulli distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k)=\begin{cases} p   & \text{if } k=1\\
                                    1-p & \text{if } k=0\end{cases}
        
    where
        - :math:`p` is the success probability in :math:`[0,1]`
        - :math:`k` is the outcome of the trial (0 or 1)

    Attributes:
        p:
            Floating point value representing the success probability of the Bernoulli distribution.
    """

    def __init__(self, scope: Scope, p: float = 0.5) -> None:
        r"""Initializes ``Bernoulli`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            p:
                Floating point value representing the success probability of the Bernoulli distribution between zero and one.
                Defaults to 0.5.

        Raises:
            ValueError: Invalid arguments.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'Bernoulli' should be 1, but was {len(scope.query)}."
            )
        if len(scope.evidence) != 0:
            raise ValueError(
                f"Evidence scope for 'Bernoulli' should be empty, but was {scope.evidence}."
            )

        super(Bernoulli, self).__init__(scope=scope)

        # set parameters
        self.set_params(p)

    @classmethod
    def accepts(self, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Bernoulli`` can represent a single univariate node with ``MetaType.Discrete`` or ``BernoulliType`` domain.

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
            or len(feature_ctx.scope.evidence) != 0
        ):
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
    def from_signatures(self, signatures: List[FeatureContext]) -> "Bernoulli":
        """Creates an instance from a specified signature.

        Returns:
            ``Bernoulli`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not self.accepts(signatures):
            raise ValueError(
                f"'Bernoulli' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Discrete:
            p = 0.5
        elif domain == FeatureTypes.Bernoulli:
            # instantiate object
            p = domain().p
        elif isinstance(domain, FeatureTypes.Bernoulli):
            p = domain.p
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Bernoulli' that was not caught during acception checking."
            )

        return Bernoulli(feature_ctx.scope, p=p)

    @property
    def dist(self) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return bernoulli(p=self.p)

    def set_params(self, p: float) -> None:
        """Sets the parameters for the represented distribution.

        Args:
            p:
                Floating point value representing the success probability of the Bernoulli distribution between zero and one.
        """
        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of 'p' for 'Bernoulli' must to be between 0.0 and 1.0, but was: {p}"
            )

        self.p = p

    def get_params(self) -> Tuple[float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Floating point value representing the success probability.
        """
        return (self.p,)

    def check_support(
        self, data: np.ndarray, is_scope_data: bool = False
    ) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Bernoulli distribution, which is:

        .. math::

            \text{supp}(\text{Bernoulli})=\{0,1\}

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
        valid[valid & ~nan_mask] &= (
            np.remainder(scope_data[valid & ~nan_mask], 1) == 0
        )

        # check if values are in valid range
        valid[valid & ~nan_mask] &= (scope_data[valid & ~nan_mask] >= 0) & (
            scope_data[valid & ~nan_mask] <= 1
        )

        return valid
