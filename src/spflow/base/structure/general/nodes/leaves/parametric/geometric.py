"""Contains Geometric leaf node for SPFlow in the ``base`` backend.
"""
from typing import List, Tuple

import numpy as np
from scipy.stats import geom  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.base.structure.general.nodes.leaf_node import LeafNode
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes, MetaType
from spflow.meta.data.scope import Scope


class Geometric(LeafNode):
    r"""(Univariate) Geometric distribution leaf node in the ``base`` backend.

    Represents an univariate Geometric distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) =  p(1-p)^{k-1}

    where
        - :math:`k` is the number of trials
        - :math:`p` is the success probability of each trial

    Attributes:
        p:
            Floating points representing the probability of success in the range :math:`(0,1]`.
    """

    def __init__(self, scope: Scope, p: float = 0.5) -> None:
        r"""Initializes ``Geometric`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            p:
                Floating points representing the probability of success in the range :math:`(0,1]`.
                Defaults to 0.5.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'Geometric' should be 1, but was {len(scope.query)}."
            )
        if len(scope.evidence) != 0:
            raise ValueError(
                f"Evidence scope for 'Geometric' should be empty, but was {scope.evidence}."
            )

        super().__init__(scope=scope)
        self.set_params(p)

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Geometric`` can represent a single univariate node with ``MetaType.Discrete`` or ``GeometricType`` domain.

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

        # leaf is a discrete Geometric distribution
        if not (
            domains[0] == FeatureTypes.Discrete
            or domains[0] == FeatureTypes.Geometric
            or isinstance(domains[0], FeatureTypes.Geometric)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "Geometric":
        """Creates an instance from a specified signature.

        Returns:
            ``Geometric`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'Geometric' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Discrete:
            p = 0.5
        elif domain == FeatureTypes.Geometric:
            # instantiate object
            p = domain().p
        elif isinstance(domain, FeatureTypes.Geometric):
            p = domain.p
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Geometric' that was not caught during acception checking."
            )

        return Geometric(feature_ctx.scope, p=p)

    @property
    def dist(self) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return geom(p=self.p)

    def set_params(self, p: float) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            p:
                Floating points representing the probability of success in the range :math:`(0,1]`.
        """
        if p <= 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of 'p' for 'Geometric' must to be greater than 0.0 and less or equal to 1.0, but was: {p}"
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

        Determines whether or note instances are part of the support of the Geometric distribution, which is:

        .. math::

            \text{supp}(\text{Geometric})=\mathbb{N}\setminus\{0\}

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
        valid[valid & ~nan_mask] &= scope_data[valid & ~nan_mask] >= 1

        return valid
