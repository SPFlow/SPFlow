"""Contains Poisson leaf node for SPFlow in the ``base`` backend.
"""
from typing import List, Tuple

import tensorly as tl
from ......utils.helper_functions import tl_isnan, tl_isinf, tl_isfinite
from scipy.stats import poisson  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.tensorly.structure.general.nodes.leaf_node import LeafNode
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes, MetaType
from spflow.meta.data.scope import Scope


class Poisson(LeafNode):
    r"""(Univariate) Poisson distribution leaf node in the ``base`` backend.

    Represents a univariate Poisson distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \lambda^k\frac{e^{-\lambda}}{k!}

    where
        - :math:`k` is the number of occurrences
        - :math:`\lambda` is the rate parameter

    Attributes:
        l:
            Floating point value representing the rate parameter (:math:`\lambda`), expected value and variance of the Poisson distribution (must be greater than or equal to 0).
    """

    def __init__(self, scope: Scope, l: float = 1.0) -> None:
        r"""Initializes ``Poisson`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            l:
                Floating point value representing the rate parameter (:math:`\lambda`), expected value and variance of the Poisson distribution (must be greater than or equal to 0).
                Defaults to 1.0.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'Poisson' should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'Poisson' should be empty, but was {scope.evidence}.")

        super().__init__(scope=scope)
        self.set_params(l)

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Poisson`` can represent a single univariate node with ``MetaType.Discrete`` or ``PoissonType`` domain.

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

        # leaf is a discrete Poisson distribution
        if not (
            domains[0] == FeatureTypes.Discrete
            or domains[0] == FeatureTypes.Poisson
            or isinstance(domains[0], FeatureTypes.Poisson)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "Poisson":
        """Creates an instance from a specified signature.

        Returns:
            ``Poisson`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'Poisson' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Discrete:
            l = 1.0
        elif domain == FeatureTypes.Poisson:
            # instantiate object
            l = domain().l
        elif isinstance(domain, FeatureTypes.Poisson):
            l = domain.l
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Poisson' that was not caught during acception checking."
            )

        return Poisson(feature_ctx.scope, l=l)

    @property
    def dist(self) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return poisson(mu=self.l)

    def set_params(self, l: float) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            l:
                Floating point value representing the rate parameter (:math:`\lambda`), expected value and variance of the Poisson distribution (must be greater than or equal to 0).
        """
        if not tl_isfinite(l):
            raise ValueError(f"Value of 'l' for 'Poisson' must be finite, but was: {l}")

        if l < 0:
            raise ValueError(f"Value of 'l' for 'Poisson' must be non-negative, but was: {l}")

        self.l = float(l)

    def get_trainable_params(self) -> Tuple[float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Floating point value representing the rate parameter, expected value and variance.
        """
        return (self.l,)

    def get_params(self) -> Tuple[float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Floating point value representing the rate parameter, expected value and variance.
        """
        return (self.l,)

    def check_support(self, data: tl.tensor, is_scope_data: bool = False) -> tl.tensor:
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

        if tl.ndim(scope_data) != 2 or tl.shape(scope_data)[1] != len(self.scopes_out[0].query):
            raise ValueError(
                f"Expected 'scope_data' to be of shape (n,{len(self.scopes_out[0].query)}), but was: {tl.shape(scope_data)}"
            )

        valid = tl.ones(tl.shape(scope_data), dtype=bool)

        # nan entries (regarded as valid)
        nan_mask = tl_isnan(scope_data)

        # check for infinite values
        valid[~nan_mask] &= ~tl_isinf(scope_data[~nan_mask])

        # check if all values are valid integers
        valid[valid & ~nan_mask] &= (scope_data[valid & ~nan_mask] % 1) == 0

        # check if values are in valid range
        valid[valid & ~nan_mask] &= scope_data[valid & ~nan_mask] >= 0

        return valid
