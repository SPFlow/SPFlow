"""Contains Gamma leaf node for SPFlow in the ``base`` backend.
"""
from typing import List, Tuple

import numpy as np
from scipy.stats import gamma  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.base.structure.general.nodes.leaf_node import LeafNode
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes, MetaType
from spflow.meta.data.scope import Scope


class Gamma(LeafNode):
    r"""(Univariate) Gamma distribution leaf node in the ``base`` backend.

    Represents an univariate Exponential distribution, with the following probability distribution function (PDF):

    .. math::
    
        \text{PDF}(x) = \begin{cases} \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x} & \text{if } x > 0\\
                                      0 & \text{if } x <= 0\end{cases}

    where
        - :math:`x` is the input observation
        - :math:`\Gamma` is the Gamma function
        - :math:`\alpha` is the shape parameter
        - :math:`\beta` is the rate parameter

    Attributes:
        alpha:
            Floating point value representing the shape parameter (:math:`\alpha`), greater than 0.
        beta:
            Floating point value representing the rate parameter (:math:`\beta`), greater than 0.
    """

    def __init__(
        self,
        scope: Scope,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        r"""Initializes ``Exponential`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            alpha:
                Floating point value representing the shape parameter (:math:`\alpha`), greater than 0.
                Defaults to 1.0.
            beta:
                Floating point value representing the rate parameter (:math:`\beta`), greater than 0.
                Defaults to 1.0.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'Gamma' should be 1, but was {len(scope.query)}.")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'Gamma' should be empty, but was {scope.evidence}.")

        super().__init__(scope=scope)
        self.set_params(alpha, beta)

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Gamma`` can represent a single univariate node with ``MetaType.Continuous`` or ``GammaType`` domain.

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

        # leaf is a continuous Gamma distribution
        if not (
            domains[0] == FeatureTypes.Continuous
            or domains[0] == FeatureTypes.Gamma
            or isinstance(domains[0], FeatureTypes.Gamma)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "Gamma":
        """Creates an instance from a specified signature.

        Returns:
            ``Gamma`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'Gamma' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Continuous:
            alpha, beta = 1.0, 1.0
        elif domain == FeatureTypes.Gamma:
            # instantiate object
            domain = domain()
            alpha, beta = domain.alpha, domain.beta
        elif isinstance(domain, FeatureTypes.Gamma):
            alpha, beta = domain.alpha, domain.beta
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Gamma' that was not caught during acception checking."
            )

        return Gamma(feature_ctx.scope, alpha=alpha, beta=beta)

    @property
    def dist(self) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return gamma(a=self.alpha, scale=1.0 / self.beta)

    def set_params(self, alpha: float, beta: float) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            alpha:
                Floating point value representing the shape parameter (:math:`\alpha`), greater than 0.
            beta:
                Floating point value representing the rate parameter (:math:`\beta`), greater than 0.
        """
        if alpha <= 0.0 or not np.isfinite(alpha):
            raise ValueError(f"Value of alpha for 'Gamma' must be greater than 0, but was: {alpha}")
        if beta <= 0.0 or not np.isfinite(beta):
            raise ValueError(f"Value of beta for 'Gamma' must be greater than 0, but was: {beta}")

        self.alpha = alpha
        self.beta = beta

    def get_params(self) -> Tuple[float, float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of the floating points representing the shape and rate parameters.
        """
        return self.alpha, self.beta

    def check_support(self, data: np.ndarray, is_scope_data: bool = False) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Gamma distribution, which is:

        .. math::

            \text{supp}(\text{Gamma})=(0,+\infty)

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
        valid[valid & ~nan_mask] &= scope_data[valid & ~nan_mask] > 0

        return valid
