"""Contains Gaussian leaf node for SPFlow in the ``base`` backend.
"""
from typing import List, Tuple

import numpy as np
from scipy.stats import norm  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.base.structure.general.nodes.leaf_node import LeafNode
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes, MetaType
from spflow.meta.data.scope import Scope


class Gaussian(LeafNode):
    r"""(Univariate) Gaussian (a.k.a. Normal) distribution leaf node in the ``base`` backend.

    Represents an univariate Gaussian distribution, with the following probability density function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})

    where
        - :math:`x` the observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

    Attributes:
        mean:
            Floating point value representing the mean (:math:`\mu`) of the distribution.
        std:
            Floating point values representing the standard deviation (:math:`\sigma`) of the distribution (must be greater than 0).
    """

    def __init__(
        self,
        scope: Scope,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> None:
        r"""Initializes ``Gaussian`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            mean:
                Floating point value representing the mean (:math:`\mu`) of the distribution.
                Defaults to 0.0.
            std:
                Floating point values representing the standard deviation (:math:`\sigma`) of the distribution (must be greater than 0).
                Defaults to 1.0.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'Gaussian' should be 1, but was: {len(scope.query)}."
            )
        if len(scope.evidence) != 0:
            raise ValueError(
                f"Evidence scope for 'Gaussian' should be empty, but was {scope.evidence}."
            )

        super().__init__(scope=scope)
        self.set_params(mean, std)

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Gaussian`` can represent a single univariate node with ``MetaType.Continuous`` or ``GaussianType`` domain.

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

        # leaf is a continuous Gaussian distribution
        if not (
            domains[0] == FeatureTypes.Continuous
            or domains[0] == FeatureTypes.Gaussian
            or isinstance(domains[0], FeatureTypes.Gaussian)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "Gaussian":
        """Creates an instance from a specified signature.

        Returns:
            ``Gaussian`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'Gaussian' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Continuous:
            mean, std = 0.0, 1.0
        elif domain == FeatureTypes.Gaussian:
            # instantiate object
            domain = domain()
            mean, std = domain.mean, domain.std
        elif isinstance(domain, FeatureTypes.Gaussian):
            mean, std = domain.mean, domain.std
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Gaussian' that was not caught during acception checking."
            )

        return Gaussian(feature_ctx.scope, mean=mean, std=std)

    @property
    def dist(self) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return norm(loc=self.mean, scale=self.std)

    def set_params(self, mean: float, std: float) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            mean:
                Floating point value representing the mean (:math:`\mu`) of the distribution.
            std:
                Floating point values representing the standard deviation (:math:`\sigma`) of the distribution (must be greater than 0).
        """
        if not (np.isfinite(mean) and np.isfinite(std)):
            raise ValueError(
                f"Values for 'mean' and 'std' for 'Gaussian' must be finite, but were: {mean}, {std}."
            )
        if std <= 0.0:
            raise ValueError(
                f"Value for 'std' for 'Gaussian' must be greater than 0.0, but was: {std}."
            )

        self.mean = mean
        self.std = std

    def get_params(self) -> Tuple[float, float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of floating point values representing the mean and standard deviation.
        """
        return self.mean, self.std

    def check_support(
        self, data: np.ndarray, is_scope_data: bool = False
    ) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Gaussian distribution, which is:

        .. math::

            \text{supp}(\text{Gaussian})=(-\infty,+\infty)

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

        return valid
