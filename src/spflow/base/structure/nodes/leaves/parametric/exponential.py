# -*- coding: utf-8 -*-
"""Contains Exponential leaf node for SPFlow in the ``base`` backend.
"""
from typing import Tuple, List, Union, Type
import numpy as np
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import MetaType, FeatureType, FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import expon  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore


class Exponential(LeafNode):
    r"""(Univariate) Exponential distribution leaf node in the ``base`` backend.

    Represents an univariate Exponential distribution, with the following probability distribution function (PDF):

    .. math::
        
        \text{PDF}(x) = \begin{cases} \lambda e^{-\lambda x} & \text{if } x > 0\\
                                      0                      & \text{if } x <= 0\end{cases}
    
    where
        - :math:`x` is the input observation
        - :math:`\lambda` is the rate parameter
    
    Attributes:
        l:
            Floating point value representing the rate parameter (:math:`\lambda`) of the Exponential distribution (must be greater than 0; default 1.0).
    """

    def __init__(self, scope: Scope, l: float = 1.0) -> None:
        r"""Initializes ``Exponential`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            l:
                Floating point value representing the rate parameter (:math:`\lambda`) of the Exponential distribution (must be greater than 0).
                Defaults to 1.0.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'Exponential' should be 1, but was {len(scope.query)}."
            )
        if len(scope.evidence) != 0:
            raise ValueError(
                f"Evidence scope for 'Exponential' should be empty, but was {scope.evidence}."
            )

        super(Exponential, self).__init__(scope=scope)
        self.set_params(l)

    @classmethod
    def accepts(self, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Exponential`` can represent a single univariate node with ``MetaType.Continuous`` or ``ExponentialType`` domain.

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
    ) -> "Exponential":
        """Creates an instance from a specified signature.

        Returns:
            ``Exponential`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not self.accepts(signatures):
            raise ValueError(
                f"'Exponential' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Continuous:
            l = 1.0
        elif domain == FeatureTypes.Exponential:
            # instantiate object
            l = domain().l
        elif isinstance(domain, FeatureTypes.Exponential):
            l = domain.l
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Exponential' that was not caught during acception checking."
            )

        return Exponential(feature_ctx.scope, l=l)

    @property
    def dist(self) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return expon(scale=1.0 / self.l)

    def set_params(self, l: float) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            l:
                Floating point value representing the rate parameter (:math:`\lambda`) of the Exponential distribution (must be greater than 0).
        """
        if l <= 0.0 or not np.isfinite(l):
            raise ValueError(
                f"Value of 'l' for 'Exponential' must be greater than 0, but was: {l}"
            )

        self.l = l

    def get_params(self) -> Tuple[float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Floating point value representing the rate parameter.
        """
        return (self.l,)

    def check_support(
        self, data: np.ndarray, is_scope_data: bool = False
    ) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Exponential distribution, which is:

        .. math::

            \text{supp}(\text{Exponential})=[0,+\infty)

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            scope_data:
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
