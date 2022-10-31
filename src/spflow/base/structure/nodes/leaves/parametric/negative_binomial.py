# -*- coding: utf-8 -*-
"""Contains Negative Binomial leaf node for SPFlow in the ``base`` backend.
"""
from typing import Tuple, List, Union, Type
import numpy as np
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import MetaType, FeatureType, FeatureTypes
from spflow.base.structure.nodes.node import LeafNode

from scipy.stats import nbinom  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore


class NegativeBinomial(LeafNode):
    r"""(Univariate) Negative Binomial distribution leaf node in the 'base' backend.

    Represents an univariate Negative Binomial distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \binom{k+n-1}{n-1}p^n(1-p)^k

    where
        - :math:`k` is the number of failures
        - :math:`n` is the maximum number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Attributes:
        n:
            Integer representing the number of successes (greater or equal to 0).
        p:
            Floating point value representing the success probability of each trial in the range :math:`(0,1]`.
    """

    def __init__(self, scope: Scope, n: int, p: float = 0.5) -> None:
        r"""Initializes ``NegativeBinomial`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            n:
                Integer representing the number of successes (greater or equal to 0).
            p:
                Floating point value representing the success probability of each trial in the range :math:`(0,1]`.
                Defaults to 0.5.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'NegativeBinomial' should be 1, but was: {len(scope.query)}."
            )
        if len(scope.evidence) != 0:
            raise ValueError(
                f"Evidence scope for 'NegativeBinomial' should be empty, but was {scope.evidence}."
            )

        super(NegativeBinomial, self).__init__(scope=scope)
        self.set_params(n, p)

    @classmethod
    def accepts(self, signatures: List[Tuple[List[Union[MetaType, FeatureType, Type[FeatureType]]], Scope]]) -> bool:
        """TODO"""
        # leaf only has one output
        if len(signatures) != 1:
            return False

        # get single output signature
        types, scope = signatures[0]

        # leaf is a single non-conditional univariate node
        if len(types) != 1 or len(scope.query) != len(types) or len(scope.evidence) != 0:
            return False
        
        # leaf is a discrete Negative Binomial distribution
        # NOTE: only accept instances of 'FeatureTypes.NegativeBinomial', otherwise required parameter 'n' is not specified. Reject 'FeatureTypes.Discrete' for the same reason.
        if not isinstance(types[0], FeatureTypes.NegativeBinomial):
            return False

        return True

    @classmethod
    def from_signatures(self, signatures: List[Tuple[List[Union[MetaType, FeatureType, Type[FeatureType]]], Scope]]) -> "NegativeBinomial":
        """TODO"""
        if not self.accepts(signatures):
            raise ValueError(f"'NegativeBinomial' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        types, scope = signatures[0]
        type = types[0]

        # read or initialize parameters
        if isinstance(type, FeatureTypes.NegativeBinomial):
            n, p = type.n, type.p
        else:
            raise ValueError(f"Unknown signature type {type} for 'NegativeBinomial' that was not caught during acception checking.")

        return NegativeBinomial(scope, n=n, p=p)

    @property
    def dist(self) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return nbinom(n=self.n, p=self.p)

    def set_params(self, n: int, p: float) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            n:
               Integer representing the number of successes (greater or equal to 0).
            p:
                Floating point value representing the success probability of each trial in the range :math:`(0,1]`.
        """
        if p <= 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for Negative Binomial distribution must to be between 0.0 (excluding) and 1.0 (including), but was: {p}"
            )
        if n < 0 or not np.isfinite(n):
            raise ValueError(
                f"Value of n for Negative Binomial distribution must to greater of equal to 0, but was: {n}"
            )

        if not (np.remainder(n, 1.0) == 0.0):
            raise ValueError(
                f"Value of n for Negative Binomial distribution must be (equal to) an integer value, but was: {n}"
            )

        self.n = n
        self.p = p

    def get_params(self) -> Tuple[int, float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of the number of successes and the floating point value representing the success probability.
        """
        return self.n, self.p

    def check_support(
        self, data: np.ndarray, is_scope_data: bool = False
    ) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Negative Binomial distribution, which is:

        .. math::

            \text{supp}(\text{NegativeBinomial})=\mathbb{N}\cup\{0\}

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

        # check if all values are valid integers
        valid[valid & ~nan_mask] &= (
            np.remainder(scope_data[valid & ~nan_mask], 1) == 0
        )

        # check if values are in valid range
        valid[valid & ~nan_mask] &= scope_data[valid & ~nan_mask] >= 0

        return valid
