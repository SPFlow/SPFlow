"""
Created on November 6, 2021

@authors: Bennet Wittelsbach, Philipp Deibert
"""

from .parametric import ParametricLeaf
from .statistical_types import ParametricType
from .exceptions import InvalidParametersError
from typing import Tuple, Dict, List
import numpy as np
from scipy.stats import bernoulli  # type: ignore
from scipy.stats._distn_infrastructure import rv_discrete  # type: ignore

from multipledispatch import dispatch  # type: ignore


class Bernoulli(ParametricLeaf):
    r"""(Univariate) Bernoulli distribution.

    .. math::

        \text{PMF}(k)=\begin{cases} p   & \text{if } k=1\\
                                    1-p & \text{if } k=0\end{cases}
        
    where
        - :math:`p` is the success probability
        - :math:`k` is the outcome of the trial (0 or 1)

    Args:
        scope:
            List of integers specifying the variable scope.
        p:
            Probability of success in the range :math:`[0,1]`.
    """

    type = ParametricType.BINARY

    def __init__(self, scope: List[int], p: float) -> None:

        if len(scope) != 1:
            raise ValueError(f"Scope size for Bernoulli should be 1, but was: {len(scope)}")

        super().__init__(scope)
        self.set_params(p)

    def set_params(self, p: float) -> None:

        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for Bernoulli distribution must to be between 0.0 and 1.0, but was: {p}"
            )

        self.p = p

    def get_params(self) -> Tuple[float]:
        return (self.p,)

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if instances are part of the support of the Bernoulli distribution.

        .. math::

            \text{supp}(\text{Bernoulli})=\{0,1\}

        Args:
            scope_data:
                Torch tensor containing possible distribution instances.
        Returns:
            Torch tensor indicating for each possible distribution instance, whether they are part of the support (True) or not (False).
        """

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scope):
            raise ValueError(
                f"Expected scope_data to be of shape (n,{len(self.scope)}), but was: {scope_data.shape}"
            )

        valid = np.ones(scope_data.shape[0], dtype=bool)

        # check for infinite values
        valid &= ~np.isinf(scope_data).sum(axis=-1).astype(bool)

        # check if all values are valid integers
        # TODO: runtime warning due to nan values
        valid[valid] &= (np.remainder(scope_data[valid], 1) == 0).sum(axis=-1).astype(bool)

        # check if values are in valid range
        valid[valid] &= (
            ((scope_data[valid] >= 0) & (scope_data[valid] <= 1)).sum(axis=-1).astype(bool)
        )

        return valid


@dispatch(Bernoulli)  # type: ignore[no-redef]
def get_scipy_object(node: Bernoulli) -> rv_discrete:
    return bernoulli


@dispatch(Bernoulli)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Bernoulli) -> Dict[str, float]:
    if node.p is None:
        raise InvalidParametersError(f"Parameter 'p' of {node} must not be None")
    parameters = {"p": node.p}
    return parameters
