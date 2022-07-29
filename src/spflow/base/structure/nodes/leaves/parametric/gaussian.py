"""
Created on November 6, 2021

@authors: Bennet Wittelsbach, Philipp Deibert
"""

from .parametric import ParametricLeaf
from .statistical_types import ParametricType
from .exceptions import InvalidParametersError
from typing import Tuple, Dict, List, Optional
import numpy as np
from scipy.stats import norm  # type: ignore
from scipy.stats._distn_infrastructure import rv_continuous  # type: ignore
from multipledispatch import dispatch  # type: ignore


class Gaussian(ParametricLeaf):
    r"""(Univariate) Normal distribution.

    .. math::

        \text{PDF}(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})

    where
        - :math:`x` the observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

    Args:
        scope:
            List of integers specifying the variable scope.
        mean:
            mean (:math:`\mu`) of the distribution (default 0.0).
        stdev:
            standard deviation (:math:`\sigma`) of the distribution (must be greater than 0; default 0.0).
    """

    type = ParametricType.CONTINUOUS

    def __init__(
        self,
        scope: List[int],
        mean: float = None,
        stdev: float = None,
    ) -> None:
        if len(scope) != 1:
            raise ValueError(f"Scope size for Gaussian should be 1, but was: {len(scope)}")
        super().__init__(scope)
        self.mean = mean if mean is not None else np.random.uniform(-1, 1)
        self.stdev = stdev if stdev is not None else np.random.uniform(0, 1)
        self.set_params(self.mean, self.stdev)

    def set_params(self, mean: Optional[float]=0.0, stdev: Optional[float]=1.0) -> None:

        if not (np.isfinite(mean) and np.isfinite(stdev)):
            raise ValueError(
                f"Mean and standard deviation for Gaussian distribution must be finite, but were: {mean}, {stdev}"
            )
        if stdev <= 0.0:
            raise ValueError(
                f"Standard deviation for Gaussian distribution must be greater than 0.0, but was: {stdev}"
            )

        self.mean = mean
        self.stdev = stdev

    def get_params(self) -> Tuple[float, float]:
        return self.mean, self.stdev

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if instances are part of the support of the Gaussian distribution.

        .. math::

            \text{supp}(\text{Gaussian})=(-\infty,+\infty)

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

        return valid


@dispatch(Gaussian)  # type: ignore[no-redef]
def get_scipy_object(node: Gaussian) -> rv_continuous:
    return norm


@dispatch(Gaussian)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Gaussian) -> Dict[str, float]:
    if node.mean is None:
        raise InvalidParametersError(f"Parameter 'mean' of {node} must not be None")
    if node.stdev is None:
        raise InvalidParametersError(f"Parameter 'stdev' of {node} must not be None")
    parameters = {"loc": node.mean, "scale": node.stdev}
    return parameters
