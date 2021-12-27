"""
Created on November 6, 2021

@authors: Bennet Wittelsbach, Philipp Deibert
"""

from .parametric import ParametricLeaf
from .statistical_types import ParametricType
from .exceptions import InvalidParametersError
from typing import Tuple, Dict, List
import numpy as np
from scipy.stats import lognorm  # type: ignore
from scipy.stats._distn_infrastructure import rv_continuous  # type: ignore

from multipledispatch import dispatch  # type: ignore


class LogNormal(ParametricLeaf):
    r"""(Univariate) Log-Normal distribution.

    .. math::

        \text{PDF}(x) = \frac{1}{x\sigma\sqrt{2\pi}}\exp\left(-\frac{(\ln(x)-\mu)^2}{2\sigma^2}\right)

    where
        - :math:`x` is an observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

    Args:
        scope:
            List of integers specifying the variable scope.
        mean:
            mean (:math:`\mu`) of the distribution.
        stdev:
            standard deviation (:math:`\sigma`) of the distribution (must be greater than 0).
    """

    type = ParametricType.POSITIVE

    def __init__(self, scope: List[int], mean: float, stdev: float) -> None:

        if len(scope) != 1:
            raise ValueError(f"Scope size for LogNormal should be 1, but was: {len(scope)}")

        super().__init__(scope)
        self.set_params(mean, stdev)

    def set_params(self, mean: float, stdev: float) -> None:

        if not (np.isfinite(mean) and np.isfinite(stdev)):
            raise ValueError(
                f"Mean and standard deviation for LogNormal distribution must be finite, but were: {mean}, {stdev}"
            )
        if stdev <= 0.0:
            raise ValueError(
                f"Standard deviation for LogNormal distribution must be greater than 0.0, but was: {stdev}"
            )

        self.mean = mean
        self.stdev = stdev

    def get_params(self) -> Tuple[float, float]:
        return self.mean, self.stdev

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if instances are part of the support of the LogNormal distribution.

        .. math::

            \text{supp}(\text{LogNormal})=(0,\infty)

        Args:
            scope_data:
                Torch tensor containing possible distribution instances.
        Returns:
            Torch tensor indicating for each possible distribution instance, whether they are part of the support (True) or not (False).
        """

        valid = np.ones(scope_data.shape, dtype=bool)

        # check for infinite values
        valid &= ~np.isinf(scope_data)

        # check if values are in valid range
        valid[valid] &= scope_data[valid] > 0

        return valid


@dispatch(LogNormal)  # type: ignore[no-redef]
def get_scipy_object(node: LogNormal) -> rv_continuous:
    return lognorm


@dispatch(LogNormal)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: LogNormal) -> Dict[str, float]:
    if node.mean is None:
        raise InvalidParametersError(f"Parameter 'mean' of {node} must not be None")
    if node.stdev is None:
        raise InvalidParametersError(f"Parameter 'stdev' of {node} must not be None")

    parameters = {"loc": 0.0, "scale": np.exp(node.mean), "s": node.stdev}
    return parameters
