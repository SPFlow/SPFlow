from .parametric import ParametricLeaf
from .statistical_types import ParametricType
from .exceptions import InvalidParametersError
from typing import Tuple, Dict, List
import numpy as np
from scipy.stats import lognorm  # type: ignore
from scipy.stats._distn_infrastructure import rv_continuous  # type: ignore

from multipledispatch import dispatch  # type: ignore


class LogNormal(ParametricLeaf):
    """(Univariate) Log-Normal distribution.

    PDF(x) =
        1/(x*sigma*sqrt(2*pi) * exp(-(ln(x)-mu)^2/(2*sigma^2)), where
            - x is an observation
            - mu is the mean
            - sigma is the standard deviation

    Attributes:
        mean:
            mean (mu) of the distribution.
        stdev:
            standard deviation (sigma) of the distribution (must be greater than 0).
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
