from .parametric import ParametricLeaf
from .statistical_types import ParametricType
from .exceptions import InvalidParametersError
from typing import Tuple, Dict, List
import numpy as np
from scipy.stats import geom  # type: ignore
from scipy.stats._distn_infrastructure import rv_discrete  # type: ignore

from multipledispatch import dispatch  # type: ignore


class Geometric(ParametricLeaf):
    """(Univariate) Geometric distribution.

    PMF(k) =
        p * (1-p)^(k-1)

    Attributes:
        p:
            Probability of success in the range (0,1].
    """

    type = ParametricType.BINARY

    def __init__(self, scope: List[int], p: float) -> None:

        if(len(scope) != 1):
            raise ValueError(f"Scope size for Geometric should be 1, but was: {len(scope)}")

        super().__init__(scope)
        self.set_params(p)

    def set_params(self, p: float) -> None:

        if p <= 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of p for Geometric distribution must to be greater than 0.0 and less or equal to 1.0, but was: {p}"
            )

        self.p = p

    def get_params(self) -> Tuple[float]:
        return (self.p,)


@dispatch(Geometric)  # type: ignore[no-redef]
def get_scipy_object(node: Geometric) -> rv_discrete:
    return geom


@dispatch(Geometric)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Geometric) -> Dict[str, float]:
    if node.p is None:
        raise InvalidParametersError(f"Parameter 'p' of {node} must not be None")
    parameters = {"p": node.p}
    return parameters
