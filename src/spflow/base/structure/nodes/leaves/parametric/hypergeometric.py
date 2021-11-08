from .parametric import ParametricLeaf
from .statistical_types import ParametricType
from .exceptions import InvalidParametersError
from typing import Tuple, Dict, List
import numpy as np
from scipy.stats import hypergeom  # type: ignore
from scipy.stats._distn_infrastructure import rv_discrete  # type: ignore

from multipledispatch import dispatch  # type: ignore


class Hypergeometric(ParametricLeaf):
    """(Univariate) Hypergeometric distribution.

    PMF(k) =
        (M)C(k) * (N-M)C(n-k) / (N)C(n), where
            - (n)C(k) is the binomial coefficient (n choose k)

    Attributes:
        N:
            Total number of entities (in the population), greater or equal to 0.
        M:
            Number of entities with property of interest (in the population), greater or equal to zero and less than or equal to N.
        n:
            Number of observed entities (sample size), greater or equal to zero and less than or equal to N.
    """

    type = ParametricType.COUNT

    def __init__(self, scope: List[int], N: int, M: int, n: int) -> None:

        if(len(scope) != 1):
            raise ValueError(f"Scope size for Hypergeometric should be 1, but was: {len(scope)}")

        super().__init__(scope)
        self.set_params(N, M, n)

    def set_params(self, N: int, M: int, n: int) -> None:

        if N < 0 or not np.isfinite(N):
            raise ValueError(
                f"Value of N for Hypergeometric distribution must be greater of equal to 0, but was: {N}"
            )
        if M < 0 or M > N or not np.isfinite(M):
            raise ValueError(
                f"Value of M for Hypergeometric distribution must be greater of equal to 0 and less or equal to N, but was: {M}"
            )
        if n < 0 or n > N or not np.isfinite(n):
            raise ValueError(
                f"Value of n for Hypergeometric distribution must be greater of equal to 0 and less or equal to N, but was: {n}"
            )

        self.N = N
        self.M = M
        self.n = n

    def get_params(self) -> Tuple[int, int, int]:
        return self.N, self.M, self.n


@dispatch(Hypergeometric)  # type: ignore[no-redef]
def get_scipy_object(node: Hypergeometric) -> rv_discrete:
    return hypergeom


@dispatch(Hypergeometric)  # type: ignore[no-redef]
def get_scipy_object_parameters(node: Hypergeometric) -> Dict[str, int]:
    if node.N is None:
        raise InvalidParametersError(f"Parameter 'N' of {node} must not be None")
    if node.M is None:
        raise InvalidParametersError(f"Parameter 'M' of {node} must not be None")
    if node.n is None:
        raise InvalidParametersError(f"Parameter 'n' of {node} must not be None")
    # note: scipy hypergeom has switched semantics for the parameters
    parameters = {"M": node.N, "n": node.M, "N": node.n}
    return parameters
