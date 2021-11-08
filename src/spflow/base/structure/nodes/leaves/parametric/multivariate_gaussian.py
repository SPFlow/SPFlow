from .parametric import ParametricLeaf
from .statistical_types import ParametricType
from .exceptions import InvalidParametersError
from typing import Tuple, Dict, List, Union
import numpy as np
from scipy.stats import multivariate_normal  # type: ignore
from scipy.stats._distn_infrastructure import rv_continuous  # type: ignore

from multipledispatch import dispatch  # type: ignore


class MultivariateGaussian(ParametricLeaf):
    """Multivariate Normal distribution.

    PDF(x) =
        1/sqrt((2*pi)^d * det(cov)) * exp(-1/2 (x-mu)^T * cov^(-1) * (x-mu)), where
            - d is the dimension of the distribution
            - x is the d-dim. vector of observations
            - mu is the d-dim. mean_vector
            - cov is the dxd covariance_matrix

    Attributes:
        mean_vector:
            A list or NumPy array holding the means (mu) of each of the one-dimensional Normal distributions.
            Has exactly as many elements as the scope of this leaf.
        covariance_matrix:
            A list of lists or NumPy array (representing a two-dimensional NxN matrix, where N is the length
            of the scope) describing the covariances of the distribution. The diagonal holds
            the variances (sigma^2) of each of the one-dimensional distributions.
    """

    type = ParametricType.CONTINUOUS

    def __init__(
        self,
        scope: List[int],
        mean_vector: Union[List[float], np.ndarray],
        covariance_matrix: Union[List[List[float]], np.ndarray],
    ) -> None:
        if len(scope) != len(mean_vector):
            raise ValueError(
                f"Size of mean vector for TorchMultivariateGaussian should match scope size {len(scope)}, but was: {len(mean_vector)}"
            )

        super().__init__(scope)
        self.set_params(mean_vector, covariance_matrix)

    def set_params(
        self,
        mean_vector: Union[List[float], np.ndarray],
        covariance_matrix: Union[List[List[float]], np.ndarray],
    ) -> None:
        self.mean_vector = mean_vector
        self.covariance_matrix = covariance_matrix

    def get_params(
        self,
    ) -> Tuple[Union[List[float], np.ndarray], Union[List[List[float]], np.ndarray]]:
        return self.mean_vector, self.covariance_matrix


@dispatch(MultivariateGaussian)  # type: ignore[no-redef]
def get_scipy_object(node: MultivariateGaussian) -> rv_continuous:
    return multivariate_normal


@dispatch(MultivariateGaussian)  # type: ignore[no-redef]
def get_scipy_object_parameters(
    node: MultivariateGaussian,
) -> Dict[str, Union[Union[List[float], np.ndarray], Union[List[List[float]], np.ndarray]]]:
    if node.mean_vector is None:
        raise InvalidParametersError(f"Parameter 'mean_vector' of {node} must not be None")
    if node.covariance_matrix is None:
        raise InvalidParametersError(f"Parameter 'covariance_matrix' of {node} must not be None")
    parameters = {"mean": node.mean_vector, "cov": node.covariance_matrix}
    return parameters
