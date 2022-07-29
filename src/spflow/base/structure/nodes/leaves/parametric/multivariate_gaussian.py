"""
Created on November 6, 2021

@authors: Bennet Wittelsbach, Philipp Deibert
"""

from .parametric import ParametricLeaf
from .statistical_types import ParametricType
from .exceptions import InvalidParametersError
from typing import Tuple, Dict, List, Union, Optional
import numpy as np
from scipy.stats import multivariate_normal  # type: ignore
from scipy.stats._distn_infrastructure import rv_continuous  # type: ignore

from multipledispatch import dispatch  # type: ignore


class MultivariateGaussian(ParametricLeaf):
    r"""Multivariate Normal distribution.

    .. math::

        \text{PDF}(x) = \frac{1}{\sqrt{(2\pi)^d\det\Sigma}}\exp\left(-\frac{1}{2} (x-\mu)^T\Sigma^{-1}(x-\mu)\right)

    where
        - :math:`d` is the dimension of the distribution
        - :math:`x` is the :math:`d`-dim. vector of observations
        - :math:`\mu` is the :math:`d`-dim. mean vector
        - :math:`\Sigma` is the :math:`d\times d` covariance matrix

    Args:
        scope:
            List of integers specifying the variable scope.
        mean_vector:
            A list, NumPy array or a PyTorch tensor holding the means (:math:`\mu`) of each of the one-dimensional Normal distributions (defaults to all zeros).
            Has exactly as many elements as the scope of this leaf.
        covariance_matrix:
            A list of lists, NumPy array or PyTorch tensor (representing a two-dimensional :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
            of the scope) describing the covariances of the distribution (defaults to the identity matrix). The diagonal holds
            the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
    """

    type = ParametricType.CONTINUOUS

    def __init__(
        self,
        scope: List[int],
        mean_vector: Optional[Union[List[float], np.ndarray]]=None,
        covariance_matrix: Optional[Union[List[List[float]], np.ndarray]]=None,
    ) -> None:

        # check if scope contains duplicates
        if(len(set(scope)) != len(scope)):
            raise ValueError("Scope for MultivariateGaussian contains duplicate variables.")

        super().__init__(scope)

        if(mean_vector is None):
            mean_vector = np.zeros((1,len(scope)))
        if(covariance_matrix is None):
            covariance_matrix = np.eye(len(scope))

        self.set_params(mean_vector, covariance_matrix)

    def set_params(
        self,
        mean_vector: Union[List[float], np.ndarray],
        covariance_matrix: Union[List[List[float]], np.ndarray],
    ) -> None:

        # cast lists to numpy arrays
        if isinstance(mean_vector, List):
            mean_vector = np.array(mean_vector)
        if isinstance(covariance_matrix, List):
            covariance_matrix = np.array(covariance_matrix)

        # check mean vector dimensions
        if (
            (mean_vector.ndim == 1 and mean_vector.shape[0] != len(self.scope))
            or (mean_vector.ndim == 2 and mean_vector.shape[1] != len(self.scope))
            or mean_vector.ndim > 2
        ):
            raise ValueError(
                f"Dimensions of mean vector for MultivariateGaussian should match scope size {len(self.scope)}, but was: {mean_vector.shape}"
            )

        # check mean vector for nan or inf values
        if np.any(np.isinf(mean_vector)):
            raise ValueError("Mean vector for MultivariateGaussian may not contain infinite values")
        if np.any(np.isnan(mean_vector)):
            raise ValueError("Mean vector for MultivariateGaussian may not contain NaN values")

        # test whether or not matrix has correct shape
        if covariance_matrix.ndim != 2 or (
            covariance_matrix.ndim == 2
            and (
                covariance_matrix.shape[0] != len(self.scope)
                or covariance_matrix.shape[1] != len(self.scope)
            )
        ):
            raise ValueError(
                f"Covariance matrix for MultivariateGaussian expected to be of shape ({len(self.scope), len(self.scope)}), but was: {covariance_matrix.shape}"
            )

        # check covariance matrix for nan or inf values
        if np.any(np.isinf(covariance_matrix)):
            raise ValueError(
                "Covariance matrix for MultivariateGaussian may not contain infinite values"
            )
        if np.any(np.isnan(covariance_matrix)):
            raise ValueError(
                "Covariance matrix for MultivariateGaussian may not contain NaN values"
            )

        # test covariance matrix for symmetry
        if not np.allclose(covariance_matrix, covariance_matrix.T):
            raise ValueError("Covariance matrix for MultivariateGaussian must be symmetric")
        # test covariance matrix for positive semi-definiteness
        if np.any(np.linalg.eigvals(covariance_matrix) < 0):
            raise ValueError(
                "Covariance matrix for MultivariateGaussian must be positive semi-definite"
            )

        self.mean_vector = mean_vector
        self.covariance_matrix = covariance_matrix

    def get_params(
        self,
    ) -> Tuple[Union[List[float], np.ndarray], Union[List[List[float]], np.ndarray]]:
        return self.mean_vector, self.covariance_matrix

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if instances are part of the support of the MultivariateGaussian distribution.

        .. math::

            \text{supp}(\text{MultivariateGaussian})=(-\infty,+\infty)^k

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
        # additionally check for infinite values (may return NaNs despite support)
        valid &= ~np.isinf(scope_data).sum(axis=-1).astype(bool)

        return valid


@dispatch(MultivariateGaussian)  # type: ignore[no-redef]
def get_scipy_object(node: MultivariateGaussian) -> rv_continuous:
    return multivariate_normal


@dispatch(MultivariateGaussian)  # type: ignore[no-redef]
def get_scipy_object_parameters(
    node: MultivariateGaussian,
) -> Dict[str, np.ndarray]:
    if node.mean_vector is None:
        raise InvalidParametersError(f"Parameter 'mean_vector' of {node} must not be None")
    if node.covariance_matrix is None:
        raise InvalidParametersError(f"Parameter 'covariance_matrix' of {node} must not be None")
    parameters = {"mean": node.mean_vector, "cov": node.covariance_matrix}
    return parameters
