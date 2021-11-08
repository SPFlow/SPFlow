"""
Created on November 06, 2021

@authors: Philipp Deibert
"""

import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter
from typing import List, Tuple, Union
from .parametric import TorchParametricLeaf, proj_bounded_to_real, proj_real_to_bounded
from spflow.base.structure.nodes.leaves.parametric.statistical_types import ParametricType
from spflow.base.structure.nodes.leaves.parametric import MultivariateGaussian

from multipledispatch import dispatch  # type: ignore


class TorchMultivariateGaussian(TorchParametricLeaf):
    """Multivariate Normal distribution.
    PDF(x) =
        1/sqrt((2*pi)^d * det(cov)) * exp(-1/2 (x-mu)^T * cov^(-1) * (x-mu)), where
            - d is the dimension of the distribution
            - x is the d-dim. vector of observations
            - mu is the d-dim. mean_vector
            - cov is the dxd covariance_matrix
    Attributes:
        mean_vector:
            A list, NumPy array or a PyTorch tensor holding the means (mu) of each of the one-dimensional Normal distributions.
            Has exactly as many elements as the scope of this leaf.
        covariance_matrix:
            A list of lists, NumPy array or PyTorch tensor (representing a two-dimensional NxN matrix, where N is the length
            of the scope) describing the covariances of the distribution. The diagonal holds
            the variances (sigma^2) of each of the one-dimensional distributions.
    """

    ptype = ParametricType.POSITIVE

    def __init__(
        self,
        scope: List[int],
        mean_vector: Union[List[float], torch.Tensor, np.ndarray],
        covariance_matrix: Union[List[List[float]], torch.Tensor, np.ndarray],
    ) -> None:

        super(TorchMultivariateGaussian, self).__init__(scope)

        # dimensions
        self.d = len(scope)

        # register mean vector as torch parameters
        self.mean_vector = Parameter()

        # internally we use the lower triangular matrix (Cholesky decomposition) to encode the covariance matrix
        # register (auxiliary) values for diagonal and non-diagonal values of lower triangular matrix as torch parameters
        self.tril_diag_aux = Parameter()
        self.tril_nondiag = Parameter()

        # pre-compute and store indices of non-diagonal values for lower triangular matrix
        self.tril_nondiag_indices = torch.tril_indices(self.d, self.d, offset=-1)

        # set parameters
        self.set_params(mean_vector, covariance_matrix)

    @property
    def covariance_tril(self) -> torch.Tensor:
        # create zero matrix of appropriate dimension
        L_nondiag = torch.zeros(self.d, self.d)
        # fill non-diagonal values of lower triangular matrix
        L_nondiag[self.tril_nondiag_indices[0], self.tril_nondiag_indices[1]] = self.tril_nondiag  # type: ignore
        # add (projected) diagonal values
        L = L_nondiag + proj_real_to_bounded(self.tril_diag_aux, lb=0.0) * torch.eye(self.d)  # type: ignore
        # return lower triangular matrix
        return L

    @property
    def covariance_matrix(self) -> torch.Tensor:
        # get lower triangular matrix
        L = self.covariance_tril
        # return covariance matrix
        return torch.matmul(L, L.T)

    @property
    def dist(self) -> D.Distribution:
        return D.MultivariateNormal(loc=self.mean_vector, scale_tril=self.covariance_tril)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        batch_size: int = data.shape[0]

        # get information relevant for the scope
        scope_data = data[:, list(self.scope)]

        # initialize empty tensor (number of output values matches batch_size)
        log_prob: torch.Tensor = torch.empty(batch_size, 1)

        # ----- marginalization -----

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.isnan(scope_data).sum(dim=1) == len(self.scope)] = 0.0

        # ----- log probabilities -----

        # create Torch distribution with specified parameters
        dist = D.MultivariateNormal(loc=self.mean_vector, scale_tril=self.covariance_tril)

        # compute probabilities on data samples where we have all values
        prob_mask = torch.isnan(scope_data).sum(dim=1) == 0
        log_prob[prob_mask] = dist.log_prob(scope_data[prob_mask]).unsqueeze(-1)

        return log_prob

    def set_params(
        self,
        mean_vector: Union[List[float], torch.Tensor, np.ndarray],
        covariance_matrix: Union[List[List[float]], torch.Tensor, np.ndarray],
    ) -> None:

        if isinstance(mean_vector, list):
            # convert float list to torch tensor
            mean_vector = torch.tensor([float(v) for v in mean_vector])
        elif isinstance(mean_vector, np.ndarray):
            # convert numpy array to torch tensor
            mean_vector = torch.from_numpy(mean_vector).type(torch.get_default_dtype())

        if isinstance(covariance_matrix, list):
            # convert numpy array to torch tensor
            covariance_matrix = torch.tensor([[float(v) for v in row] for row in covariance_matrix])
        elif isinstance(covariance_matrix, np.ndarray):
            # convert numpy array to torch tensor
            covariance_matrix = torch.from_numpy(covariance_matrix).type(torch.get_default_dtype())

        # dimensions
        d = mean_vector.numel()

        # make sure that number of dimensions matches scope length
        if d != self.d:
            raise ValueError(
                f"Mean vector length {mean_vector.numel()} does not match scope length {self.d}"
            )

        # make sure that dimensions of covariance matrix are correct
        if len(covariance_matrix.shape) != 2 or any(
            shape != d for shape in covariance_matrix.shape
        ):
            raise ValueError(
                f"Covariance matrix has shape {covariance_matrix.shape}, but should be of shape ({d},{d})"
            )

        # set mean vector
        self.mean_vector.data = mean_vector

        # compute lower triangular matrix (also check if covariance matrix is symmetric positive definite)
        L = torch.linalg.cholesky(covariance_matrix)  # type: ignore

        # set diagonal and non-diagonal values of lower triangular matrix
        self.tril_diag_aux.data = proj_bounded_to_real(torch.diag(L), lb=0.0)
        self.tril_nondiag.data = L[self.tril_nondiag_indices[0], self.tril_nondiag_indices[1]]

    def get_params(self) -> Tuple[List[float], List[List[float]]]:
        return self.mean_vector.data.cpu().tolist(), self.covariance_matrix.data.cpu().tolist()  # type: ignore


@dispatch(MultivariateGaussian)  # type: ignore[no-redef]
def toTorch(node: MultivariateGaussian) -> TorchMultivariateGaussian:
    return TorchMultivariateGaussian(node.scope, *node.get_params())


@dispatch(TorchMultivariateGaussian)  # type: ignore[no-redef]
def toNodes(torch_node: TorchMultivariateGaussian) -> MultivariateGaussian:
    return MultivariateGaussian(torch_node.scope, *torch_node.get_params())
