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
            A list, NumPy array or a PyTorch tensor holding the means (:math:`\mu`) of each of the one-dimensional Normal distributions.
            Has exactly as many elements as the scope of this leaf.
        covariance_matrix:
            A list of lists, NumPy array or PyTorch tensor (representing a two-dimensional :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
            of the scope) describing the covariances of the distribution. The diagonal holds
            the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
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

        # check for partially marginalized instances
        n_marg = torch.isnan(scope_data).sum(dim=1)

        if torch.any((n_marg > 0) & (n_marg < len(self.scope))):
            raise ValueError(
                f"Partial marginalization not yet supported for TorchMultivariateGaussian."
            )

        marg_ids = n_marg == len(self.scope)

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[marg_ids] = 0.0

        # ----- log probabilities -----

        if len(scope_data[~marg_ids]) > 0:
            # create masked based on distribution's support
            valid_ids = self.check_support(scope_data[~marg_ids])

            if not all(valid_ids):
                raise ValueError(
                    f"Encountered data instances that are not in the support of the TorchMultivariateGaussian distribution."
                )

            # compute probabilities for values inside distribution support
            log_prob[~marg_ids, 0] = self.dist.log_prob(
                scope_data[~marg_ids].type(torch.get_default_dtype())
            )

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

        # check mean vector for nan or inf values
        if torch.any(torch.isinf(mean_vector)):
            raise ValueError(
                "Mean vector for TorchMultivariateGaussian may not contain infinite values"
            )
        if torch.any(torch.isnan(mean_vector)):
            raise ValueError("Mean vector for TorchMultivariateGaussian may not contain NaN values")

        # dimensions
        d = mean_vector.numel()

        # make sure that number of dimensions matches scope length
        if (
            (mean_vector.ndim == 1 and mean_vector.shape[0] != len(self.scope))
            or (mean_vector.ndim == 2 and mean_vector.shape[1] != len(self.scope))
            or mean_vector.ndim > 2
        ):
            raise ValueError(
                f"Dimensions of mean vector for TorchMultivariateGaussian should match scope size {len(self.scope)}, but was: {mean_vector.shape}"
            )

        # make sure that dimensions of covariance matrix are correct
        if covariance_matrix.ndim != 2 or (
            covariance_matrix.ndim == 2
            and (
                covariance_matrix.shape[0] != len(self.scope)
                or covariance_matrix.shape[1] != len(self.scope)
            )
        ):
            raise ValueError(
                f"Covariance matrix for TorchMultivariateGaussian expected to be of shape ({len(self.scope), len(self.scope)}), but was: {covariance_matrix.shape}"
            )

        # set mean vector
        self.mean_vector.data = mean_vector

        # check covariance matrix for nan or inf values
        if torch.any(torch.isinf(mean_vector)):
            raise ValueError(
                "Covariance matrix vector for TorchMultivariateGaussian may not contain infinite values"
            )
        if torch.any(torch.isnan(mean_vector)):
            raise ValueError(
                "Covariance matrix for TorchMultivariateGaussian may not contain NaN values"
            )

        # compute lower triangular matrix (also check if covariance matrix is symmetric positive definite)
        L = torch.linalg.cholesky(covariance_matrix)  # type: ignore

        # set diagonal and non-diagonal values of lower triangular matrix
        self.tril_diag_aux.data = proj_bounded_to_real(torch.diag(L), lb=0.0)
        self.tril_nondiag.data = L[self.tril_nondiag_indices[0], self.tril_nondiag_indices[1]]

    def get_params(self) -> Tuple[List[float], List[List[float]]]:
        return self.mean_vector.data.cpu().tolist(), self.covariance_matrix.data.cpu().tolist()  # type: ignore

    def check_support(self, scope_data: torch.Tensor) -> torch.Tensor:
        r"""Checks if instances are part of the support of the MultivariateGaussian distribution.

        .. math::

            \text{supp}(\text{MultivariateGaussian})=(-\infty,+\infty)^k

        Args:
            scope_data:
                Torch tensor containing possible distribution instances.
        Returns:
            Torch tensor indicating for each possible distribution instance, whether they are part of the support (True) or not (False).
        """

        valid = self.dist.support.check(scope_data)  # type: ignore

        # additionally check for infinite values (may return NaNs despite support)
        mask = valid.clone()
        valid[mask] &= ~scope_data[mask].isinf().sum(dim=-1).bool()

        return valid


@dispatch(MultivariateGaussian)  # type: ignore[no-redef]
def toTorch(node: MultivariateGaussian) -> TorchMultivariateGaussian:
    return TorchMultivariateGaussian(node.scope, *node.get_params())


@dispatch(TorchMultivariateGaussian)  # type: ignore[no-redef]
def toNodes(torch_node: TorchMultivariateGaussian) -> MultivariateGaussian:
    return MultivariateGaussian(torch_node.scope, *torch_node.get_params())
