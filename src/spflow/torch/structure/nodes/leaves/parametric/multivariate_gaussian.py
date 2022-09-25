"""
Created on November 06, 2021

@authors: Philipp Deibert
"""
import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter
from typing import List, Tuple, Union, Optional, Iterable
from .projections import proj_bounded_to_real, proj_real_to_bounded
from spflow.meta.scope.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.nodes.node import LeafNode
from spflow.torch.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.structure.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian as BaseMultivariateGaussian
from packaging import version


def torch_spacing(A: torch.Tensor) -> torch.Tensor:
    return torch.min(
        torch.nextafter(A,  torch.tensor(float("inf")))-A, 
        torch.nextafter(A, -torch.tensor(float("inf")))-A
    )


def nearest_sym_pd(A: torch.Tensor) -> torch.Tensor:
    # compute closest positive definite matrix as described in (Higham, 1988) https://www.sciencedirect.com/science/article/pii/0024379588902236
    # based on MATLAB implementation found here: https://mathworks.com/matlabcentral/fileexchange/42885-nearestspd?s_tid=mwa_osa_a and this Python port: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194

    if version.parse(torch.__version__) < version.parse("1.11.0"):
        exception = RuntimeError
    else:
        exception = torch.linalg.LinAlgError

    def is_pd(A: torch.Tensor) -> torch.Tensor:
        try:
            torch.linalg.cholesky(A)
            return True
        except torch.linalg.LinAlgError:
            return False

    # make sure matrix is symmetric
    B = (A + A)/2

    # compute symmetric polar factor of B from SVD (which is symmetric positive definite)
    U, s, _ = torch.linalg.svd(B)
    H = torch.matmul(U, torch.matmul(torch.diag(s), U.T))
    
    # compute closest symmetric positive semi-definite matrix to A in Frobenius norm (see paper linked above)
    A_hat = (B+H)/2
    # again, make sure matrix is symmetric
    A_hat = (A_hat + A_hat.T)/2

    # check if matrix is actually symmetric positive-definite
    if is_pd(A_hat):
        return A_hat

    # else fix it
    spacing = torch_spacing(torch.linalg.norm(A_hat))
    I = torch.eye(A.shape[0])
    k = 1

    while not is_pd(A_hat):
        # compute smallest real part eigenvalue
        min_eigval = torch.min(torch.real(torch.linalg.eigvalsh(A_hat)))
        # adjust matrix
        A_hat += I*(-min_eigval*(k**2) + spacing)
        k += 1

    return A_hat


class MultivariateGaussian(LeafNode):
    r"""Multivariate Normal distribution for Torch backend.

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
        mean:
            A list, NumPy array or a torch tensor holding the means (:math:`\mu`) of each of the one-dimensional Normal distributions (defaults to all zeros).
            Has exactly as many elements as the scope of this leaf.
        cov:
            A list of lists, NumPy array or torch tensor (representing a two-dimensional :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
            of the scope) describing the covariances of the distribution (defaults to the identity matrix). The diagonal holds
            the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
    """
    def __init__(
        self,
        scope: Scope,
        mean: Optional[Union[List[float], torch.Tensor, np.ndarray]]=None,
        cov: Optional[Union[List[List[float]], torch.Tensor, np.ndarray]]=None,
    ) -> None:

        # check if scope contains duplicates
        if(len(set(scope.query)) != len(scope.query)):
            raise ValueError("Query scope for MultivariateGaussian contains duplicate variables.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for MultivariateGaussian should be empty, but was {scope.evidence}.")
        if len(scope.query) < 1:
            raise ValueError("Size of query scope for MultivariateGaussian must be at least 1.")

        super(MultivariateGaussian, self).__init__(scope=scope)

        if(mean is None):
            mean = torch.zeros((1,len(scope)))
        if(cov is None):
            cov = torch.eye(len(scope))

        # dimensions
        self.d = len(scope)

        # register mean vector as torch parameters
        self.mean = Parameter()

        # internally we use the lower triangular matrix (Cholesky decomposition) to encode the covariance matrix
        # register (auxiliary) values for diagonal and non-diagonal values of lower triangular matrix as torch parameters
        self.tril_diag_aux = Parameter()
        self.tril_nondiag = Parameter()

        # pre-compute and store indices of non-diagonal values for lower triangular matrix
        self.tril_nondiag_indices = torch.tril_indices(self.d, self.d, offset=-1)

        # set parameters
        self.set_params(mean, cov)

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
    def cov(self) -> torch.Tensor:
        # get lower triangular matrix
        L = self.covariance_tril
        # return covariance matrix
        return torch.matmul(L, L.T)

    @property
    def dist(self) -> D.Distribution:
        return D.MultivariateNormal(loc=self.mean, scale_tril=self.covariance_tril)

    def set_params(
        self,
        mean: Union[List[float], torch.Tensor, np.ndarray],
        cov: Union[List[List[float]], torch.Tensor, np.ndarray],
    ) -> None:

        if isinstance(mean, list):
            # convert float list to torch tensor
            mean = torch.tensor([float(v) for v in mean])
        elif isinstance(mean, np.ndarray):
            # convert numpy array to torch tensor
            mean = torch.from_numpy(mean).type(torch.get_default_dtype())

        if isinstance(cov, list):
            # convert numpy array to torch tensor
            cov = torch.tensor([[float(v) for v in row] for row in cov])
        elif isinstance(cov, np.ndarray):
            # convert numpy array to torch tensor
            cov = torch.from_numpy(cov).type(torch.get_default_dtype())

        # check mean vector for nan or inf values
        if torch.any(torch.isinf(mean)):
            raise ValueError(
                "Mean vector for MultivariateGaussian may not contain infinite values"
            )
        if torch.any(torch.isnan(mean)):
            raise ValueError("Mean vector for MultivariateGaussian may not contain NaN values")

        # dimensions
        d = mean.numel()

        # make sure that number of dimensions matches scope length
        if (
            (mean.ndim == 1 and mean.shape[0] != len(self.scope.query))
            or (mean.ndim == 2 and mean.shape[1] != len(self.scope.query))
            or mean.ndim > 2
        ):
            raise ValueError(
                f"Dimensions of mean vector for MultivariateGaussian should match scope size {len(self.scope.query)}, but was: {mean.shape}"
            )
    
        if(mean.ndim == 2):
            mean = mean.squeeze(0)

        # make sure that dimensions of covariance matrix are correct
        if cov.ndim != 2 or (
            cov.ndim == 2
            and (
                cov.shape[0] != len(self.scope.query)
                or cov.shape[1] != len(self.scope.query)
            )
        ):
            raise ValueError(
                f"Covariance matrix for MultivariateGaussian expected to be of shape ({len(self.scope.query), len(self.scope.query)}), but was: {cov.shape}"
            )

        # set mean vector
        self.mean.data = mean

        # check covariance matrix for nan or inf values
        if torch.any(torch.isinf(cov)):
            raise ValueError(
                "Covariance matrix vector for MultivariateGaussian may not contain infinite values"
            )
        if torch.any(torch.isnan(cov)):
            raise ValueError(
                "Covariance matrix for MultivariateGaussian may not contain NaN values"
            )

        # compute eigenvalues (can use eigvalsh here since we already know matrix is symmetric)        
        eigvals = torch.linalg.eigvalsh(cov)

        if torch.any(eigvals < 0.0):
            raise ValueError("Covariance matrix for MultivariateGaussian is not symmetric positive semi-definite (contains negative real eigenvalues).")

        # edge case: covariance matrix is positive semi-definite but NOT positive definite (needed for projection)
        if torch.any(eigvals == 0):
            # find nearest symmetric positive definite matrix in Frobenius norm
            cov = nearest_sym_pd(cov)

        # compute lower triangular matrix
        L = torch.linalg.cholesky(cov)  # type: ignore

        # set diagonal and non-diagonal values of lower triangular matrix
        self.tril_diag_aux.data = proj_bounded_to_real(torch.diag(L), lb=0.0)
        self.tril_nondiag.data = L[self.tril_nondiag_indices[0], self.tril_nondiag_indices[1]]

    def get_params(self) -> Tuple[List[float], List[List[float]]]:
        return self.mean.data.cpu().tolist(), self.cov.data.cpu().tolist()  # type: ignore

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

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scopes_out[0].query):
            raise ValueError(
                f"Expected scope_data to be of shape (n,{len(self.scopes_out[0].query)}), but was: {scope_data.shape}"
            )

        # different to univariate distributions, cannot simply check via torch distribution's support due to possible incomplete data in multivariate case; therefore do it ourselves (not difficult here since support is R)
        valid = torch.ones(scope_data.shape[0], 1, dtype=torch.bool)
        
        # check for infinite values (may return NaNs despite support)
        valid &= ~scope_data.isinf().sum(dim=1, keepdim=True).bool()

        return valid
    
    def marginalize(self, marg_rvs: Iterable[int]) -> Union["MultivariateGaussian", Gaussian, None]:
    
        # scope after marginalization (important: must remain order of scope indices since they map to the indices of the mean vector and covariance matrix!)
        marg_scope = []
        marg_scope_ids = []

        for rv in self.scope.query:
            if rv not in marg_rvs:
                marg_scope.append(rv)
                marg_scope_ids.append(self.scope.query.index(rv))

        # return univariate Gaussian if one-dimensional
        if(len(marg_scope) == 1):
            # note: Gaussian requires standard deviations instead of variance (take square root)
            return Gaussian(Scope(marg_scope), self.mean[marg_scope_ids[0]].detach().cpu().item(), torch.sqrt(self.cov[marg_scope_ids[0]][marg_scope_ids[0]].detach()).cpu().item())
        # entire node is marginalized over
        elif len(marg_scope) == 0:
            return None
        # node is partially marginalized over
        else:
            # compute marginalized mean vector and covariance matrix
            marg_mean = self.mean[marg_scope_ids]
            marg_cov = self.cov[marg_scope_ids][:, marg_scope_ids]

            return MultivariateGaussian(Scope(marg_scope), marg_mean, marg_cov)


@dispatch(memoize=True)
def marginalize(node: MultivariateGaussian, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[MultivariateGaussian,Gaussian,None]:
    return node.marginalize(marg_rvs)


@dispatch(memoize=True)
def toTorch(node: BaseMultivariateGaussian, dispatch_ctx: Optional[DispatchContext]=None) -> MultivariateGaussian:
    return MultivariateGaussian(node.scope, *node.get_params())


@dispatch(memoize=True)
def toBase(torch_node: MultivariateGaussian, dispatch_ctx: Optional[DispatchContext]=None) -> BaseMultivariateGaussian:
    return BaseMultivariateGaussian(torch_node.scope, *torch_node.get_params())