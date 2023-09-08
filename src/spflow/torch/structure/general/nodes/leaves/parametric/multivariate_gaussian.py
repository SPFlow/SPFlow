"""Contains Multivariate Normal leaf node for SPFlow in the ``torch`` backend.
"""
import warnings
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter

from spflow.base.structure.general.nodes.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussian as BaseMultivariateGaussian,
)
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_multivariate_gaussian import MultivariateGaussian as GeneralMultivariateGaussian
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes, MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.nodes.leaf_node import LeafNode
from spflow.torch.structure.general.nodes.leaves.parametric.gaussian import Gaussian
from spflow.torch.utils.nearest_sym_pd import nearest_sym_pd
from spflow.torch.utils.projections import proj_bounded_to_real, proj_real_to_bounded


class MultivariateGaussian(LeafNode):
    r"""Multivariate Gaussian distribution leaf node in the 'base' backend.

    Represents a multivariate Gaussian distribution, with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{\sqrt{(2\pi)^d\det\Sigma}}\exp\left(-\frac{1}{2} (x-\mu)^T\Sigma^{-1}(x-\mu)\right)

    where
        - :math:`d` is the dimension of the distribution
        - :math:`x` is the :math:`d`-dim. vector of observations
        - :math:`\mu` is the :math:`d`-dim. mean vector
        - :math:`\Sigma` is the :math:`d\times d` covariance matrix

    Note, that different to ``MultivariateGaussian`` in the ``base`` backend, the ``torch`` implementation only accepts positive definite (as opposed to positive semi-definite) covariance matrices.
    Internally :math:`\Sigma` is represented using two unbounded parametes that are combined and projected to yield a positive definite matrix, representing the actual covariance matrix.

    Attributes:
        mean:
            A two-dimensional PyTorch tensor containing the means (:math:`\mu`) of each of the one-dimensional Gaussian distributions.
            Has exactly as many elements as the scope of this leaf.
        tril_diag_aux:
            Unbounded one-dimensional PyTorch tensor containing values that are projected onto the range :math:`(0,\infty)` for the diagonal entries of the lower triangular Cholesky decomposition of the actual covariance matrix.
        tril_nondiag:
            Unbounded one-dimensional PyTorch tensor containing the values of the non-diagonal elements of the lower triangular Cholesky decomposition of the actual covariance matrix
        cov:
            Two-dimensional PyTorch tensor (representing a :math:`d\times d` symmetric positive definite matrix, where :math:`d` is the length
            of the scope) describing the covariances of the distribution. The diagonal holds the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
            Projected from ``tril_diag_aux`` and ``tril_nondiag``.
    """

    def __init__(
        self,
        scope: Scope,
        mean: Optional[Union[List[float], torch.Tensor, np.ndarray]] = None,
        cov: Optional[Union[List[List[float]], torch.Tensor, np.ndarray]] = None,
    ) -> None:
        r"""Initializes ``MultivariateGaussian`` leaf node.

        Args:
            mean:
                A list of floating points, one-dimensional NumPy array or one-dimensional PyTorch tensor containing the means (:math:`\mu`) of each of the one-dimensional Normal distributions.
                Must have exactly as many elements as the scope of this leaf.
                Defaults to all zeros.
            cov:
                A list of lists of floating points, a two-dimensional NumPy array or a two-dimensional PyTorch array (representing a :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
                of the scope) describing the covariances of the distribution. The diagonal holds the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
                Defaults to the identity matrix.
        """
        # check if scope contains duplicates
        if len(set(scope.query)) != len(scope.query):
            raise ValueError("Query scope for 'MultivariateGaussian' contains duplicate variables.")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'MultivariateGaussian' should be empty, but was {scope.evidence}.")
        if len(scope.query) < 1:
            raise ValueError("Size of query scope for 'MultivariateGaussian' must be at least 1.")

        super().__init__(scope=scope)

        if mean is None:
            mean = torch.zeros((1, len(scope)))
        if cov is None:
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
        self.backend = "pytorch"

    @property
    def covariance_tril(self) -> torch.Tensor:
        """Returns the lower triangular matrix of the Cholesky-decomposed covariance matrix."""
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
        """Returns the covariance matrix."""
        # get lower triangular matrix
        L = self.covariance_tril
        # return covariance matrix
        return torch.matmul(L, L.T)

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``MultivariateGaussian`` can represent a single univariate node with ``MetaType.Continuous`` or ``GaussianType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf only has one output
        if len(signatures) != 1:
            return False

        # get single output signature
        feature_ctx = signatures[0]
        domains = feature_ctx.get_domains()

        # leaf is a single non-conditional (possibly multivariate) node
        if len(domains) < 1 or len(feature_ctx.scope.query) != len(domains) or len(feature_ctx.scope.evidence) != 0:
            return False

        # leaf is a continuous (multivariate) Gaussian distribution
        if not all(
            [
                domain == FeatureTypes.Continuous
                or domain == FeatureTypes.Gaussian
                or isinstance(domain, FeatureTypes.Gaussian)
                for domain in domains
            ]
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "MultivariateGaussian":
        """Creates an instance from a specified signature.

        Returns:
            ``MultivariateGaussian`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'MultivariateGaussian' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]

        mean, cov = np.zeros(len(feature_ctx.scope.query)), np.eye(len(feature_ctx.scope.query))

        for i, domain in enumerate(feature_ctx.get_domains()):
            # read or initialize parameters
            if domain == MetaType.Continuous:
                pass
            elif domain == FeatureTypes.Gaussian:
                # instantiate object
                domain = domain()
                mean[i], cov[i][i] = domain.mean, domain.std
            elif isinstance(domain, FeatureTypes.Gaussian):
                mean[i], cov[i][i] = domain.mean, domain.std
            else:
                raise ValueError(
                    f"Unknown signature type {domain} for 'MultivariateGaussian' that was not caught during acception checking."
                )

        return MultivariateGaussian(feature_ctx.scope, mean=mean, cov=cov)

    @property
    def dist(self) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Returns:
            ``torch.distributions.MultivariateNormal`` instance.
        """
        return D.MultivariateNormal(loc=self.mean, scale_tril=self.covariance_tril)

    def set_params(
        self,
        mean: Union[List[float], torch.Tensor, np.ndarray],
        cov: Union[List[List[float]], torch.Tensor, np.ndarray],
    ) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            mean:
                A list of floating points, one-dimensional NumPy array or one-dimensional PyTorch tensor containing the means (:math:`\mu`) of each of the one-dimensional Normal distributions.
                Must have exactly as many elements as the scope of this leaf.
                Defaults to all zeros.
            cov:
                A list of lists of floating points, a two-dimensional NumPy array or a two-dimensional PyTorch array (representing a :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
                of the scope) describing the covariances of the distribution. The diagonal holds the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
                If a positive semi-definite matrix is specified, the closest positive definite matrix in the Frobenius norm is used instead.
                Defaults to the identity matrix.
        """
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
            raise ValueError("Value of 'mean' for 'MultivariateGaussian' may not contain infinite values")
        if torch.any(torch.isnan(mean)):
            raise ValueError("Value of 'mean' for 'MultivariateGaussian' may not contain NaN values")

        # make sure that number of dimensions matches scope length
        if (
            (mean.ndim == 1 and mean.shape[0] != len(self.scope.query))
            or (mean.ndim == 2 and mean.shape[1] != len(self.scope.query))
            or mean.ndim > 2
        ):
            raise ValueError(
                f"Dimensions of 'mean' for 'MultivariateGaussian' should match scope size {len(self.scope.query)}, but was: {mean.shape}"
            )

        if mean.ndim == 2:
            mean = mean.squeeze(0)

        # make sure that dimensions of covariance matrix are correct
        if cov.ndim != 2 or (
            cov.ndim == 2 and (cov.shape[0] != len(self.scope.query) or cov.shape[1] != len(self.scope.query))
        ):
            raise ValueError(
                f"Value of 'cov' for 'MultivariateGaussian' expected to be of shape ({len(self.scope.query), len(self.scope.query)}), but was: {cov.shape}"
            )

        # set mean vector
        self.mean.data = mean

        # check covariance matrix for nan or inf values
        if torch.any(torch.isinf(cov)):
            raise ValueError("Value of 'cov' for 'MultivariateGaussian' may not contain infinite values")
        if torch.any(torch.isnan(cov)):
            raise ValueError("Value of 'cov' for 'MultivariateGaussian' may not contain NaN values")

        # compute eigenvalues (can use eigvalsh here since we already know matrix is symmetric)
        eigvals = torch.linalg.eigvalsh(cov)

        if torch.any(eigvals < 0.0):
            raise ValueError(
                "Value of 'cov' for 'MultivariateGaussian' is not symmetric positive semi-definite (contains negative real eigenvalues)."
            )

        # edge case: covariance matrix is positive semi-definite but NOT positive definite (needed for projection)
        if torch.any(eigvals == 0):
            warnings.warn(
                "Value of 'cov' for 'MultivariateGaussian' is positive semi-definite, but not positive definite. Using closest positive definite matrix instead. Required for interal projection of learnable parameters.",
                RuntimeWarning,
            )
            # find nearest symmetric positive definite matrix in Frobenius norm
            cov = nearest_sym_pd(cov)

        # compute lower triangular matrix
        L = torch.linalg.cholesky(cov)  # type: ignore

        # set diagonal and non-diagonal values of lower triangular matrix
        self.tril_diag_aux.data = proj_bounded_to_real(torch.diag(L), lb=0.0)
        self.tril_nondiag.data = L[self.tril_nondiag_indices[0], self.tril_nondiag_indices[1]]

    def get_trainable_params(self) -> Tuple[List[float], List[List[float]]]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of a one-dimensional and a two-dimensional PyTorch tensor representing the mean and covariance matrix, respectively.
        """

        #return self.mean.data.cpu().detach().tolist(), self.cov.data.cpu().detach().tolist()  # type: ignore
        return [self.mean, self.tril_diag_aux, self.tril_nondiag]  # type: ignore

    def get_params(self) -> Tuple[List[float], List[List[float]]]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of a one-dimensional and a two-dimensional PyTorch tensor representing the mean and covariance matrix, respectively.
        """
        return self.mean.data.cpu().detach().tolist(), self.cov.data.cpu().detach().tolist()  # type: ignore

    def check_support(self, data: torch.Tensor, is_scope_data: bool = False) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Multivariate Gaussian distribution, which is:

        .. math::

            \text{supp}(\text{MultivariateGaussian})=(-\infty,+\infty)^k

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional PyTorch tensor containing sample instances.
                Each row is regarded as a sample.
                Unless ``is_scope_data`` is set to True, it is assumed that the relevant data is located in the columns corresponding to the scope indices.
            is_scope_data:
                Boolean indicating if the given data already contains the relevant data for the leaf's scope in the correct order (True) or if it needs to be extracted from the full data set.
                Defaults to False.

        Returns:
            Two-dimensional PyTorch tensor indicating for each instance, whether they are part of the support (True) or not (False).
        """
        if is_scope_data:
            scope_data = data
        else:
            # select relevant data for scope
            scope_data = data[:, self.scope.query]

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scopes_out[0].query):
            raise ValueError(
                f"Expected 'scope_data' to be of shape (n,{len(self.scopes_out[0].query)}), but was: {scope_data.shape}"
            )

        # different to univariate distributions, cannot simply check via torch distribution's support due to possible incomplete data in multivariate case; therefore do it ourselves (not difficult here since support is R)
        valid = torch.ones(scope_data.shape[0], 1, dtype=torch.bool)

        # check for infinite values (may return NaNs despite support)
        valid &= ~scope_data.isinf().sum(dim=1, keepdim=True).bool()

        return valid


@dispatch(memoize=True)  # type: ignore
def marginalize(
    node: MultivariateGaussian,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[MultivariateGaussian, Gaussian, None]:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # scope after marginalization (important: must remain order of scope indices since they map to the indices of the mean vector and covariance matrix!)
    marg_scope = []
    marg_scope_ids = []

    scope = node.scope

    for rv in scope.query:
        if rv not in marg_rvs:
            marg_scope.append(rv)
            marg_scope_ids.append(scope.query.index(rv))

    # return univariate Gaussian if one-dimensional
    if len(marg_scope) == 1:
        # note: Gaussian requires standard deviations instead of variance (take square root)
        return Gaussian(
            Scope(marg_scope),
            node.mean[marg_scope_ids[0]].detach().cpu().item(),
            torch.sqrt(node.cov[marg_scope_ids[0]][marg_scope_ids[0]].detach()).cpu().item(),
        )
    # entire node is marginalized over
    elif len(marg_scope) == 0:
        return None
    # node is partially marginalized over
    else:
        # compute marginalized mean vector and covariance matrix
        marg_mean = node.mean[marg_scope_ids]
        marg_cov = node.cov[marg_scope_ids][:, marg_scope_ids]

        return MultivariateGaussian(Scope(marg_scope), marg_mean, marg_cov)


@dispatch(memoize=True)  # type: ignore
def toTorch(
    node: BaseMultivariateGaussian,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> MultivariateGaussian:
    """Conversion for ``MultivariateGaussian`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return MultivariateGaussian(node.scope, *node.get_trainable_params())


@dispatch(memoize=True)  # type: ignore
def toBase(
    torch_node: MultivariateGaussian,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> BaseMultivariateGaussian:
    """Conversion for ``MultivariateGaussian`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseMultivariateGaussian(torch_node.scope, *torch_node.get_trainable_params())

@dispatch(memoize=True)  # type: ignore
def updateBackend(leaf_node: MultivariateGaussian, dispatch_ctx: Optional[DispatchContext] = None):
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return GeneralMultivariateGaussian(scope=leaf_node.scope, mean=leaf_node.mean.data.numpy(), cov=leaf_node.cov.data.numpy())
