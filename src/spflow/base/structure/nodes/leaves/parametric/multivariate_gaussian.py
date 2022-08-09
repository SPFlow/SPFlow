"""
Created on November 6, 2021

@authors: Philipp Deibert, Bennet Wittelsbach
"""
from typing import Tuple, List, Union, Optional, Iterable
import numpy as np
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import LeafNode
from spflow.base.structure.nodes.leaves.parametric.gaussian import Gaussian

from scipy.stats import multivariate_normal
from scipy.stats.distributions import rv_frozen


class MultivariateGaussian(LeafNode):
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
            Scope object specifying the variable scope.
        mean:
            A list, NumPy array or a PyTorch tensor holding the means (:math:`\mu`) of each of the one-dimensional Normal distributions (defaults to all zeros).
            Has exactly as many elements as the scope of this leaf.
        cov:
            A list of lists, NumPy array or PyTorch tensor (representing a two-dimensional :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
            of the scope) describing the covariances of the distribution (defaults to the identity matrix). The diagonal holds
            the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
    """
    def __init__(
        self,
        scope: Scope,
        mean: Optional[Union[List[float], np.ndarray]]=None,
        cov: Optional[Union[List[List[float]], np.ndarray]]=None,
    ) -> None:

        # check if scope contains duplicates
        if(len(set(scope.query)) != len(scope.query)):
            raise ValueError("Query scope for MultivariateGaussian contains duplicate variables.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for MultivariateGaussian should be empty, but was {scope.evidence}.")

        super(MultivariateGaussian, self).__init__(scope=scope)

        if(mean is None):
            mean = np.zeros((1,len(scope.query)))
        if(cov is None):
            cov = np.eye(len(scope.query))

        self.set_params(mean, cov)
    
    @property
    def dist(self) -> rv_frozen:
        return multivariate_normal(mean=self.mean, cov=self.cov)

    def set_params(
        self,
        mean: Union[List[float], np.ndarray],
        cov: Union[List[List[float]], np.ndarray],
    ) -> None:

        # cast lists to numpy arrays
        if isinstance(mean, List):
            mean = np.array(mean)
        if isinstance(cov, List):
            cov = np.array(cov)

        # check mean vector dimensions
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

        # check mean vector for nan or inf values
        if np.any(np.isinf(mean)):
            raise ValueError("Mean vector for MultivariateGaussian may not contain infinite values")
        if np.any(np.isnan(mean)):
            raise ValueError("Mean vector for MultivariateGaussian may not contain NaN values")

        # test whether or not matrix has correct shape
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

        # check covariance matrix for nan or inf values
        if np.any(np.isinf(cov)):
            raise ValueError(
                "Covariance matrix for MultivariateGaussian may not contain infinite values"
            )
        if np.any(np.isnan(cov)):
            raise ValueError(
                "Covariance matrix for MultivariateGaussian may not contain NaN values"
            )

        # test covariance matrix for symmetry
        if not np.allclose(cov, cov.T):
            raise ValueError("Covariance matrix for MultivariateGaussian must be symmetric")
        # test covariance matrix for positive semi-definiteness
        if np.any(np.linalg.eigvals(cov) < 0):
            raise ValueError(
                "Covariance matrix for MultivariateGaussian must be positive semi-definite"
            )

        self.mean = mean
        self.cov = cov

    def get_params(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.mean, self.cov

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

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scope.query):
            raise ValueError(
                f"Expected scope_data to be of shape (n,{len(self.scope.query)}), but was: {scope_data.shape}"
            )

        valid = np.ones(scope_data.shape[0], dtype=bool)

        # check for infinite values
        # additionally check for infinite values (may return NaNs despite support)
        valid &= ~np.isinf(scope_data).sum(axis=-1).astype(bool)

        return valid
    
    def marginalize(self, marg_rvs: Iterable[int]) -> Union["MultivariateGaussian",Gaussian,None]:

        # scope after marginalization (important: must remain order of scope indices since they map to the indices of the mean vector and covariance matrix!)
        marg_scope = [rv for rv in self.scope.query if rv not in marg_rvs]

        # return univariate Gaussian if one-dimensional
        if(len(marg_scope) == 1):
            # note: Gaussian requires standard deviations instead of variance (take square root)
            return Gaussian(Scope(marg_scope), self.mean[marg_scope[0]], np.sqrt(self.cov[marg_scope[0]][marg_scope[0]]))
        # entire node is marginalized over
        elif not marg_scope:
            return None
        # node is partially marginalized over
        else:
            # compute marginalized mean vector and covariance matrix
            marg_mean = self.mean[marg_scope]
            marg_cov = self.cov[marg_scope][:, marg_scope]

            return MultivariateGaussian(Scope(marg_scope), marg_mean, marg_cov)


@dispatch(memoize=True)
def marginalize(node: MultivariateGaussian, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[MultivariateGaussian,Gaussian,None]:
    return node.marginalize(marg_rvs)