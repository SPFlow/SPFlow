"""
Created on October 18, 2022

@authors: Philipp Deibert
"""
from typing import Tuple, List, Union, Optional, Iterable, Union, Callable
import numpy as np
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import LeafNode
from spflow.base.structure.nodes.leaves.parametric.cond_gaussian import CondGaussian

from scipy.stats import multivariate_normal
from scipy.stats.distributions import rv_frozen


class CondMultivariateGaussian(LeafNode):
    r"""Conditional multivariate Normal distribution.

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
        cond_f:
            Callable that provides the conditional parameters (mean, std) of this distribution. TODO
    """
    def __init__(
        self,
        scope: Scope,
        cond_f: Optional[Callable]=None,
    ) -> None:

        # check if scope contains duplicates
        if(len(set(scope.query)) != len(scope.query)):
            raise ValueError("Query scope for MultivariateGaussian contains duplicate variables.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for MultivariateGaussian should be empty, but was {scope.evidence}.")
        if len(scope.query) < 1:
            raise ValueError("Size of query scope for MultivariateGaussian must be at least 1.")

        super(CondMultivariateGaussian, self).__init__(scope=scope)

        self.set_cond_f(cond_f)

    def set_cond_f(self, cond_f: Optional[Callable]=None) -> None:
        self.cond_f = cond_f

    def retrieve_params(self, data: np.ndarray, dispatch_ctx: DispatchContext) -> Tuple[np.ndarray,np.ndarray]:
        
        mean, cov, cond_f = None, None, None

        # check dispatch cache for required conditional parameters 'mean', 'cov'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if values for 'mean', 'cov' are specified (highest priority)
            if "mean" in args:
                mean = args["mean"]
            if "cov" in args:
                cov = args["cov"]
            # check if alternative function to provide 'mean', 'cov' is specified (second to highest priority)
            if "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'mean','cov' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'mean' or 'cov' nor 'cond_f' is specified (via node or arguments)
        if (mean is None or cov is None) and cond_f is None:
            raise ValueError("'CondMultivariateGaussian' requires either 'mean' and 'cov' or 'cond_f' to retrieve 'mean', 'cov' to be specified.")

        # if 'mean' or 'cov' not already specified, retrieve them
        if mean is None or cov is None:
            params = cond_f(data)
            mean = params['mean']
            cov = params['cov']
        
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
        if not np.all(cov == cov.T):
            raise ValueError("Covariance matrix for MultivariateGaussian must be symmetric")

        # test covariance matrix for positive semi-definiteness
        # NOTE: since we established in the test right before that matrix is symmetric we can use numpy's eigvalsh instead of eigvals
        if np.any(np.linalg.eigvalsh(cov) < 0):
            raise ValueError(
                "Covariance matrix for MultivariateGaussian must be positive semi-definite"
            )

        return mean, cov

    def dist(self, mean: np.ndarray, cov: np.ndarray) -> rv_frozen:
        return multivariate_normal(mean=mean, cov=cov)

    def get_params(self) -> Tuple:
        return tuple([])

    def check_support(self, scope_data: np.ndarray) -> np.ndarray:
        r"""Checks if instances are part of the support of the MultivariateGaussian distribution.

        .. math::

            \text{supp}(\text{MultivariateGaussian})=(-\infty,+\infty)^k
        
        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

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

        valid = np.ones(scope_data.shape, dtype=bool)

        # nan entries (regarded as valid)
        nan_mask = np.isnan(scope_data)

        # check for infinite values
        # additionally check for infinite values (may return NaNs despite support)
        valid[~nan_mask] &= ~np.isinf(scope_data[~nan_mask])

        return valid
    
    def marginalize(self, marg_rvs: Iterable[int]) -> Union["CondMultivariateGaussian",CondGaussian,None]:

        # scope after marginalization (important: must remain order of scope indices since they map to the indices of the mean vector and covariance matrix!)
        marg_scope = []
        marg_scope_ids = []

        for rv in self.scope.query:
            if rv not in marg_rvs:
                marg_scope.append(rv)
                marg_scope_ids.append(self.scope.query.index(rv))
        
        if any([rv in marg_rvs for rv in self.scope.evidence]):
            raise ValueError("")

        # return univariate Gaussian if one-dimensional
        if(len(marg_scope) == 1):
            # note: Gaussian requires standard deviations instead of variance (take square root)
            return CondGaussian(Scope(marg_scope))
        # entire node is marginalized over
        elif len(marg_scope) == 0:
            return None
        # node is partially marginalized over
        else:
            # compute marginalized mean vector and covariance matrix
            marg_scope_ids = [self.scope.query.index(rv) for rv in marg_scope]

            return CondMultivariateGaussian(Scope(marg_scope))


@dispatch(memoize=True)
def marginalize(node: CondMultivariateGaussian, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[CondMultivariateGaussian,CondGaussian,None]:
    return node.marginalize(marg_rvs)