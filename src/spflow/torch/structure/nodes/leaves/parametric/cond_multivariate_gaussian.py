"""
Created on October 20, 2022

@authors: Philipp Deibert
"""
import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter
from typing import List, Tuple, Union, Optional, Iterable, Callable
from spflow.meta.scope.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.utils.nearest_sym_pd import nearest_sym_pd
from spflow.torch.structure.nodes.node import LeafNode
from spflow.torch.structure.nodes.leaves.parametric.cond_gaussian import CondGaussian
from spflow.base.structure.nodes.leaves.parametric.cond_multivariate_gaussian import CondMultivariateGaussian as BaseCondMultivariateGaussian
import warnings


class CondMultivariateGaussian(LeafNode):
    r"""Conditional multivariate Normal distribution for Torch backend.

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
        cond_f:
            TODO
    """
    def __init__(
        self,
        scope: Scope,
        cond_f: Optional[Callable]=None,
    ) -> None:

        # check if scope contains duplicates
        if(len(set(scope.query)) != len(scope.query)):
            raise ValueError("Query scope for CondMultivariateGaussian contains duplicate variables.")
        if len(scope.evidence):
            raise ValueError(f"Evidence scope for CondMultivariateGaussian should be empty, but was {scope.evidence}.")
        if len(scope.query) < 1:
            raise ValueError("Size of query scope for COndMultivariateGaussian must be at least 1.")

        super(CondMultivariateGaussian, self).__init__(scope=scope)

        # dimensions
        self.d = len(scope)

        self.set_cond_f(cond_f)

    def set_cond_f(self, cond_f: Optional[Callable]=None) -> None:
        self.cond_f = cond_f

    def dist(self, mean: torch.Tensor, cov: Optional[torch.Tensor]=None, cov_tril: Optional[torch.Tensor]=None) -> D.Distribution:
        if cov is None and cov_tril is None:
            raise ValueError("Calling 'dist' of CondMultivariateGaussian requries either 'cov' or 'cov_tril' to be specified.")
        elif cov is not None and cov_tril is not None:
            raise ValueError("Calling 'dist' of CondMultivariateGaussian requries either 'cov' or 'cov_tril' to be specified, but not both.")

        if cov is not None:
            return D.MultivariateNormal(loc=mean, covariance_matrix=cov)
        else:
            return D.MultivariateNormal(loc=mean, scale_tril=cov_tril)
    
    def retrieve_params(self, data: torch.Tensor, dispatch_ctx: DispatchContext) -> Tuple[torch.Tensor,Optional[torch.Tensor],Optional[torch.Tensor]]:
        
        mean, cov, cov_tril, cond_f = None, None, None, None
        specified_tril = False

        # check dispatch cache for required conditional parameters 'mean', 'cov'/'cov_tril'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if values for 'mean', 'cov'/'cov_tril' are specified (highest priority)
            if "mean" in args:
                mean = args["mean"]
            if "cov" in args:
                cov = args["cov"]
            if "cov_tril" in args:
                cov_tril = args["cov_tril"]
            # check if alternative function to provide 'mean', 'cov'/'cov_tril' is specified (second to highest priority)
            if "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'mean','cov'/'cov_tril' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'mean' or 'cov'/'cov_tril' nor 'cond_f' is specified (via node or arguments)
        if (mean is None or (cov is None and cov_tril is None)) and cond_f is None:
            raise ValueError("'CondMultivariateGaussian' requires either 'mean' and 'cov'/'cov_tril' or 'cond_f' to retrieve 'mean', 'cov'/'cov_tril' to be specified.")

        # if 'mean' or 'cov' not already specified, retrieve them
        if mean is None or (cov is None and cov_tril is None):
            params = cond_f(data)
            mean = params['mean']

            if 'cov' in params:
                cov = params['cov']
            if 'cov_tril' in params:
                cov_tril = params['cov_tril']
            
            if cov is None and cov_tril is None:
                raise ValueError("CondMultivariateGaussian requries either 'cov' or 'cov_tril' to be specified.")
            elif cov is not None and cov_tril is not None:
                raise ValueError("CondMultivariateGaussian requries either 'cov' or 'cov_tril' to be specified, but not both.")
        
        # cov_tril specified (and not cov)
        if cov_tril is not None:
            cov = cov_tril
            specified_tril = True

        if cov is not None:
            # cast lists to torch tensors
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

            # check covariance matrix for nan or inf values
            if torch.any(torch.isinf(cov)):
                raise ValueError(
                    "Covariance matrix vector for MultivariateGaussian may not contain infinite values"
                )
            if torch.any(torch.isnan(cov)):
                raise ValueError(
                    "Covariance matrix for MultivariateGaussian may not contain NaN values"
                )

            if specified_tril:
                # compute eigenvalues of cov variance matrix
                eigvals = torch.linalg.eigvalsh(torch.matmul(cov, cov.T))
            else:
                # compute eigenvalues (can use eigvalsh here since we already know matrix is symmetric)        
                eigvals = torch.linalg.eigvalsh(cov)

            if torch.any(eigvals < 0.0):
                raise ValueError("Covariance matrix for MultivariateGaussian is not symmetric positive semi-definite (contains negative real eigenvalues).")

        if specified_tril:
            return mean, None, cov_tril
        else:
            return mean, cov, None

    def get_params(self) -> Tuple:
        return tuple([])

    def check_support(self, scope_data: torch.Tensor) -> torch.Tensor:
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

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scopes_out[0].query):
            raise ValueError(
                f"Expected scope_data to be of shape (n,{len(self.scopes_out[0].query)}), but was: {scope_data.shape}"
            )

        # different to univariate distributions, cannot simply check via torch distribution's support due to possible incomplete data in multivariate case; therefore do it ourselves (not difficult here since support is R)
        valid = torch.ones(scope_data.shape[0], 1, dtype=torch.bool)
        
        # check for infinite values (may return NaNs despite support)
        valid &= ~scope_data.isinf().sum(dim=1, keepdim=True).bool()

        return valid
    
    def marginalize(self, marg_rvs: Iterable[int]) -> Union["CondMultivariateGaussian", CondGaussian, None]:

        # scope after marginalization (important: must remain order of scope indices since they map to the indices of the mean vector and covariance matrix!)
        marg_scope = []
        marg_scope_ids = []

        for rv in self.scope.query:
            if rv not in marg_rvs:
                marg_scope.append(rv)
                marg_scope_ids.append(self.scope.query.index(rv))

        # return univariate Gaussian if one-dimensional
        if(len(marg_scope) == 1):
            return CondGaussian(Scope(marg_scope))
        # entire node is marginalized over
        elif len(marg_scope) == 0:
            return None
        # node is partially marginalized over
        else:
            return CondMultivariateGaussian(Scope(marg_scope))


@dispatch(memoize=True)
def marginalize(node: CondMultivariateGaussian, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[CondMultivariateGaussian,CondGaussian,None]:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return node.marginalize(marg_rvs)


@dispatch(memoize=True)
def toTorch(node: BaseCondMultivariateGaussian, dispatch_ctx: Optional[DispatchContext]=None) -> CondMultivariateGaussian:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CondMultivariateGaussian(node.scope)


@dispatch(memoize=True)
def toBase(torch_node: CondMultivariateGaussian, dispatch_ctx: Optional[DispatchContext]=None) -> BaseCondMultivariateGaussian:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCondMultivariateGaussian(torch_node.scope)