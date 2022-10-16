"""
Created on August 29, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import torch
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.utils.nearest_sym_pd import nearest_sym_pd
from spflow.torch.structure.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian


@dispatch(memoize=True)
def maximum_likelihood_estimation(leaf: MultivariateGaussian, data: torch.Tensor, weights: Optional[torch.Tensor]=None, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None, dispatch_ctx: Optional[DispatchContext]=None) -> None:
    """TODO."""

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # select relevant data for scope
    scope_data = data[:, leaf.scope.query]

    if weights is None:
        weights = torch.ones(data.shape[0])

    if weights.ndim != 1 or weights.shape[0] != data.shape[0]:
        raise ValueError("Number of specified weights for maximum-likelihood estimation does not match number of data points.")

    # reshape weights
    weights = weights.reshape(-1, 1)

    if torch.any(~leaf.check_support(scope_data)):
        raise ValueError("Encountered values outside of the support for 'MultivariateGaussian'.")

    # NaN entries (no information)
    nan_mask = torch.isnan(scope_data)

    if torch.all(nan_mask):
        raise ValueError("Cannot compute maximum-likelihood estimation on nan-only data.")
    
    if nan_strategy is None and torch.any(nan_mask):
        raise ValueError("Maximum-likelihood estimation cannot be performed on missing data by default. Set a strategy for handling missing values if this is intended.")
    
    if isinstance(nan_strategy, str):
        if nan_strategy == "ignore":
            pass # handle it during computation
        else:
            raise ValueError("Unknown strategy for handling missing (NaN) values for 'MultivariateGaussian'.")
    elif isinstance(nan_strategy, Callable):
        scope_data = nan_strategy(scope_data)
        # TODO: how to handle weights?
    elif nan_strategy is not None:
        raise ValueError(f"Expected 'nan_strategy' to be of type '{type(str)}, or '{Callable}' or '{None}', but was of type {type(nan_strategy)}.")
    
    # normalize weights to sum to n_samples
    weights /= weights.sum() / scope_data.shape[0]

    if nan_strategy == "ignore":
        n_total = (weights * ~nan_mask).sum(dim=0)
        # compute mean of available data
        mean_est = torch.sum(weights * torch.nan_to_num(scope_data, nan=0.0), dim=0) / n_total
        # compute covariance of full samples only!
        full_sample_mask = (~nan_mask).sum(dim=1) == scope_data.shape[1]
        cov_est = torch.cov(scope_data[full_sample_mask].T, aweights=weights[full_sample_mask].squeeze(-1), correction=1 if bias_correction else 0)
    else:
        n_total = (weights * ~nan_mask).sum(dim=0)
        # calculate mean and standard deviation from data
        mean_est = (weights * scope_data).sum(dim=0) / n_total
        cov_est = torch.cov(scope_data.T, aweights=weights.squeeze(-1), correction=1 if bias_correction else 0)

    if len(leaf.scope.query) == 1:
        cov_est = cov_est.reshape(1,1)
    
    # edge case (if all values are the same, not enough samples or very close to each other)
    for i in range(scope_data.shape[1]):
        if torch.isclose(cov_est[i][i], torch.tensor(0.0)):
            cov_est[i][i] = 1e-8    
    
    # sometimes numpy returns a matrix with non-positive eigenvalues (i.e., not a valid positive definite matrix)
    # NOTE: we need test for non-positive here instead of negative for NumPy, because we need to be able to perform cholesky decomposition
    if torch.any(torch.linalg.eigvalsh(cov_est) <= 0):
        # compute nearest symmetric positive semidefinite matrix
        cov_est = nearest_sym_pd(cov_est)

    # set parameters of leaf node
    leaf.set_params(mean=mean_est, cov=cov_est)


@dispatch(memoize=True)
def em(leaf: MultivariateGaussian, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> None:

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    with torch.no_grad():
        # ----- expectation step -----

        # get cached log-likelihood gradients w.r.t. module log-likelihoods
        expectations = dispatch_ctx.cache['log_likelihood'][leaf].grad
        # normalize expectations for better numerical stability
        expectations /= expectations.sum()

        # ----- maximization step -----

        # update parameters through maximum weighted likelihood estimation
        maximum_likelihood_estimation(leaf, data, weights=expectations.squeeze(1), bias_correction=False, dispatch_ctx=dispatch_ctx)

    # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation', we do not need to zero/None parameter gradients