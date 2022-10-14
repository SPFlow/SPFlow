"""
Created on August 29, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import numpy as np
import numpy.ma as ma
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.base.utils.nearest_sym_pd import nearest_sym_pd
from spflow.base.structure.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian


@dispatch(memoize=True)
def maximum_likelihood_estimation(leaf: MultivariateGaussian, data: np.ndarray, weights: Optional[np.ndarray]=None, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None, dispatch_ctx: Optional[DispatchContext]=None) -> None:
    """TODO."""

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # select relevant data for scope
    scope_data = data[:, leaf.scope.query]

    if weights is None:
        weights = np.ones(data.shape[0])

    if weights.ndim != 1 or weights.shape[0] != data.shape[0]:
        raise ValueError("Number of specified weights for maximum-likelihood estimation does not match number of data points.")

    # reshape weights
    weights = weights.reshape(-1, 1)

    if np.any(~leaf.check_support(scope_data)):
        raise ValueError("Encountered values outside of the support for 'MultivariateGaussian'.")

    # NaN entries (no information)
    nan_mask = np.isnan(scope_data)

    if np.all(nan_mask):
        raise ValueError("Cannot compute maximum-likelihood estimation on nan-only data.")
    
    if nan_strategy is None and np.any(nan_mask):
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
        # compute mean of available data
        mean_est = np.sum(weights * np.nan_to_num(scope_data, nan=0.0), axis=0)/(weights * ~nan_mask).sum(axis=0)
        # compute covariance of full samples only!
        full_sample_mask = (~nan_mask).sum(axis=1) == scope_data.shape[1]
        cov_est = np.cov(scope_data[full_sample_mask].T, aweights=weights[full_sample_mask].squeeze(-1), ddof=1 if bias_correction else 0)
    else:
        # calculate mean and standard deviation from data
        mean_est = np.mean(scope_data, axis=0)
        cov_est = np.cov(scope_data.T, aweights=weights.squeeze(-1), ddof=1 if bias_correction else 0)

    if len(leaf.scope.query) == 1:
        cov_est = cov_est.reshape(1,1)
    
    # edge case (if all values are the same, not enough samples or very close to each other)
    for i in range(scope_data.shape[1]):
        if np.isclose(cov_est[i][i], 0):
            cov_est[i][i] = 1e-8
    
    # sometimes numpy returns a matrix with negative eigenvalues (i.e., not a valid positive semi-definite matrix)
    if np.any(np.linalg.eigvalsh(cov_est) < 0):
        # compute nearest symmetric positive semidefinite matrix
        cov_est = nearest_sym_pd(cov_est)
    
    # set parameters of leaf node
    leaf.set_params(mean=mean_est, cov=cov_est)