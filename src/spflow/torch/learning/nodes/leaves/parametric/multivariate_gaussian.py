"""
Created on August 29, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import torch
from spflow.meta.dispatch.dispatch import dispatch
from spflow.torch.structure.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian, nearest_sym_pd



@dispatch(memoize=True) # TODO: swappable
def maximum_likelihood_estimation(leaf: MultivariateGaussian, data: torch.Tensor, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None) -> None:
    """TODO."""

    # select relevant data for scope
    scope_data = data[:, leaf.scope.query]

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
    elif nan_strategy is not None:
        raise ValueError(f"Expected 'nan_strategy' to be of type '{type(str)}, or '{Callable}' or '{None}', but was of type {type(nan_strategy)}.")

    if nan_strategy == "ignore":
        # compute mean of available data
        mean_est = torch.sum(torch.nan_to_num(scope_data, nan=0.0), dim=0)/~nan_mask.sum(dim=0)
        # compute covariance of full samples only!
        cov_est = torch.cov(scope_data[(~nan_mask).sum(dim=1) == scope_data.shape[1]].T, correction=1 if bias_correction else 0)
        
        # if only one full sample all values are nan
    else:
        # calculate mean and standard deviation from data
        mean_est = torch.mean(scope_data, dim=0)
        cov_est = torch.cov(scope_data.T, correction=1 if bias_correction else 0)

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