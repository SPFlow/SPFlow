"""
Created on September 25, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import torch
from scipy.stats import gamma
from spflow.meta.dispatch.dispatch import dispatch
from spflow.torch.structure.layers.leaves.parametric.gamma import GammaLayer


@dispatch(memoize=True) # TODO: swappable
def maximum_likelihood_estimation(layer: GammaLayer, data: torch.Tensor, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None) -> None:
    """TODO."""

    # select relevant data for scope
    scope_data = torch.hstack([data[:, scope.query] for scope in layer.scopes_out])

    if torch.any(~layer.check_support(scope_data)):
        raise ValueError("Encountered values outside of the support for 'GammaLayer'.")
    
    # NaN entries (no information)
    nan_mask = torch.isnan(scope_data)

    # check if any columns (i.e., data for a output scope) contain only NaN values
    if torch.any(nan_mask.sum(dim=0) == scope_data.shape[0]):
        raise ValueError("Cannot compute maximum-likelihood estimation on nan-only data for a specified scope.")

    if nan_strategy is None and torch.any(nan_mask):
        raise ValueError("Maximum-likelihood estimation cannot be performed on missing data by default. Set a strategy for handling missing values if this is intended.")
    
    if isinstance(nan_strategy, str):
        # simply ignore missing data
        if nan_strategy == "ignore":
            # compute some prelimiary values
            mean = torch.nan_to_num(scope_data, nan=0.0).sum(dim=0)/(~nan_mask).sum(dim=0)
            log_mean = mean.log()
            mean_log = torch.nan_to_num(scope_data, nan=1.0).log().sum(dim=0)/(~nan_mask).sum(dim=0) # covert nan to 1s instead of 0s due to log
        else:
            raise ValueError("Unknown strategy for handling missing (NaN) values for 'GammaLayer'.")
    elif isinstance(nan_strategy, Callable) or nan_strategy is None:
        if isinstance(nan_strategy, Callable):
            scope_data = nan_strategy(scope_data)
        # compute some prelimiary values
        mean = scope_data.mean(dim=0)
        log_mean = mean.log()
        mean_log = scope_data.log().mean(dim=0)
    else:
        raise ValueError(f"Expected 'nan_strategy' to be of type '{type(str)}, or '{Callable}' or '{None}', but was of type {type(nan_strategy)}.")

    # compute two parameter gamma estimates according to (Mink, 2002): https://tminka.github.io/papers/minka-gamma.pdf
    # also see this VBA implementation for reference: https://github.com/jb262/MaximumLikelihoodGammaDist/blob/main/MLGamma.bas
    
    # mean, log_mean and mean_log already calculated above

    # start values
    alpha_prev = torch.zeros(scope_data.shape[1])
    alpha_est = 0.5 / (log_mean - mean_log)

    # iteratively compute alpha estimate
    while torch.any(torch.abs(alpha_prev - alpha_est) > 1e-6):
        # mask to only further refine relevant estimates
        iter_mask = torch.abs(alpha_prev - alpha_est) > 1e-6
        alpha_prev = alpha_est
        alpha_est[iter_mask] = 1.0 / (1.0 / alpha_prev[iter_mask] + (mean_log[iter_mask] - log_mean[iter_mask] + alpha_prev[iter_mask].log() - torch.digamma(alpha_prev[iter_mask])) / (alpha_prev[iter_mask]**2 * (1.0 / alpha_prev[iter_mask] - torch.polygamma(n=1, input=alpha_prev[iter_mask]))))

    # compute beta estimate
    # NOTE: different to the original paper we compute the inverse since beta=1.0/scale
    beta_est = (alpha_est / mean)

    # edge case: if alpha/beta 0, set to larger value (should not happen, but just in case)
    alpha_est[torch.allclose(alpha_est, torch.tensor(0.0))] = 1e-8
    beta_est[torch.allclose(beta_est, torch.tensor(0.0))] = 1e-8

    # set parameters of leaf node
    layer.set_params(alpha=alpha_est, beta=beta_est)