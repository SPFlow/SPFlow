"""
Created on August 29, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import torch
from scipy.stats import gamma
from spflow.meta.dispatch.dispatch import dispatch
from spflow.torch.structure.nodes.leaves.parametric.gamma import Gamma


@dispatch(memoize=True) # TODO: swappable
def maximum_likelihood_estimation(leaf: Gamma, data: torch.Tensor, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None) -> None:
    """TODO."""

    # select relevant data for scope
    scope_data = data[:, leaf.scope.query]

    if torch.any(~leaf.check_support(scope_data)):
        raise ValueError("Encountered values outside of the support for 'Gamma'.")
    
    # NaN entries (no information)
    nan_mask = torch.isnan(scope_data)

    if torch.all(nan_mask):
        raise ValueError("Cannot compute maximum-likelihood estimation on nan-only data.")
    
    if nan_strategy is None and torch.any(nan_mask):
        raise ValueError("Maximum-likelihood estimation cannot be performed on missing data by default. Set a strategy for handling missing values if this is intended.")
    
    if isinstance(nan_strategy, str):
        if nan_strategy == "ignore":
            # simply ignore missing data
            scope_data = scope_data[~nan_mask]
        else:
            raise ValueError("Unknown strategy for handling missing (NaN) values for 'Gamma'.")
    elif isinstance(nan_strategy, Callable):
        scope_data = nan_strategy(scope_data)
    elif nan_strategy is not None:
        raise ValueError(f"Expected 'nan_strategy' to be of type '{type(str)}, or '{Callable}' or '{None}', but was of type {type(nan_strategy)}.")

    # compute two parameter gamma estimates according to (Mink, 2002): https://tminka.github.io/papers/minka-gamma.pdf
    # also see this VBA implementation for reference: https://github.com/jb262/MaximumLikelihoodGammaDist/blob/main/MLGamma.bas
    mean = scope_data.mean()
    log_mean = mean.log()
    mean_log = scope_data.log().mean()

    # start values
    alpha_prev = torch.tensor(0.0)
    alpha_est = 0.5 / (log_mean - mean_log)

    # iteratively compute alpha estimate
    while torch.abs(alpha_prev - alpha_est) > 1e-6:
        alpha_prev = alpha_est
        alpha_est = 1.0 / (1.0 / alpha_prev + (mean_log - log_mean + alpha_prev.log() - torch.digamma(alpha_prev)) / (alpha_prev**2 * (1.0 / alpha_prev - torch.polygamma(n=1, input=alpha_prev))))

    # compute beta estimate
    # NOTE: different to the original paper we compute the inverse since beta=1.0/scale
    beta_est = (alpha_est / mean)

    # edge case: if alpha/beta 0, set to larger value (should not happen, but just in case)
    if torch.isclose(alpha_est, torch.tensor(0.0)):
        alpha_est = 1e-8
    if torch.isclose(beta_est, torch.tensor(0.0)):
        beta_est = 1e-8

    # set parameters of leaf node
    leaf.set_params(alpha=alpha_est, beta=beta_est)