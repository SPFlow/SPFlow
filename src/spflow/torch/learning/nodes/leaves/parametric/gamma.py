"""
Created on August 29, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import torch
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.nodes.leaves.parametric.gamma import Gamma


# TODO: MLE dispatch context?


@dispatch(memoize=True)
def maximum_likelihood_estimation(leaf: Gamma, data: torch.Tensor, weights: Optional[torch.Tensor]=None, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None) -> None:
    """TODO."""

    # select relevant data for scope
    scope_data = data[:, leaf.scope.query]

    if weights is None:
        weights = torch.ones(data.shape[0])

    if weights.ndim != 1 or weights.shape[0] != data.shape[0]:
        raise ValueError("Number of specified weights for maximum-likelihood estimation does not match number of data points.")

    # reshape weights
    weights = weights.reshape(-1, 1)

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
            weights = weights[~nan_mask]
        else:
            raise ValueError("Unknown strategy for handling missing (NaN) values for 'Gamma'.")
    elif isinstance(nan_strategy, Callable):
        scope_data = nan_strategy(scope_data)
    elif nan_strategy is not None:
        raise ValueError(f"Expected 'nan_strategy' to be of type '{type(str)}, or '{Callable}' or '{None}', but was of type {type(nan_strategy)}.")

    # normalize weights to sum to n_samples
    weights /=  weights.sum() / scope_data.shape[0]

    # compute two parameter gamma estimates according to (Mink, 2002): https://tminka.github.io/papers/minka-gamma.pdf
    # also see this VBA implementation for reference: https://github.com/jb262/MaximumLikelihoodGammaDist/blob/main/MLGamma.bas
    # adapted to take weights

    n_total = weights.sum()
    mean = (weights * scope_data).sum() / n_total
    log_mean = mean.log()
    mean_log = (weights * scope_data.log()).sum() / n_total

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

    # TODO: bias correction?

    # edge case: if alpha/beta 0, set to larger value (should not happen, but just in case)
    if torch.isclose(alpha_est, torch.tensor(0.0)):
        alpha_est = 1e-8
    if torch.isclose(beta_est, torch.tensor(0.0)):
        beta_est = 1e-8

    # set parameters of leaf node
    leaf.set_params(alpha=alpha_est, beta=beta_est)


@dispatch
def em(leaf: Gamma, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> None:

    with torch.no_grad():
        # ----- expectation step -----

        # get cached log-likelihood gradients w.r.t. module log-likelihoods
        expectations = dispatch_ctx.cache['log_likelihood'][leaf].grad
        # normalize expectations for better numerical stability
        expectations /= expectations.sum()

        # ----- maximization step -----

        # update parameters through maximum weighted likelihood estimation
        maximum_likelihood_estimation(leaf, data, weights=expectations.squeeze(1), bias_correction=False)

    # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation', we do not need to zero/None parameter gradients