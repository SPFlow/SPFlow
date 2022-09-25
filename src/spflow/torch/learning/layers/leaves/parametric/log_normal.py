"""
Created on September 25, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import torch
from spflow.meta.dispatch.dispatch import dispatch
from spflow.torch.structure.layers.leaves.parametric.log_normal import LogNormalLayer


@dispatch(memoize=True) # TODO: swappable
def maximum_likelihood_estimation(layer: LogNormalLayer, data: torch.Tensor, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None) -> None:
    """TODO."""

    # select relevant data for scope
    scope_data = torch.hstack([data[:, scope.query] for scope in layer.scopes_out])

    if torch.any(~layer.check_support(scope_data)):
        raise ValueError("Encountered values outside of the support for 'LogNormalLayer'.")

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
            n_total = (~nan_mask).sum(dim=0)
            mean_est = torch.nan_to_num(scope_data, nan=1.0).log().sum(dim=0)/n_total

            if bias_correction:
                n_total -= 1
            
            std_est = torch.sqrt(
                torch.nan_to_num(scope_data.log()-mean_est, nan=0.0).pow(2).sum(dim=0) / n_total
            )
        else:
            raise ValueError("Unknown strategy for handling missing (NaN) values for 'LogNormalLayer'.")
    elif isinstance(nan_strategy, Callable) or nan_strategy is None:
        if isinstance(nan_strategy, Callable):
            scope_data = nan_strategy(scope_data)
        # calculate mean and standard deviation from data
        mean_est = torch.log(scope_data).mean(dim=0)
        std_est = torch.log(scope_data).std(unbiased=True if bias_correction else False, dim=0)
    else:
        raise ValueError(f"Expected 'nan_strategy' to be of type '{type(str)}, or '{Callable}' or '{None}', but was of type {type(nan_strategy)}.")

    # edge case (if all values are the same, not enough samples or very close to each other)
    std_est[torch.allclose(std_est, torch.tensor(0.0))] = 1e-8
    std_est[torch.isnan(std_est)] = 1e-8

    # set parameters of leaf node
    layer.set_params(mean=mean_est, std=std_est)