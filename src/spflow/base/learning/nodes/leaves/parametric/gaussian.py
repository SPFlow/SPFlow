"""
Created on August 29, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import numpy as np
from spflow.meta.dispatch.dispatch import dispatch
from spflow.base.structure.nodes.leaves.parametric.gaussian import Gaussian


@dispatch(memoize=True) # TODO: swappable
def maximum_likelihood_estimation(leaf: Gaussian, data: np.ndarray, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None) -> None:
    """TODO."""

    # select relevant data for scope
    scope_data = data[:, leaf.scope.query]

    if np.any(~leaf.check_support(scope_data)):
        raise ValueError("Encountered values outside of the support for 'Gaussian'.")

    # NaN entries (no information)
    nan_mask = np.isnan(scope_data)

    if np.all(nan_mask):
        raise ValueError("Cannot compute maximum-likelihood estimation on nan-only data.")

    if nan_strategy is None and np.any(nan_mask):
        raise ValueError("Maximum-likelihood estimation cannot be performed on missing data by default. Set a strategy for handling missing values if this is intended.")
    
    if isinstance(nan_strategy, str):
        if nan_strategy == "ignore":
            # simply ignore missing data
            scope_data = scope_data[~nan_mask]
        else:
            raise ValueError("Unknown strategy for handling missing (NaN) values for 'Gaussian'.")
    elif isinstance(nan_strategy, Callable):
        scope_data = nan_strategy(scope_data)
    elif nan_strategy is not None:
        raise ValueError(f"Expected 'nan_strategy' to be of type '{type(str)}, or '{Callable}' or '{None}', but was of type {type(nan_strategy)}.")

    # calculate mean and standard deviation from data
    mean_est = scope_data.mean()
    std_est = scope_data.std(ddof=1 if bias_correction else 0)

    # edge case (if all values are the same, not enough samples or very close to each other)
    if np.isclose(std_est, 0.0) or np.isnan(std_est):
        std_est = 1e-8

    # set parameters of leaf node
    leaf.set_params(mean=mean_est, std=std_est)