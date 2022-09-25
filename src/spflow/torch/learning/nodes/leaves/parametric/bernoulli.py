"""
Created on September 23, 2022

@authors: Philipp Deibert
"""
from typing import Optional, Union, Callable
import torch
from spflow.meta.dispatch.dispatch import dispatch
from spflow.torch.structure.nodes.leaves.parametric.bernoulli import Bernoulli


@dispatch(memoize=True) # TODO: swappable
def maximum_likelihood_estimation(leaf: Bernoulli, data: torch.Tensor, bias_correction: bool=True, nan_strategy: Optional[Union[str, Callable]]=None) -> None:
    """TODO."""

    # select relevant data for scope
    scope_data = data[:, leaf.scope.query]

    if torch.any(~leaf.check_support(scope_data)):
        raise ValueError("Encountered values outside of the support for 'Bernoulli'.")

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
            raise ValueError("Unknown strategy for handling missing (NaN) values for 'Bernoulli'.")
    elif isinstance(nan_strategy, Callable):
        scope_data = nan_strategy(scope_data)
    elif nan_strategy is not None:
        raise ValueError(f"Expected 'nan_strategy' to be of type '{type(str)}, or '{Callable}' or '{None}', but was of type {type(nan_strategy)}.")

    # total number of instances
    n_total = scope_data.shape[0]

    # count number of total successes
    n_success = scope_data.sum()

    p_est = n_success/n_total

    # edge case: if prob. 1 (or 0), set to smaller (or larger) value
    if torch.isclose(p_est, torch.tensor(0.0)):
        p_est = 1e-8
    elif torch.isclose(p_est, torch.tensor(1.0)):
        p_est = 1 - 1e-8
    
    # set parameters of leaf node
    leaf.set_params(p=p_est)