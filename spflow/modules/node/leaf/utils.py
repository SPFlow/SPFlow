from typing import Callable

from spflow import tensor as T


def apply_nan_strategy(nan_strategy, scope_data, leaf, weights, check_support):
    if weights is None:
        weights = T.ones(scope_data.shape[0], device=leaf.device)
    if weights.ndim != 1 or weights.shape[0] != scope_data.shape[0]:
        raise ValueError(
            "Number of specified weights for maximum-likelihood estimation does not match number of data points."
        )
    # reshape weights
    weights = weights.reshape((-1, 1))
    if check_support:
        if T.any(~leaf.check_support(scope_data, is_scope_data=True)):
            raise ValueError("Encountered values outside of the support.")
    # NaN entries (no information)
    nan_mask = T.isnan(scope_data)
    if T.all(nan_mask):
        raise ValueError("Cannot compute maximum-likelihood estimation on nan-only data.")
    if nan_strategy is None and T.any(nan_mask):
        raise ValueError(
            "Maximum-likelihood estimation cannot be performed on missing data by default. Set a strategy for handling missing values if this is intended."
        )
    if isinstance(nan_strategy, str):
        if nan_strategy == "ignore":
            # simply ignore missing data
            scope_data = scope_data[~nan_mask.squeeze(1)]
            weights = weights[~nan_mask.squeeze(1)]
        else:
            raise ValueError("Unknown strategy for handling missing (NaN) values.")
    elif isinstance(nan_strategy, Callable):
        scope_data = nan_strategy(scope_data)
        # TODO: how to handle weights
    elif nan_strategy is not None:
        raise ValueError(
            f"Expected 'nan_strategy' to be of type '{type(str)}, or '{Callable}' or '{None}', but was of type {type(nan_strategy)}."
        )
    return scope_data, weights
