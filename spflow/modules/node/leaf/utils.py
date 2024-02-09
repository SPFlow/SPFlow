from typing import Callable, Tuple


import torch
from torch import Tensor


def apply_nan_strategy(nan_strategy, scope_data, leaf, weights, check_support) -> tuple[Tensor, Tensor]:
    if weights is None:
        weights = torch.ones(scope_data.shape[0], device=leaf.device)
    if weights.ndim != 1 or weights.shape[0] != scope_data.shape[0]:
        raise ValueError(
            "Number of specified weights for maximum-likelihood estimation does not match number of data points."
        )
    # reshape weights
    weights = weights.reshape((-1, 1))
    if check_support:
        if torch.any(~leaf.distribution.check_support(scope_data)):
            raise ValueError("Encountered values outside of the support.")
    # NaN entries (no information)
    nan_mask = torch.isnan(scope_data)
    if torch.all(nan_mask):
        raise ValueError("Cannot compute maximum-likelihood estimation on nan-only data.")
    if nan_strategy is None and torch.any(nan_mask):
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

    # normalize weights to sum to n_samples
    weights /= weights.sum() / scope_data.shape[0]

    return scope_data, weights


def init_parameter(param: Tensor, event_shape: Tuple[int, ...], init: Callable) -> Tensor:
    """Initializes a parameter tensor of a leaf node."""

    if param is None:
        return init(event_shape)
    else:
        return param
    # else:
    #     if param.ndim == 0:
    #         return param.view(1,)
    #     elif param.ndim == 1:
    #         # Make space for scope and n_out dimensions
    #         return param.view(-1, 1)
    #     elif param.ndim == 2:
    #         return param
    #     else:
    #         raise ValueError(f"Invalid shape for 'param': {param.shape}. Must must be 0D (scalar), 1D or 2D.")
