from typing import Tuple
from collections.abc import Callable

import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterCombinationError


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
        # create dimension for check_support
        scope_data = scope_data.unsqueeze(2)
        if torch.any(~leaf.distribution.check_support(scope_data)):
            raise ValueError("Encountered values outside of the support.")
        scope_data = scope_data.squeeze(2)
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
        if not torch.isfinite(param).all():
            raise ValueError("Parameter must be finite.")
        return param


def parse_leaf_args(scope, out_channels, num_repetitions, params) -> tuple[int, int, int]:
    """
    Parse the arguments of a leaf node and return the event_shape.

    Args:
        scope: Leaf scope.
        out_channels: Number of output channels for the leaf module.
        params: List of parameters of the leaf distribution.

    Returns:
        event_shape: Shape of the event space.
    """

    # Either all params are None or no params are None
    if not (all(param is None for param in params) ^ all(param is not None for param in params)):
        raise InvalidParameterCombinationError("Either all parameters or none of them must be given.")

    if all(param is None for param in params):
        if out_channels is None:
            raise InvalidParameterCombinationError(
                "Either out_channels or distribution parameters must be given."
            )
        if num_repetitions is None:
            event_shape = (len(scope.query), out_channels)
        else:
            event_shape = (len(scope.query), out_channels, num_repetitions)
    else:
        if out_channels is not None:
            raise InvalidParameterCombinationError(
                "Either out_channels or distribution parameters must be given."
            )

        if len(scope.query) != params[0].shape[0]:
            raise InvalidParameterCombinationError(
                "The number of scope dimensions must match the number of parameters out_features (first dimension)."
            )

        event_shape = params[0].shape
    return event_shape
