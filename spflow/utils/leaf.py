from typing import Tuple
from collections.abc import Callable

import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterCombinationError


def apply_nan_strategy(
    nan_strategy: str | None, scope_data: Tensor, device: torch.device, weights: Tensor | None
) -> tuple[Tensor, Tensor]:
    """Apply a strategy for handling NaN values in data for maximum-likelihood estimation.

    Args:
        nan_strategy: Strategy for handling missing (NaN) values. Currently supports:
            - None: Raise an error if NaN values are present (default, strict mode)
            - "ignore": Remove all rows containing any NaN values
        scope_data: Data tensor of shape (batch_size, num_features)
        device: Device to use for tensor operations
        weights: Optional sample weights of shape (batch_size,) or (batch_size, 1).
                 If None, uniform weights are used.

    Returns:
        Tuple of (processed_data, normalized_weights) where:
            - processed_data: Data with NaN strategy applied
            - normalized_weights: Weights normalized to sum to the number of samples

    Raises:
        ValueError: If weights shape doesn't match data, if all data is NaN,
                   if NaN values are present without a strategy, or if an unknown strategy is given.
    """
    # Initialize weights if not provided
    if weights is None:
        weights = torch.ones(scope_data.shape[0], device=device)

    # Validate weights shape
    if weights.ndim != 1 or weights.shape[0] != scope_data.shape[0]:
        raise ValueError(
            f"Weights shape {weights.shape} does not match number of data points {scope_data.shape[0]}. "
            f"Expected shape ({scope_data.shape[0]},) for maximum-likelihood estimation."
        )

    # Reshape weights to column vector for broadcasting
    weights = weights.reshape((-1, 1))

    # Check for NaN entries
    nan_mask = torch.isnan(scope_data)

    if torch.all(nan_mask):
        raise ValueError("Cannot compute maximum-likelihood estimation: all data is NaN.")

    if torch.any(nan_mask):
        if nan_strategy is None:
            raise ValueError(
                "Cannot perform maximum-likelihood estimation on data with missing (NaN) values. "
                "Set 'nan_strategy' parameter to specify how to handle missing values (e.g., 'ignore')."
            )

        if nan_strategy == "ignore":
            # Remove all rows containing any NaN values
            valid_rows = ~nan_mask.any(dim=1)
            scope_data = scope_data[valid_rows]
            weights = weights[valid_rows]
        else:
            raise ValueError(
                f"Unknown nan_strategy '{nan_strategy}'. Supported strategies: 'ignore'"
            )

    # Normalize weights to sum to the number of samples
    weights = weights * (scope_data.shape[0] / weights.sum())

    return scope_data, weights


def init_parameter(param: Tensor | None, event_shape: tuple[int, ...], init: Callable) -> Tensor:
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
        event_shape = (len(scope.query), out_channels)
    else:
        if out_channels is not None:
            if any(param.shape[1] != out_channels for param in params):
                raise InvalidParameterCombinationError(
                    "If out_channels is given, it must match the second dimension of all parameter tensors."
                )
            # raise InvalidParameterCombinationError(
        #     "Either out_channels or distribution parameters must be given."
        # )

        if len(scope.query) != params[0].shape[0]:
            raise InvalidParameterCombinationError(
                "The number of scope dimensions must match the number of parameters out_features (first dimension)."
            )

        event_shape = params[0].shape

    if num_repetitions is not None and len(event_shape) == 2:
        event_shape = torch.Size(list(event_shape) + [num_repetitions])

    return event_shape
