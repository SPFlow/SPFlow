from __future__ import annotations

from typing import Callable, Any

import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope


def validate_all_or_none(**params: Any) -> bool:
    """Ensure paired parameters are provided together (all or none).

    Args:
        **params: Parameter tensors (or None).

    Returns:
        True if any parameters provided, False if none provided.

    Raises:
        InvalidParameterCombinationError: If some but not all provided.
    """
    provided = [name for name, value in params.items() if value is not None]
    if provided and len(provided) != len(params):
        missing = [name for name in params if name not in provided]
        group = ", ".join(params.keys())
        missing_group = ", ".join(missing)
        provided_group = ", ".join(provided)
        raise InvalidParameterCombinationError(
            f"Parameters ({group}) must be provided together; missing {missing_group} when only "
            f"{provided_group or 'none'} were provided."
        )
    return bool(provided)


def apply_nan_strategy(
    nan_strategy: str | None, scope_data: Tensor, device: torch.device, weights: Tensor | None
) -> tuple[Tensor, Tensor]:
    """Apply NaN handling strategy for MLE.

    Args:
        nan_strategy: None (strict), 'ignore' (drop rows), or callable.
        scope_data: Data tensor.
        device: Device for operations.
        weights: Optional sample weights.

    Returns:
        Processed data and normalized weights.

    Raises:
        ValueError: Invalid inputs or NaN without strategy.
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
            raise ValueError(f"Unknown nan_strategy '{nan_strategy}'. Supported strategies: 'ignore'")

    # Normalize weights to sum to the number of samples
    weights = weights * (scope_data.shape[0] / weights.sum())

    return scope_data, weights


def init_parameter(param: Tensor | None, event_shape: tuple[int, ...], init: Callable) -> Tensor:
    """Initialize parameter tensor (uses init function if None).

    Args:
        param: Parameter tensor or None.
        event_shape: Event shape for initialization.
        init: Initialization function.

    Returns:
        Initialized parameter tensor.

    Raises:
        ValueError: If parameter is not finite.
    """
    if param is None:
        return init(event_shape)
    else:
        if not torch.isfinite(param).all():
            raise ValueError("Parameter must be finite.")
        return param


def parse_leaf_args(
    scope: int | list[int] | Scope, out_channels, num_repetitions, params
) -> tuple[int, int, int]:
    """Parse leaf arguments and return event_shape.

    Args:
        scope: Variable scope (int, list[int], or Scope).
        out_channels: Number of output channels.
        num_repetitions: Number of repetitions.
        params: Distribution parameters.

    Returns:
        Event shape tuple.

    Raises:
        ValueError: If scope type is invalid.
        InvalidParameterCombinationError: If parameter combinations are invalid.
    """
    # We need to accept different types for scope here since parse_leaf_args is called before the LeafModule constructor
    # which turns the scope variable (may be an int or list of ints for convenience) into a Scope object.
    match scope:
        case Scope():
            query_length = len(scope.query)
        case int():
            query_length = 1
        case list():
            query_length = len(scope)
        case _:
            raise ValueError("scope must be of type Scope, int, or list of int.")

    # Either all params are None or no params are None
    if params and not (all(param is None for param in params) ^ all(param is not None for param in params)):
        raise InvalidParameterCombinationError("Either all parameters or none of them must be given.")

    if not params or all(param is None for param in params):
        if out_channels is None:
            raise InvalidParameterCombinationError(
                "Either out_channels or distribution parameters must be given."
            )
        event_shape = (query_length, out_channels)
    else:
        if out_channels is not None:
            if any(param.shape[1] != out_channels for param in params):
                raise InvalidParameterCombinationError(
                    "If out_channels is given, it must match the second dimension of all parameter tensors."
                )
            # raise InvalidParameterCombinationError(
        #     "Either out_channels or distribution parameters must be given."
        # )

        if query_length != params[0].shape[0]:
            raise InvalidParameterCombinationError(
                "The number of scope dimensions must match the number of parameters out_features (first dimension)."
            )

        event_shape = params[0].shape

    if num_repetitions is not None and len(event_shape) == 2:
        event_shape = torch.Size(list(event_shape) + [num_repetitions])

    return event_shape


def _handle_mle_edge_cases(
    param_est: Tensor,
    lb: float | Tensor | None = None,
    ub: float | Tensor | None = None,
) -> Tensor:
    """Handle NaNs, zeros, and bounds in parameter estimation.

    Args:
        param_est: Parameter estimate tensor.
        lb: Lower bound (inclusive).
        ub: Upper bound (inclusive).

    Returns:
        Processed parameter estimate with edge cases handled.
    """
    eps = torch.tensor(1e-8, dtype=param_est.dtype, device=param_est.device)

    def _to_tensor(bound: float | Tensor | None) -> Tensor | None:
        """Convert bound to tensor with proper device and dtype.

        Args:
            bound: Bound value as float, Tensor, or None.

        Returns:
            Bound as tensor with correct device and dtype, or None.
        """
        if bound is None:
            return None
        if isinstance(bound, Tensor):
            return bound.to(device=param_est.device, dtype=param_est.dtype)
        return torch.as_tensor(bound, device=param_est.device, dtype=param_est.dtype)

    lb_tensor = _to_tensor(lb)
    ub_tensor = _to_tensor(ub)

    nan_mask = torch.isnan(param_est)
    if nan_mask.any():
        param_est = param_est.clone()
        param_est[nan_mask] = eps

    if lb_tensor is not None:
        below_mask = param_est <= lb_tensor
        if below_mask.any():
            param_est = param_est.clone()
            param_est[below_mask] = lb_tensor[below_mask] + eps if lb_tensor.ndim else lb_tensor + eps
    else:
        zero_mask = torch.isclose(
            param_est, torch.tensor(0.0, device=param_est.device, dtype=param_est.dtype)
        )
        if zero_mask.any():
            param_est = param_est.clone()
            param_est[zero_mask] = eps

    if ub_tensor is not None:
        above_mask = param_est >= ub_tensor
        if above_mask.any():
            param_est = param_est.clone()
            param_est[above_mask] = ub_tensor[above_mask] - eps if ub_tensor.ndim else ub_tensor - eps

    return param_est


def _prepare_mle_weights(data: Tensor, weights: Tensor | None = None) -> Tensor:
    """Prepare weights for MLE with proper shape for broadcasting.

    Args:
        data: Input data tensor.
        weights: Optional sample weights.

    Returns:
        Weights tensor with proper shape for broadcasting.
    """
    if weights is None:
        _shape = (data.shape[0], *([1] * (data.dim() - 1)))
        weights = torch.ones(_shape, device=data.device)
    elif weights.dim() == 1 and data.dim() > 1:
        # Reshape 1D weights to broadcast properly with multi-dimensional data
        # e.g., weights [batch_size] -> [batch_size, 1, 1, ...]
        _shape = (weights.shape[0], *([1] * (data.dim() - 1)))
        weights = weights.view(_shape)
    return weights
