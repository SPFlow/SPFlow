from __future__ import annotations

from typing import Callable, Any

import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.utils.projections import (
    proj_real_to_bounded,
    proj_bounded_to_real,
    proj_real_to_convex,
    proj_convex_to_real,
)


class LogSpaceParameter:
    """Descriptor for parameters stored in log-space, exposed in real space.

    Transparently stores values as `log_{name}` for numerical stability during optimization.

    Args:
        name: Parameter name. Storage uses 'log_{name}'.
        validator: Validator function (defaults to positive + finite check).
    """

    def __init__(self, name: str, validator: Callable[[Tensor], None] | None = None):
        self.name = name
        self.log_name = f"log_{name}"
        self.validator = validator or self._default_positive_validator

    def _default_positive_validator(self, value: Tensor) -> None:
        """Validate that values are finite and strictly positive."""
        if not torch.isfinite(value).all():
            raise ValueError(f"Values for '{self.name}' must be finite, but was: {value}")
        if torch.any(value <= 0.0):
            raise ValueError(f"Value for '{self.name}' must be greater than 0.0, but was: {value}")

    def __get__(self, obj: Any, objtype: Any = None) -> Tensor:
        """Return parameter value in real space (exp of log-space storage).

        Args:
            obj: Object instance or None for class access.
            objtype: Class type (unused).

        Returns:
            Parameter value in real space.
        """
        if obj is None:
            return self
        # Retrieve the underlying log-space parameter and exponentiate it
        return getattr(obj, self.log_name).exp()

    def __set__(self, obj: Any, value: Tensor) -> None:
        """Validate and store parameter as log(value) for numerical stability.

        Args:
            obj: Object instance to set parameter on.
            value: Parameter value to store (will be converted to log-space).
        """
        param = getattr(obj, self.log_name)
        tensor_value = torch.as_tensor(value, dtype=param.dtype, device=param.device)
        # Validate before storing (ensures only valid values are stored)
        self.validator(tensor_value)
        # Store in log-space for numerical stability during optimization
        param.data = tensor_value.log()


class SimplexParameter:
    """Descriptor for parameters constrained to the probability simplex.

    Transparently stores values as logits (unconstrained real numbers) for
    numerical stability during optimization, exposes them as normalized
    probabilities via softmax.

    Args:
        name: Parameter name. Storage uses 'logits_{name}'.
        validator: Validator function (defaults to probability simplex check).
    """

    def __init__(self, name: str, validator: Callable[[Tensor], None] | None = None):
        self.name = name
        self.logits_name = f"logits_{name}"
        self.validator = validator or self._default_simplex_validator

    def _default_simplex_validator(self, value: Tensor) -> None:
        """Validate that values are finite and form a valid probability distribution."""
        if not torch.isfinite(value).all():
            raise ValueError(f"Values for '{self.name}' must be finite, but was: {value}")
        if torch.any(value < 0.0):
            raise ValueError(f"Values for '{self.name}' must be non-negative, but was: {value}")
        # Check that values sum to 1 (within numerical tolerance)
        sums = value.sum(dim=-1, keepdim=True)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6):
            raise ValueError(f"Values for '{self.name}' must sum to 1, but sums were: {sums.squeeze(-1)}")

    def __get__(self, obj: Any, objtype: Any = None) -> Tensor:
        """Return parameter value as normalized probabilities via softmax.

        Args:
            obj: Object instance or None for class access.
            objtype: Class type (unused).

        Returns:
            Parameter value as probabilities summing to 1 along the last dimension.
        """
        if obj is None:
            return self
        # Retrieve the underlying logits parameter and apply softmax
        logits = getattr(obj, self.logits_name)
        return proj_real_to_convex(logits)

    def __set__(self, obj: Any, value: Tensor) -> None:
        """Validate and store parameter as logits for numerical stability.

        Args:
            obj: Object instance to set parameter on.
            value: Parameter value to store (probabilities, will be converted to logits).
        """
        param = getattr(obj, self.logits_name)
        tensor_value = torch.as_tensor(value, dtype=param.dtype, device=param.device)
        # Validate before storing (ensures only valid probabilities are stored)
        self.validator(tensor_value)
        # Store as logits (unconstrained space) for numerical stability during optimization
        param.data = proj_convex_to_real(tensor_value)


class BoundedParameter:
    """Descriptor for parameters constrained to [lb, ub] via projection.

    Stores values in unbounded space for unconstrained optimization.

    Args:
        name: Parameter name. Storage uses 'log_{name}'.
        lb: Lower bound (inclusive, None = unbounded).
        ub: Upper bound (inclusive, None = unbounded).
        validator: Validator function (defaults to bound check).
    """

    def __init__(
        self,
        name: str,
        lb: float | Tensor | None = None,
        ub: float | Tensor | None = None,
        validator: Callable[[Tensor], None] | None = None,
    ):
        """Initialize a BoundedParameter descriptor.

        Args:
            name: Parameter name. Storage uses 'log_{name}'.
            lb: Lower bound (inclusive, None = unbounded).
            ub: Upper bound (inclusive, None = unbounded).
            validator: Validator function (defaults to bound check).
        """
        self.name = name
        self.log_name = f"log_{name}"
        self.lb = lb
        self.ub = ub
        self.validator = validator or self._make_bounds_validator()

    def _make_bounds_validator(self) -> Callable[[Tensor], None]:
        """Create a validator function that checks bounds and finiteness."""

        def validate(value: Tensor) -> None:
            """Validate parameter value against bounds and finiteness.

            Args:
                value: Parameter value to validate.

            Raises:
                ValueError: If value is not finite or outside bounds.
            """
            if not torch.isfinite(value).all():
                raise ValueError(f"Values for '{self.name}' must be finite, but was: {value}")
            if self.lb is not None:
                lb_tensor = torch.as_tensor(self.lb, dtype=value.dtype, device=value.device)
                if torch.lt(value, lb_tensor).any():
                    raise ValueError(f"Values for '{self.name}' must be >= {self.lb}, but was: {value}")
            if self.ub is not None:
                ub_tensor = torch.as_tensor(self.ub, dtype=value.dtype, device=value.device)
                if torch.gt(value, ub_tensor).any():
                    raise ValueError(f"Values for '{self.name}' must be <= {self.ub}, but was: {value}")

        return validate

    def _project_bounds(self, param: Tensor) -> tuple[Tensor | None, Tensor | None]:
        """Project bounds to parameter tensor device and dtype.

        Args:
            param: Parameter tensor for device/dtype reference.

        Returns:
            Tuple of (lower_bound, upper_bound) as tensors or None.
        """
        lb = torch.as_tensor(self.lb, dtype=param.dtype, device=param.device) if self.lb is not None else None
        ub = torch.as_tensor(self.ub, dtype=param.dtype, device=param.device) if self.ub is not None else None
        return lb, ub

    def __get__(self, obj: Any, objtype: Any = None) -> Tensor:
        """Return parameter value in bounded space via projection.

        Args:
            obj: Object instance or None for class access.
            objtype: Class type (unused).

        Returns:
            Parameter value projected to bounded space.
        """
        if obj is None:
            return self
        # Retrieve the underlying unbounded parameter
        param = getattr(obj, self.log_name)
        # Convert bounds to tensors on correct device/dtype
        lb, ub = self._project_bounds(param)
        # Project from unbounded space to bounded space
        return proj_real_to_bounded(param, lb=lb, ub=ub)

    def __set__(self, obj: Any, value: Tensor) -> None:
        """Validate and store parameter in unbounded space via projection.

        Args:
            obj: Object instance to set parameter on.
            value: Parameter value to store (will be projected to unbounded space).
        """
        param = getattr(obj, self.log_name)
        tensor_value = torch.as_tensor(value, dtype=param.dtype, device=param.device)
        # Validate bounds before storing (ensures only valid values are stored)
        self.validator(tensor_value)
        # Convert bounds to tensors on correct device/dtype
        lb, ub = self._project_bounds(param)
        # Project to unbounded space for storage and optimization
        param.data = proj_bounded_to_real(tensor_value, lb=lb, ub=ub)


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
    if not (all(param is None for param in params) ^ all(param is not None for param in params)):
        raise InvalidParameterCombinationError("Either all parameters or none of them must be given.")

    if all(param is None for param in params):
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
