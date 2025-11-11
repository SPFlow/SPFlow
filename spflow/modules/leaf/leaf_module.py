from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from torch import Tensor, nn
from spflow.utils.projections import proj_real_to_bounded, proj_bounded_to_real

from spflow.meta.data.scope import Scope
from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow.modules.module import Module
from spflow.utils.cache import Cache, init_cache
from spflow.utils.leaf import apply_nan_strategy
import time
from spflow.exceptions import InvalidParameterCombinationError


class LogSpaceParameter:
    """Descriptor for parameters stored in log-space but exposed in real space.

    This descriptor pattern eliminates repeated property boilerplate for log-space parameters
    (e.g., std, rate, alpha, beta) across distributions. The actual parameter is stored internally
    as `log_{name}` for numerical stability, while the descriptor exposes it in real space via
    exponential transformation.

    Example:
        >>> class Normal(LeafModule):
        ...     std = LogSpaceParameter('std')
        ...     def __init__(self, scope, std=None):
        ...         super().__init__(scope)
        ...         self.log_std = nn.Parameter(std.log() if std is not None else ...)
        ...
        >>> normal = Normal(scope, std=torch.tensor(1.0))
        >>> normal.std  # Returns exp(log_std)
        >>> normal.std = torch.tensor(2.0)  # Validates and stores as log_std = ln(2.0)

    Args:
        name: Name of the parameter (e.g., 'std', 'rate'). The actual storage uses 'log_{name}'.
        validator: Optional custom validator function. Defaults to _default_positive_validator
                  which checks finiteness and positivity.
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
        """Get the parameter in real space (exponential of log-space storage)."""
        if obj is None:
            return self
        return getattr(obj, self.log_name).exp()

    def __set__(self, obj: Any, value: Tensor) -> None:
        """Set the parameter by storing its logarithm in log-space.

        Args:
            obj: The instance owning this descriptor.
            value: The parameter value in real space to store.
        """
        param = getattr(obj, self.log_name)
        tensor_value = torch.as_tensor(value, dtype=param.dtype, device=param.device)
        self.validator(tensor_value)
        param.data = tensor_value.log()


class BoundedParameter:
    """Descriptor for parameters constrained to [lb, ub], stored via projections.

    This descriptor pattern eliminates repeated property boilerplate for bounded parameters
    (e.g., probability p in [0,1]). The actual parameter is stored in an auxiliary unbounded space
    using projection functions, allowing unconstrained optimization while enforcing bounds when
    accessed.

    The descriptor uses `proj_real_to_bounded` and `proj_bounded_to_real` from spflow.utils.projections
    to map between unbounded and bounded representations, providing numerical stability for parameters
    that must stay within strict bounds.

    Example:
        >>> class Binomial(LeafModule):
        ...     p = BoundedParameter('p', lb=0.0, ub=1.0)
        ...     def __init__(self, scope, p=None):
        ...         super().__init__(scope)
        ...         if p is not None:
        ...             self.log_p = nn.Parameter(proj_bounded_to_real(p, lb=0.0, ub=1.0))
        ...
        >>> binomial = Binomial(scope, p=torch.tensor(0.5))
        >>> binomial.p  # Returns proj_real_to_bounded(log_p, lb=0.0, ub=1.0)
        >>> binomial.p = torch.tensor(0.3)  # Validates and stores as log_p = proj_bounded_to_real(...)

    Args:
        name: Name of the parameter (e.g., 'p'). Storage uses 'log_{name}'.
        lb: Lower bound (inclusive). None means unbounded below. Default is None.
        ub: Upper bound (inclusive). None means unbounded above. Default is None.
        validator: Optional custom validator function. Defaults to _make_bounds_validator()
                  which checks finiteness and bounds.
    """

    def __init__(
        self,
        name: str,
        lb: float | Tensor | None = None,
        ub: float | Tensor | None = None,
        validator: Callable[[Tensor], None] | None = None,
    ):
        self.name = name
        self.log_name = f"log_{name}"
        self.lb = lb
        self.ub = ub
        self.validator = validator or self._make_bounds_validator()

    def _make_bounds_validator(self) -> Callable[[Tensor], None]:
        """Create a validator function that checks bounds and finiteness."""

        def validate(value: Tensor) -> None:
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
        lb = torch.as_tensor(self.lb, dtype=param.dtype, device=param.device) if self.lb is not None else None
        ub = torch.as_tensor(self.ub, dtype=param.dtype, device=param.device) if self.ub is not None else None
        return lb, ub

    def __get__(self, obj: Any, objtype: Any = None) -> Tensor:
        """Get the parameter in bounded space (via projection from unbounded storage)."""
        if obj is None:
            return self
        param = getattr(obj, self.log_name)
        lb, ub = self._project_bounds(param)
        return proj_real_to_bounded(param, lb=lb, ub=ub)

    def __set__(self, obj: Any, value: Tensor) -> None:
        """Set the parameter by projecting from bounded to unbounded space.

        Args:
            obj: The instance owning this descriptor.
            value: The parameter value in bounded space to store.
        """
        param = getattr(obj, self.log_name)
        tensor_value = torch.as_tensor(value, dtype=param.dtype, device=param.device)
        self.validator(tensor_value)
        lb, ub = self._project_bounds(param)
        param.data = proj_bounded_to_real(tensor_value, lb=lb, ub=ub)


def validate_all_or_none(**params: Any) -> bool:
    """Ensure paired parameters are provided together (all or none).

    This utility function eliminates repeated validation logic in distributions that require
    pairs of parameters (e.g., mean+std in Normal, alpha+beta in Gamma).

    Raises InvalidParameterCombinationError if some but not all parameters are provided.

    Args:
        **params: Keyword arguments where names are parameter names and values are the
                 actual parameter tensors (or None if not provided).

    Returns:
        bool: True if any parameters were provided (all of them), False if none were provided.

    Raises:
        InvalidParameterCombinationError: If some parameters are provided but not all.

    Example:
        >>> validate_all_or_none(mean=mean_val, std=std_val)  # OK: both provided
        True
        >>> validate_all_or_none(mean=None, std=None)  # OK: neither provided
        False
        >>> validate_all_or_none(mean=mean_val, std=None)  # ERROR: only mean provided
        InvalidParameterCombinationError
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


@dataclass
class MLEBatch:
    """Container with normalized data shared across MLE template hooks.

    This dataclass encapsulates all the normalized inputs and metadata needed by the
    MLE template pattern. It is created by _init_mle_batch and passed to _mle_compute_statistics.

    Attributes:
        data: Scope-filtered data tensor of shape (batch_size, num_scope_features).
              Missing values have been handled according to nan_strategy.
        weights: Normalized weight tensor of shape (batch_size, 1, ...) with proper
                broadcasting shape for element-wise multiplication with data.
        bias_correction: Boolean flag indicating whether bias correction was requested.
        cache: Dictionary for storing intermediate computation results for reuse in EM steps.
        diagnostics: Metadata dict tracking original/retained sample counts, nan_strategy used,
                    and other debugging information.
    """

    data: Tensor
    weights: Tensor
    bias_correction: bool
    cache: Cache
    diagnostics: dict[str, Any]


@dataclass
class MLEParameterEstimate:
    """Metadata wrapper for per-parameter MLE updates.

    When returning from _mle_compute_statistics, individual parameter estimates can be
    wrapped in this dataclass to provide additional metadata (bounds, broadcast requirements)
    that _apply_mle_estimates will respect. Alternatively, plain tensors can be returned
    and will be treated with default metadata.

    Example:
        >>> return {
        ...     'std': LogSpaceParameter.estimate(tensor_value),
        ...     'mean': tensor_value,  # Will use defaults: broadcast=True, no bounds
        ... }

    Attributes:
        value: The estimated parameter tensor.
        lb: Optional lower bound. If provided, _apply_mle_estimates will clamp the value.
        ub: Optional upper bound. If provided, _apply_mle_estimates will clamp the value.
        broadcast: Whether to broadcast this parameter to event_shape. Defaults to True.
                  Set to False if the estimate already has the correct shape.
    """

    value: Tensor
    lb: float | Tensor | None = None
    ub: float | Tensor | None = None
    broadcast: bool = True


MLEEstimates = dict[str, Tensor | MLEParameterEstimate]


class LeafModule(Module, ABC):
    def __init__(self, scope: Scope | list[int], out_channels: int = None):
        r"""Base class for leaf modules in the SPFlow framework.

        Args:
            scope: Scope object or list of ints specifying the scope of the distribution.
            out_channels: Number of output channels.
        """
        super().__init__()

        # Convert list to Scope object
        if isinstance(scope, list):
            scope = Scope(scope)

        self.scope = scope.copy()
        self._out_channels = out_channels
        self._event_shape = None  # Will be set by subclasses

    @property
    @abstractmethod
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying torch distribution object."""
        pass

    @property
    @abstractmethod
    def _supported_value(self):
        """Returns the supported values of the distribution."""
        pass

    @abstractmethod
    def params(self) -> Dict[str, Tensor]:
        """Returns the parameters of the distribution."""
        pass

    @abstractmethod
    def _mle_compute_statistics(self, batch: MLEBatch) -> MLEEstimates:
        """Compute distribution-specific statistics required for MLE updates."""
        raise NotImplementedError

    def log_prob(self, x: Tensor) -> Tensor:
        """Computes the log probability of the given samples."""
        return self.distribution.log_prob(x)

    def mode(self) -> Tensor:
        """Returns the mode of the distribution."""
        return self.distribution.mode

    def marginalized_params(self, indices: list[int]) -> Dict[str, Tensor]:
        """Returns the marginalized parameters of the distribution.

        Args:
            indices:
                List of integers specifying the indices of the module to keep.

        Returns:
            Dictionary from parameter name to tensor containing the marginalized parameters.
        """
        return {k: v[indices] for k, v in self.params().items()}

    def check_support(self, data: Tensor) -> Tensor:
        r"""Checks whether ``data`` falls inside the support of this distribution.

        The base implementation consults the underlying torch distribution support
        (if available) and then applies optional subclass-specific masks via
        ``_custom_support_mask``. NaNs are treated as valid (they are marginalized),
        while ``+/-inf`` entries are always rejected.

        Args:
            data: Tensor that has already been reshaped for the leaf scope (see
                ``log_likelihood`` / ``apply_nan_strategy``). Additional leaf-specific
                dimensions (channels, repetitions) may be size ``1`` because all
                repetitions share the same support.

        Returns:
            Boolean tensor (same shape as ``data``) indicating per-entry support
            membership.
        """

        nan_mask = torch.isnan(data)
        valid = torch.ones_like(data, dtype=torch.bool)

        support_masks: list[Tensor] = []

        if self._use_distribution_support():
            dist_mask = self._distribution_support_mask(data)
            if dist_mask is not None:
                support_masks.append(dist_mask)

        custom_mask = self._custom_support_mask(data)
        if custom_mask is not None:
            support_masks.append(custom_mask)

        for mask in support_masks:
            aligned = self._align_support_mask(mask, data)
            valid[~nan_mask] &= aligned[~nan_mask]

        valid[~nan_mask & valid] &= ~data[~nan_mask & valid].isinf()

        return valid

    def _use_distribution_support(self) -> bool:
        """Whether ``check_support`` should consult ``self.distribution.support``."""
        return True

    def _custom_support_mask(self, data: Tensor) -> Tensor | None:
        """Hook for subclasses that need bespoke support checks."""
        return None

    def _distribution_support_mask(self, data: Tensor) -> Tensor | None:
        """Return the distribution-provided support mask if available."""
        distribution = self.distribution
        support = getattr(distribution, "support", None)
        if support is None or not hasattr(support, "check"):
            return None
        try:
            return support.check(data)
        except NotImplementedError:
            # Custom distributions may not implement support checks.
            return None

    def _align_support_mask(self, mask: Tensor, data: Tensor) -> Tensor:
        """Align a support mask with the shape of ``data`` for boolean indexing."""
        if mask.dim() < data.dim():
            expand_dims = data.dim() - mask.dim()
            mask = mask.reshape(*mask.shape, *([1] * expand_dims))

        if mask.dim() != data.dim():
            raise RuntimeError(
                f"Support mask rank {mask.dim()} incompatible with data rank {data.dim()} "
                f"in {self.__class__.__name__}. Provide a custom check_support override."
            )

        slices: list[slice] = []
        for mask_size, data_size in zip(mask.shape, data.shape):
            if mask_size == data_size:
                slices.append(slice(None))
            elif data_size == 1 and mask_size > 1:
                slices.append(slice(0, 1))
            elif mask_size == 1 and data_size > 1:
                slices.append(slice(None))
            else:
                raise RuntimeError(
                    f"Support mask shape {tuple(mask.shape)} incompatible with data shape "
                    f"{tuple(data.shape)} in {self.__class__.__name__}."
                )

        mask = mask[tuple(slices)]
        if mask.shape != data.shape:
            mask = mask.expand_as(data)
        return mask

    # MLE Helper Methods
    def _prepare_mle_weights(self, data: Tensor, weights: Optional[Tensor] = None) -> Tensor:
        """Prepare weights for MLE, ensuring proper shape and device.

        Args:
            data: The input data tensor.
            weights: Optional weights tensor. If None, uniform weights are created.

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

    def _handle_mle_edge_cases(
        self,
        param_est: Tensor,
        lb: float | Tensor | None = None,
        ub: float | Tensor | None = None,
    ) -> Tensor:
        """Handle NaNs, zeros, and optional bounds in parameter estimation.

        Args:
            param_est: The estimated parameter tensor.
            lb: Optional lower bound (inclusive). Values at/below this bound are nudged by `eps`.
            ub: Optional upper bound (inclusive). Values at/above this bound are nudged by `eps`.

        Returns:
            Parameter tensor with edge cases handled.
        """
        eps = torch.tensor(1e-8, dtype=param_est.dtype, device=param_est.device)

        def _to_tensor(bound: float | Tensor | None) -> Tensor | None:
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

    def _broadcast_to_event_shape(self, param_est: Tensor) -> Tensor:
        """Broadcast parameter estimate to match event_shape.

        Handles broadcasting for 2D (features, channels) and 3D (features, channels, repetitions) event shapes.

        Args:
            param_est: The estimated parameter tensor with shape (features,).

        Returns:
            Parameter tensor broadcast to proper event_shape.
        """
        if len(self.event_shape) == 2:
            param_est = param_est.unsqueeze(1).repeat(
                1,
                self.out_channels,
                *([1] * (param_est.dim() - 1)),
            )
        elif len(self.event_shape) == 3:
            param_est = (
                param_est.unsqueeze(1)
                .unsqueeze(1)
                .repeat(
                    1,
                    self.out_channels,
                    self.num_repetitions,
                    *([1] * (param_est.dim() - 1)),
                )
            )

        return param_est

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Returns the event shape stored by the leaf module."""
        if self._event_shape is None:
            raise RuntimeError(f"{self.__class__.__name__} has not set _event_shape in __init__")
        return self._event_shape

    @property
    def out_features(self) -> int:
        return len(self.scope.query)

    @property
    def num_repetitions(self) -> int:
        """Returns the number of repetitions of the distribution."""
        if len(self.event_shape) == 3:
            return self.event_shape[2]
        else:
            return None

    @property
    def out_channels(self) -> int:
        """Returns the number of output channels of the distribution."""
        if len(self.event_shape) == 1:
            return 1
        else:
            return self.event_shape[1]

    @property
    def feature_to_scope(self) -> list[Scope]:
        """Returns a list of scopes corresponding to the features in the leaf module."""
        return [Scope([i]) for i in self.scope.query]

    @property
    def device(self) -> torch.device:
        """Get the device of the module.

        Returns the device of the first parameter if available, otherwise the first buffer.
        This is necessary for modules that may have no learnable parameters (e.g., Uniform,
        Hypergeometric).

        Returns:
            torch.device: The device (CPU, CUDA, etc.) where the module's parameters/buffers reside.
        """
        try:
            return next(iter(self.parameters())).device
        except StopIteration:
            return next(iter(self.buffers())).device

    def expectation_maximization(
        self,
        data: torch.Tensor,
        check_support: bool = True,
        cache: Cache | None = None,
    ) -> None:
        """Performs a single expectation maximizaton (EM) step for the leaf module.

        Args:
            data:
                Two-dimensional PyTorch tensor containing the input data.
                Each row corresponds to a sample.
            check_support:
                Boolean value indicating whether or not if the data is in the support of the leaf distributions.
                Defaults to True.
            cache:
                Optional cache dictionary for intermediate results.
        """
        # initialize cache
        cache = init_cache(cache)

        with torch.no_grad():
            # ----- expectation step -----

            # get cached log-likelihood gradients w.r.t. module log-likelihoods
            expectations = cache["log_likelihood"][self].grad
            # normalize expectations for better numerical stability
            # Reduce expectations to shape [batch_size, 1]
            dims = list(range(1, len(expectations.shape)))
            expectations = expectations.sum(dims)
            expectations /= expectations.sum(dim=None, keepdim=True)

            # ----- maximization step -----

            # update parameters through maximum weighted likelihood estimation
            self.maximum_likelihood_estimation(
                data,
                weights=expectations,
                bias_correction=False,
                check_support=check_support,
                cache=cache,
            )

        # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation',
        # we do not need to zero/None parameter gradients

    def log_likelihood(
        self,
        data: Tensor,
        check_support: bool = True,
        cache: Cache | None = None,
    ) -> Tensor:
        r"""Computes log-likelihoods for the leaf module given the data.

        Missing values (i.e., NaN) are marginalized over.

        Args:
            data:
                Two-dimensional PyTorch tensor containing the input data.
                Each row corresponds to a sample.
            check_support:
                Boolean value indicating whether or not if the data is in the support of the distribution.
                Defaults to True.
            cache:
                Optional cache dictionary.

        Returns:
            Two-dimensional PyTorch tensor containing the log-likelihoods of the input data for the sum node.
            Each row corresponds to an input sample.

        Raises:
            ValueError: Data outside of support.
        """
        # initialize cache
        cache = init_cache(cache)

        # get information relevant for the scope
        data = data[:, self.scope.query]
        if self.event_shape[0] != len(self.scope.query):
            raise RuntimeError(
                f"event_shape mismatch for {self.__class__.__name__}: event_shape={self.event_shape}, scope_len={len(self.scope.query)}"
            )

        # ----- marginalization -----
        marg_mask = torch.isnan(data)
        has_marginalizations = marg_mask.any()

        # If there are any marg_ids, set them to 0.0 to ensure that log_prob call is succesfull
        # and doesn't throw errors due to NaNs
        if has_marginalizations:
            data[marg_mask] = self._supported_value

        # ----- log probabilities -----

        # Unsqueeze scope_data to make space for num_nodes and repetition dimension
        data = data.unsqueeze(2)

        # Use self.event_shape (not self.distribution.event_shape which may be torch's event_shape)
        if len(self.event_shape) > 2:
            data = data.unsqueeze(-1)

        dist = self.distribution

        if check_support:
            # create mask based on distribution's support
            valid_mask = self.check_support(data)

            if not torch.all(valid_mask):
                raise ValueError(
                    f"Encountered data instances that are not in the support of the distribution."
                )

        # compute probabilities for values inside distribution support
        expected_shape = dist.batch_shape + dist.event_shape
        log_prob_input = data
        if expected_shape:
            target_shape = (data.shape[0],) + expected_shape
            try:
                log_prob_input = torch.broadcast_to(data, target_shape)
            except RuntimeError as err:
                raise RuntimeError(
                    f"Could not broadcast data for {self.__class__.__name__} to match "
                    f"distribution shape (batch_shape={dist.batch_shape}, event_shape={dist.event_shape}). "
                    f"data_shape={tuple(data.shape)}"
                ) from err

        log_prob = dist.log_prob(log_prob_input.to(torch.get_default_dtype()))

        # Marginalize entries - broadcast mask to log_prob shape
        if has_marginalizations:
            # Expand marg_mask to match log_prob shape by broadcasting
            # marg_mask is [batch, features], unsqueeze(2) makes it [batch, features, 1]
            marg_mask_for_log_prob = marg_mask.unsqueeze(2)  # [batch, features, 1]
            # For higher-dimensional event shapes, add another dimension
            if len(self.event_shape) > 2:
                marg_mask_for_log_prob = marg_mask_for_log_prob.unsqueeze(-1)  # [batch, features, 1, 1]
            # Broadcast to log_prob shape
            marg_mask_for_log_prob = torch.broadcast_to(marg_mask_for_log_prob, log_prob.shape)
            log_prob[marg_mask_for_log_prob] = 0.0

        # Set marginalized scope data back to NaNs
        if has_marginalizations:
            marg_mask_for_data = marg_mask.unsqueeze(2)
            if len(self.event_shape) > 2:
                marg_mask_for_data = marg_mask_for_data.unsqueeze(-1)
            data[marg_mask_for_data] = torch.nan

        # Cache the result for EM step
        if "log_likelihood" not in cache:
            cache["log_likelihood"] = {}
        cache["log_likelihood"][self] = log_prob

        return log_prob

    def _init_mle_batch(
        self,
        data: Tensor,
        weights: Optional[Tensor] = None,
        bias_correction: bool = True,
        nan_strategy: Optional[str | Callable] = None,
        check_support: bool = True,
        cache: Cache | None = None,
        preprocess_data: bool = True,
    ) -> MLEBatch:
        """Prepare normalized tensors and bookkeeping for the MLE template method.

        This is the first step of the MLE template pattern. It handles:
        - Selecting relevant features from the full data using the leaf's scope
        - Applying NaN strategies (e.g., dropping or imputing missing values)
        - Checking data support if requested
        - Normalizing weights for numerical stability
        - Creating diagnostics for debugging

        Args:
            data: Input data tensor of shape (batch_size, num_features).
            weights: Optional weight tensor for weighted MLE. Shape (batch_size,) or (batch_size, 1).
                    If None, uniform weights are used.
            bias_correction: Whether to apply bias correction in parameter estimation.
            nan_strategy: How to handle missing values (NaN). Options: 'ignore' or callable.
            check_support: Whether to validate that data falls in the distribution's support.
            cache: Optional cache dictionary for intermediate results.
            preprocess_data: Whether to select scope-relevant features. Set False if data
                           is already scope-filtered.

        Returns:
            MLEBatch: Container with normalized data, weights, and metadata for _mle_compute_statistics.
        """
        cache = init_cache(cache)
        scoped_data = data[:, self.scope.query] if preprocess_data else data
        # Apply NaN strategy + support check up-front to stay consistent across leaves.
        scoped_data, normalized_weights = apply_nan_strategy(
            nan_strategy, scoped_data, self, weights, check_support
        )
        # Convert weights back to 1D before broadcast prep.
        normalized_weights_flat = normalized_weights.squeeze(-1)
        mle_weights = self._prepare_mle_weights(scoped_data, normalized_weights_flat)
        diagnostics = {
            "original_samples": data.shape[0],
            "retained_samples": scoped_data.shape[0],
            "nan_strategy": nan_strategy,
            "check_support": check_support,
        }
        return MLEBatch(
            data=scoped_data,
            weights=mle_weights,
            bias_correction=bias_correction,
            cache=cache,
            diagnostics=diagnostics,
        )

    def _apply_mle_estimates(self, estimates: Optional[MLEEstimates]) -> dict[str, Tensor]:
        """Apply template estimates by broadcasting, clamping, and assigning via descriptors.

        This is the final step of the MLE template pattern. It handles:
        - Broadcasting parameter estimates to match the leaf's event_shape (for multi-channel/
          multi-repetition distributions)
        - Clamping values within optional bounds (lower/upper limits)
        - Assigning through descriptors (LogSpaceParameter, BoundedParameter) which handle
          projection and validation automatically
        - Falling back to direct parameter or attribute assignment if no descriptor is present

        The method respects descriptor-based storage patterns, allowing parameters to be
        kept in auxiliary spaces (log-space, projection space) while appearing in real space.

        Args:
            estimates: Dictionary mapping parameter names to estimated values. Values can be
                      either Tensor directly or MLEParameterEstimate instances that contain
                      metadata (bounds, broadcast flags).

        Returns:
            dict[str, Tensor]: Mapping of parameter names to final applied values (in real space,
                              after broadcasting and clamping).

        Example:
            >>> batch = leaf._init_mle_batch(data, weights)
            >>> estimates = leaf._mle_compute_statistics(batch)
            >>> applied = leaf._apply_mle_estimates(estimates)
            >>> applied  # {'std': tensor(...), 'mean': tensor(...)}
        """
        applied: dict[str, Tensor] = {}
        if not estimates:
            return applied
        for name, estimate in estimates.items():
            if isinstance(estimate, MLEParameterEstimate):
                value = estimate.value
                lb = estimate.lb
                ub = estimate.ub
                broadcast = estimate.broadcast
            else:
                value = estimate
                lb = None
                ub = None
                broadcast = True

            param_value = value
            if broadcast:
                param_value = self._broadcast_to_event_shape(param_value)
            if lb is not None or ub is not None:
                param_value = self._handle_mle_edge_cases(param_value, lb=lb, ub=ub)
            class_attr = getattr(type(self), name, None)
            if hasattr(class_attr, "__set__"):
                # Parameter is managed by a descriptor (e.g., LogSpaceParameter, BoundedParameter)
                # The descriptor will handle projection and validation
                class_attr.__set__(self, param_value)
            else:
                current_attr = getattr(self, name, None)
                if isinstance(current_attr, nn.Parameter):
                    current_attr.data = param_value
                else:
                    setattr(self, name, param_value)
            applied[name] = param_value
        return applied

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Optional[Tensor] = None,
        bias_correction: bool = True,
        nan_strategy: Optional[str | Callable] = None,
        check_support: bool = True,
        cache: Cache | None = None,
        preprocess_data: bool = True,
    ) -> None:
        r"""Maximum (weighted) likelihood estimation (MLE) of the leaf module.

        Weights are normalized to sum up to :math:`N`.

        Args:
            data:
                Two-dimensional PyTorch tensor containing the input data.
                Each row corresponds to a sample.
            weights:
                Optional one-dimensional PyTorch tensor containing non-negative weights for all data samples.
                Must match number of samples in ``data``.
                Defaults to None in which case all weights are initialized to ones.
            bias_correction:
                Boolean indicating whether or not to correct possible biases.
                Defaults to True.
            nan_strategy:
                Optional string or callable specifying how to handle missing data.
                If 'ignore', missing values (i.e., NaN entries) are ignored.
                If a callable, it is called using ``data`` and should return another PyTorch tensor of same size.
                Defaults to None.
            check_support:
                Boolean value indicating whether or not if the data is in the support of the leaf distributions.
                Defaults to True.
            cache:
                Optional cache dictionary.
            preprocess_data:
                Boolean indicating whether to select relevant data for scope.
                Defaults to True.

        Raises:
            ValueError: Invalid arguments.
        """
        batch = self._init_mle_batch(
            data=data,
            weights=weights,
            bias_correction=bias_correction,
            nan_strategy=nan_strategy,
            check_support=check_support,
            cache=cache,
            preprocess_data=preprocess_data,
        )
        estimates = self._mle_compute_statistics(batch)
        applied = self._apply_mle_estimates(estimates)
        diagnostics = {
            **batch.diagnostics,
            "updated_parameters": tuple(applied.keys()),
            "weights_sum": float(batch.weights.sum().item()),
        }
        return diagnostics

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        check_support: bool = True,
        cache: Cache | None = None,
        sampling_ctx: Optional[SamplingContext] = None,
    ) -> Tensor:
        r"""Samples from the leaf node given potential evidence.

        Samples missing values proportionally to its probability distribution function (PDF).

        Args:
            num_samples:
                Number of samples to generate.
            data:
                Two-dimensional PyTorch tensor containing potential evidence.
                Each row corresponds to a sample.
            is_mpe:
                Boolean value indicating whether to perform maximum a posteriori estimation (MPE).
                Defaults to False.
            check_support:
                Boolean value indicating whether if the data is in the support of the leaf distributions.
                Defaults to True.
            cache:
                Optional cache dictionary.
            sampling_ctx:
                Optional sampling context containing the instances (i.e., rows) of ``data`` to fill with sampled
                values and the output indices of the node to sample from.

        Returns:
            Two-dimensional PyTorch tensor containing the sampled values together with the specified evidence.
            Each row corresponds to a sample.
        """
        # Prepare data tensor
        data = self._prepare_sample_data(num_samples, data)

        cache = init_cache(cache)
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

        out_of_scope = list(filter(lambda x: x not in self.scope.query, range(data.shape[1])))
        marg_mask = torch.isnan(data)
        marg_mask[:, out_of_scope] = False

        # Mask that tells us which feature at which sample is relevant and should be sampled
        samples_mask = marg_mask
        samples_mask[:, self.scope.query] &= sampling_ctx.mask

        # Count number of samples to draw
        instance_mask = samples_mask.sum(1) > 0
        n_samples = instance_mask.sum()  # count number of rows which have at least one true value

        if is_mpe:
            # Get mode of distribution as MPE
            samples = self.mode().unsqueeze(0)
            if sampling_ctx.repetition_idx is not None and samples.ndim == 4:
                samples = samples.repeat(n_samples, 1, 1, 1).detach()
                # repetition_idx shape: (n_samples,)
                repetition_idx = sampling_ctx.repetition_idx[instance_mask]

                indices = repetition_idx.view(-1, 1, 1, 1).expand(-1, samples.shape[1], samples.shape[2], -1)

                # Gather samples according to repetition index
                samples = torch.gather(samples, dim=-1, index=indices).squeeze(-1)

            elif (
                sampling_ctx.repetition_idx is not None
                and samples.ndim != 4
                or sampling_ctx.repetition_idx is None
                and samples.ndim == 4
            ):
                raise ValueError(
                    "Either there is no repetition index or the samples are not 4-dimensional. This should not happen."
                )

            else:
                samples = samples.repeat(n_samples, 1, 1).detach()

        else:
            # Sample from distribution
            samples = self.distribution.sample((n_samples,))

            if sampling_ctx.repetition_idx is not None and samples.ndim == 4:
                # repetition_idx shape: (n_samples,)
                repetition_idx = sampling_ctx.repetition_idx[instance_mask]

                indices = repetition_idx.view(-1, 1, 1, 1).expand(-1, samples.shape[1], samples.shape[2], -1)

                # Gather samples according to repetition index
                samples = torch.gather(samples, dim=-1, index=indices).squeeze(-1)

            elif (
                sampling_ctx.repetition_idx is not None
                and samples.ndim != 4
                or sampling_ctx.repetition_idx is None
                and samples.ndim == 4
            ):
                raise ValueError(
                    "Either there is no repetition index or the samples are not 4-dimensional. This should not happen."
                )

        if samples.shape[0] != sampling_ctx.channel_index[instance_mask].shape[0]:
            raise ValueError(
                f"Sample shape mismatch: got {samples.shape[0]}, expected {sampling_ctx.channel_index[instance_mask].shape[0]}"
            )

        if self.out_channels == 1:
            # If the output of the input module has a single channel, set the output_ids to zero since
            # this input was broadcasted to match the channel dimension of the other inputs
            sampling_ctx.channel_index.zero_()

        index = sampling_ctx.channel_index[instance_mask].unsqueeze(-1)

        # Index the channel_index to get the correct samples for each scope
        samples = samples.gather(dim=2, index=index).squeeze(2)

        # Ensure, that no data is overwritten
        if data[samples_mask].isfinite().any():
            raise RuntimeError("Data already contains values at the specified mask. This should not happen.")

        # Update data inplace
        samples_mask_subset = samples_mask[instance_mask][:, self.scope.query]
        data[samples_mask] = samples[samples_mask_subset].to(data.dtype)

        return data

    def sample_with_evidence(
        self,
        evidence: Tensor,
        num_samples: int = 1,
        is_mpe: bool = False,
        check_support: bool = True,
        cache: Cache | None = None,
        sampling_ctx: Optional[SamplingContext] = None,
    ) -> Tensor:
        """Sample values conditioned on provided evidence.

        Leaf modules already operate directly on the evidence tensor, so this simply
        forwards to ``sample`` while ensuring cache initialization happens once.
        """
        cache = init_cache(cache)

        if evidence is None:
            raise ValueError("Evidence tensor must be provided for leaf sampling.")

        if num_samples is not None and num_samples != evidence.shape[0]:
            raise ValueError(
                f"num_samples ({num_samples}) must match evidence batch size ({evidence.shape[0]})."
            )

        return self.sample(
            data=evidence,
            is_mpe=is_mpe,
            check_support=check_support,
            cache=cache,
            sampling_ctx=sampling_ctx,
        )

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Optional["LeafModule"]:
        """Structural marginalization for leaf module.

        Structurally marginalizes the specified leaf module.
        If the leaf's scope contains none of the random variables to marginalize, then the leaf is returned unaltered.
        If the leaf's scope is fully marginalized over, then None is returned.

        Args:
            marg_rvs:
                Iterable of integers representing the indices of the random variables to marginalize.
            prune:
                Boolean indicating whether or not to prune nodes and modules where possible.
                Has no effect here. Defaults to True.
            cache:
                Optional cache dictionary.

        Returns:
            Unaltered leaf module or None if it is completely marginalized.
        """
        # initialize cache
        cache = init_cache(cache)

        # Marginalized scope
        scope_marg = Scope([q for q in self.scope.query if q not in marg_rvs])
        # Get indices of marginalized random variables in the original scope
        idxs_marg = [i for i, q in enumerate(self.scope.query) if q in scope_marg.query]

        if len(scope_marg.query) == 0:
            return None

        # Construct new leaf with marginalized scope and params
        marg_params_dict = self.marginalized_params(idxs_marg)

        # Make sure to detach the parameters first
        marg_params_dict = {k: v.detach() for k, v in marg_params_dict.items()}

        # Construct new object of the same class as the leaf
        return self.__class__(
            scope=scope_marg,
            **marg_params_dict,
        )
