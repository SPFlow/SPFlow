from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable

import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta.data.scope import Scope
from spflow.modules.base import Module
from spflow.utils.cache import Cache, cached
from spflow.utils.projections import proj_real_to_bounded, proj_bounded_to_real
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context


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


class LeafModule(Module, ABC):
    def __init__(self, scope: Scope | int | list[int], out_channels: int = None):
        """Base class for leaf distribution modules.

        Args:
            scope: Variable scope (Scope, int, or list[int]).
            out_channels: Number of output channels.
        """
        super().__init__()

        # If not already a Scope, convert int or list[int] to Scope
        if not isinstance(scope, Scope):
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
    def _mle_compute_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute distribution-specific statistics and assign parameters.

        Hook method called by maximum_likelihood_estimation() after data preparation.
        Descriptors handle validation and storage automatically.

        Args:
            data: Scope-filtered data.
            weights: Normalized weights.
            bias_correction: Apply bias correction.
        """
        pass

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability of samples.

        Args:
            x: Input tensor of samples.

        Returns:
            Log probability values.
        """
        return self.distribution.log_prob(x)

    def mode(self) -> Tensor:
        """Return distribution mode.

        Returns:
            Mode of the distribution.
        """
        return self.distribution.mode

    def marginalized_params(self, indices: list[int]) -> Dict[str, Tensor]:
        """Return parameters marginalized to specified indices.

        Args:
            indices: List of indices to marginalize to.

        Returns:
            Dictionary of marginalized parameters.
        """
        return {k: v[indices] for k, v in self.params().items()}

    # MLE Helper Methods
    def _prepare_mle_weights(self, data: Tensor, weights: Optional[Tensor] = None) -> Tensor:
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

    def _handle_mle_edge_cases(
        self,
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

    def _broadcast_to_event_shape(self, param_est: Tensor) -> Tensor:
        """Broadcast parameter estimate to match event_shape.

        Args:
            param_est: Parameter estimate tensor to broadcast.

        Returns:
            Parameter estimate broadcasted to match event_shape.
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
        """Return event shape.

        Returns:
            Event shape tuple.
        """
        if self._event_shape is None:
            raise RuntimeError(f"{self.__class__.__name__} has not set _event_shape in __init__")
        return self._event_shape

    @property
    def out_features(self) -> int:
        """Return number of output features.

        Returns:
            Number of output features.
        """
        return len(self.scope.query)

    @property
    def num_repetitions(self) -> int | None:
        """Return number of repetitions (3D event shape only).

        Returns:
            Number of repetitions or None for 2D event shapes.
        """
        if len(self.event_shape) == 3:
            return self.event_shape[2]
        else:
            return None

    @property
    def out_channels(self) -> int:
        """Return number of output channels.

        Returns:
            Number of output channels.
        """
        if len(self.event_shape) == 1:
            return 1
        else:
            return self.event_shape[1]

    @property
    def feature_to_scope(self) -> list[Scope]:
        """Return list of scopes per feature.

        Returns:
            List of Scope objects, one per feature.
        """
        return [Scope([i]) for i in self.scope.query]

    @property
    def device(self) -> torch.device:
        """Return device of first parameter or buffer.

        Returns:
            Device of the module.
        """
        try:
            return next(iter(self.parameters())).device
        except StopIteration:
            return next(iter(self.buffers())).device

    def expectation_maximization(
        self,
        data: torch.Tensor,
        cache: Cache | None = None,
    ) -> None:
        """Perform single EM step.

        Args:
            data: Input data tensor.
            cache: Optional cache dictionary.
        """
        # initialize cache

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
                cache=cache,
            )

        # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation',
        # we do not need to zero/None parameter gradients

    @cached
    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log-likelihoods, marginalizing over NaN values.

        Args:
            data: Input data tensor.
            cache: Optional cache dictionary.

        Returns:
            Log-likelihood tensor.
        """
        # get information relevant for the scope
        data = data[:, self.scope.query]
        if self.event_shape[0] != len(self.scope.query):
            raise RuntimeError(
                f"event_shape mismatch for {self.__class__.__name__}: event_shape={self.event_shape}, scope_len={len(self.scope.query)}"
            )

        # ----- marginalization -----
        marg_mask = torch.isnan(data)
        has_marginalizations = marg_mask.any()

        # If there are any marg_ids, set them to 0.0 to ensure that log_prob call is successful
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

        return log_prob

    def _prepare_mle_data(
        self,
        data: Tensor,
        weights: Optional[Tensor] = None,
        nan_strategy: Optional[str | Callable] = None,
        cache: Cache | None = None,
        preprocess_data: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Prepare normalized data and weights for MLE computation.

        Args:
            data: Input data tensor.
            weights: Optional sample weights.
            nan_strategy: Handle NaN ('ignore', callable, or None).
            cache: Optional cache dictionary.
            preprocess_data: Select scope-relevant features.

        Returns:
            Scope-filtered data and normalized weights.
        """

        # Step 1: Select scope-relevant features
        scoped_data = data[:, self.scope.query] if preprocess_data else data

        # Step 2: Apply NaN strategy (drop/impute)
        scoped_data, normalized_weights = apply_nan_strategy(nan_strategy, scoped_data, self.device, weights)

        # Step 3: Prepare weights for broadcasting
        # Convert from (batch, 1) to (batch, 1, 1, ...) for proper broadcasting
        # with multi-dimensional data
        normalized_weights_flat = normalized_weights.squeeze(-1)
        mle_weights = self._prepare_mle_weights(scoped_data, normalized_weights_flat)

        return scoped_data, mle_weights

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Optional[Tensor] = None,
        bias_correction: bool = True,
        nan_strategy: Optional[str | Callable] = None,
        cache: Cache | None = None,
        preprocess_data: bool = True,
    ) -> None:
        """Maximum (weighted) likelihood estimation via template method pattern.

        Delegates distribution-specific logic to _mle_compute_statistics() hook.
        Weights normalized to sum to N.

        Args:
            data: Input data tensor.
            weights: Optional sample weights.
            bias_correction: Apply bias correction.
            nan_strategy: Handle NaN ('ignore', callable, or None).
            cache: Optional cache dictionary.
            preprocess_data: Select scope-relevant features.
        """
        # Step 1: Prepare normalized data and weights
        data_prepared, weights_prepared = self._prepare_mle_data(
            data=data,
            weights=weights,
            nan_strategy=nan_strategy,
            cache=cache,
            preprocess_data=preprocess_data,
        )

        # Step 2: Compute distribution-specific statistics (implemented by subclass)
        # Subclass is responsible for assigning parameters directly via descriptors
        self._mle_compute_statistics(data_prepared, weights_prepared, bias_correction)

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: Optional[SamplingContext] = None,
    ) -> Tensor:
        """Sample from leaf distribution given potential evidence.

        Args:
            num_samples: Number of samples to generate.
            data: Optional evidence tensor.
            is_mpe: Perform MPE (mode) instead of sampling.
            cache: Optional cache dictionary.
            sampling_ctx: Optional sampling context.

        Returns:
            Sampled data tensor.
        """
        # Prepare data tensor
        data = self._prepare_sample_data(num_samples, data)

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
        cache: Cache | None = None,
        sampling_ctx: Optional[SamplingContext] = None,
    ) -> Tensor:
        """Sample conditioned on evidence (forwards to sample).

        Args:
            evidence: Evidence tensor.
            num_samples: Number of samples to generate.
            is_mpe: Perform MPE (mode) instead of sampling.
            cache: Optional cache dictionary.
            sampling_ctx: Optional sampling context.

        Returns:
            Sampled data tensor conditioned on evidence.
        """

        if evidence is None:
            raise ValueError("Evidence tensor must be provided for leaves sampling.")

        if num_samples is not None and num_samples != evidence.shape[0]:
            raise ValueError(
                f"num_samples ({num_samples}) must match evidence batch size ({evidence.shape[0]})."
            )

        return self.sample(
            data=evidence,
            is_mpe=is_mpe,
            cache=cache,
            sampling_ctx=sampling_ctx,
        )

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Optional["LeafModule"]:
        """Structurally marginalize specified variables.

        Args:
            marg_rvs: Variable indices to marginalize.
            prune: Unused (for interface consistency).
            cache: Optional cache dictionary.

        Returns:
            Marginalized leaf or None if fully marginalized.
        """
        # initialize cache

        # Marginalized scope
        scope_marg = Scope([q for q in self.scope.query if q not in marg_rvs])
        # Get indices of marginalized random variables in the original scope
        idxs_marg = [i for i, q in enumerate(self.scope.query) if q in scope_marg.query]

        if len(scope_marg.query) == 0:
            return None

        # Construct new leaves with marginalized scope and params
        marg_params_dict = self.marginalized_params(idxs_marg)

        # Make sure to detach the parameters first
        marg_params_dict = {k: v.detach() for k, v in marg_params_dict.items()}

        # Construct new object of the same class as the leaves
        return self.__class__(
            scope=scope_marg,
            **marg_params_dict,
        )


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
