from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable, Iterable

import numpy as np
import torch
from torch import Tensor

from spflow.meta.data.scope import Scope
from spflow.modules.base import Module
from spflow.utils.cache import Cache, cached
from spflow.utils.leaves import apply_nan_strategy, parse_leaf_args
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context


class LeafModule(Module, ABC):
    def __init__(
        self,
        scope: Scope | int | list[int],
        out_channels: int = None,
        num_repetitions: int = 1,
        params: list[Tensor | None] | None = None,
        parameter_fn: Callable[[Tensor], dict[str, Tensor]] = None,
        validate_args: bool | None = True,
    ):
        """Base class for leaf distribution modules.

        Args:
            scope: Variable scope (Scope, int, or list[int]).
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions (for 3D event shapes).
            params: List of parameter tensors (can include None to trigger random init).
            parameter_fn: Optional function that takes evidence and returns distribution parameters as dictionary.
            validate_args: Whether to enable torch.distributions argument validation.
        """
        super().__init__()

        event_shape = parse_leaf_args(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
            params=params,
        )

        # If not already a Scope, convert int or list[int] to Scope
        if not isinstance(scope, Scope):
            scope = Scope(scope)

        self.scope = scope.copy()
        self._event_shape = event_shape
        self.parameter_fn = parameter_fn
        self._validate_args = validate_args

    @property
    def inputs(self) -> Module | Iterable[Module]:
        """Leaf modules do not have inputs."""
        raise AttributeError("LeafModule does not have 'input' attribute -- this should not have been called.")

    @inputs.setter
    def inputs(self, value):
        """Leaf modules do not have inputs."""
        raise AttributeError("LeafModule does not have 'input' attribute -- this should not have been called.")

    @property
    def is_conditional(self):
        """Indicates if the leaf uses a parameter network for conditional parameters."""
        return self.parameter_fn is not None

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying torch.distributions.Distribution object."""
        return self.__make_distribution(self.params())

    @property
    @abstractmethod
    def _torch_distribution_class(self) -> type[torch.distributions.Distribution]:
        pass

    def __make_distribution(self, params: Dict[str, Tensor]) -> torch.distributions.Distribution:
        """Helper method to create distribution from given parameters.

        Args:
            params: Dictionary of distribution parameters.

        Returns:
            torch.distributions.Distribution constructed from the parameters.
        """
        return self._torch_distribution_class(validate_args=self._validate_args, **params)  # type: ignore[call-arg]

    def conditional_distribution(self, evidence: Tensor) -> torch.distributions.Distribution:
        """Generates torch.distributions object conditionally based on evidence.

        Args:
            evidence: Evidence tensor for conditioning.

        Returns:
            torch.distributions.Distribution constructed from conditional parameters.
        """
        if evidence is None:
            raise ValueError("Evidence tensor must be provided for conditional distribution.")
        params = self.parameter_fn(evidence)
        return self.__make_distribution(params)

    @property
    @abstractmethod
    def _supported_value(self) -> float:
        """Returns a value in the support of the distribution (for NaN imputation)."""
        pass

    @abstractmethod
    def params(self) -> Dict[str, Tensor]:
        """Returns the parameters of the distribution."""
        pass

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute and set MLE parameter estimates.

        Args:
            data: Input data tensor.
            weights: Weight tensor for each data point.
            bias_correction: Whether to apply bias correction to variance estimate.
        """
        data = data.view(data.shape[0], self.out_features, 1, 1)  # Add channel and repetition dims
        estimates = self._compute_parameter_estimates(data, weights, bias_correction)

        self._set_mle_parameters(estimates)

    @abstractmethod
    def _compute_parameter_estimates(
        self, data: Tensor, weights: Tensor, bias_correction: bool
    ) -> Dict[str, Tensor]:
        """Compute raw MLE parameter estimates without broadcasting.

        Used internally by both simple and KMeans clustering paths.

        Args:
            data: Scope-filtered data.
            weights: Normalized weights.
            bias_correction: Apply bias correction.

        Returns:
            Dictionary mapping parameter names to raw estimates (shape: out_features).
        """
        pass

    def _set_mle_parameters(self, params_dict: Dict[str, Tensor]) -> None:
        """Set MLE-estimated parameters.

        This method handles the assignment of estimated parameters, accounting for both
        direct nn.Parameter objects and property-based parameters with custom setters.

        Args:
            params_dict: Dictionary mapping parameter names to their estimated values.
        """
        for param_name, param_tensor in params_dict.items():
            try:
                # Try using property setter (works for properties like 'scale')
                setattr(self, param_name, param_tensor)
            except TypeError:
                # Direct parameter (like 'loc') - update .data attribute
                getattr(self, param_name).data = param_tensor

    @property
    def mode(self) -> Tensor:
        """Return distribution mode.

        Returns:
            Mode of the distribution.
        """
        return self.distribution.mode

    def marginalized_params(self, indices: list[int]) -> dict[str, Tensor]:
        """Return parameters marginalized to specified indices.

        Args:
            indices: List of indices to marginalize to.

        Returns:
            Dictionary of marginalized parameters.
        """
        return {k: v[indices] for k, v in self.params().items()}

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
    def num_repetitions(self) -> int:
        """Return number of repetitions.

        Returns:
            Number of repetitions (last dimension of event shape).
        """
        return self.event_shape[2]

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
    def feature_to_scope(self) -> np.ndarray[Scope]:
        """Return list of scopes per feature.

        Returns:
            List of Scope objects, one per feature.
        """
        scopes = np.empty((self.out_features, self.num_repetitions), dtype=Scope)
        for i in range(self.out_features):
            for j in range(self.num_repetitions):
                scopes[i, j] = Scope([self.scope.query[i]])

        return scopes

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

    def _broadcast_to_event_shape(self, param_est: Tensor) -> Tensor:
        """Broadcast parameter estimate to match event_shape.

        Args:
            param_est: Parameter estimate tensor to broadcast.

        Returns:
            Parameter estimate broadcasted to match event_shape.
        """
        target_shape = tuple(self.event_shape)

        # If the parameter already matches the event shape (possibly with extra trailing dims),
        # there is nothing to broadcast. This prevents unsqueezing again during chained calls.
        if tuple(param_est.shape[: len(target_shape)]) == target_shape:
            return param_est

        if len(target_shape) == 2:
            param_est = param_est.unsqueeze(1).repeat(
                1,
                self.out_channels,
                *([1] * (param_est.dim() - 1)),
            )
        elif len(target_shape) == 3:
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

    def expectation_maximization(
        self,
        data: torch.Tensor,
        bias_correction: bool = False,
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
            expectations += 1e-12  # numerical stability
            expectations /= expectations.sum(0, keepdim=True)  # Normalize

            # ----- maximization step -----
            # update parameters through maximum weighted likelihood estimation
            self.maximum_likelihood_estimation(
                data,
                weights=expectations,
                bias_correction=bias_correction,
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

        if data.dim() != 2:
            raise ValueError(f"Data must be 2-dimensional (batch, num_features), got shape {data.shape}.")
        # get information relevant for the scope
        data_q = data[:, self.scope.query]
        if self.event_shape[0] != len(self.scope.query):
            raise RuntimeError(
                f"event_shape mismatch for {self.__class__.__name__}: event_shape={self.event_shape}, scope_len={len(self.scope.query)}"
            )

        # ----- marginalization -----
        marg_mask = torch.isnan(data_q)
        has_marginalizations = marg_mask.any()

        # If there are any marg_ids, set them to 0.0 to ensure that log_prob call is successful
        # and doesn't throw errors due to NaNs
        if has_marginalizations:
            data_q[marg_mask] = self._supported_value

        # ----- log probabilities -----

        # Unsqueeze scope_data to make space for out_channels and repetition dimensions
        # event_shape is now always [features, out_channels, num_repetitions]
        data_q = data_q.unsqueeze(2).unsqueeze(-1)

        if self.is_conditional:
            # Get evidence
            data_e = data[:, self.scope.evidence]
            dist = self.conditional_distribution(data_e)
        else:
            dist = self.distribution

        log_prob = dist.log_prob(data_q)

        # Marginalize entries - broadcast mask to log_prob shape
        if has_marginalizations:
            # Expand marg_mask to match log_prob shape by broadcasting
            # marg_mask is [batch, features], expand to [batch, features, 1, 1]
            marg_mask_for_log_prob = marg_mask.unsqueeze(2).unsqueeze(-1)
            # Broadcast to log_prob shape
            marg_mask_for_log_prob = torch.broadcast_to(marg_mask_for_log_prob, log_prob.shape)
            log_prob[marg_mask_for_log_prob] = 0.0

        # Set marginalized scope data back to NaNs
        if has_marginalizations:
            marg_mask_for_data = marg_mask.unsqueeze(2).unsqueeze(-1)
            data_q[marg_mask_for_data] = torch.nan

        return log_prob

    def _prepare_mle_data(
        self,
        data: Tensor,
        weights: Tensor | None = None,
        nan_strategy: str | Callable | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Prepare normalized data and weights for MLE computation.

        Args:
            data: Input data tensor.
            weights: Optional sample weights.
            nan_strategy: Handle NaN ('ignore', callable, or None).

        Returns:
            Scope-filtered data and normalized weights.
        """

        # Step 1: Select scope-relevant features
        scoped_data = data[:, self.scope.query]

        if weights is None:
            weights = torch.ones(
                scoped_data.shape[0],
                self.out_features,
                self.out_channels,
                self.num_repetitions,
                device=self.device,
            )

        # Step 2: Apply NaN strategy (drop/impute)
        scoped_data, normalized_weights = apply_nan_strategy(nan_strategy, scoped_data, self.device, weights)

        return scoped_data, normalized_weights

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Optional[Tensor] = None,
        bias_correction: bool = True,
        nan_strategy: str | Callable | None = None,
        cache: Cache | None = None,
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
        """

        if self.is_conditional:
            raise RuntimeError(f"MLE not supported for conditional leaf {self.__class__.__name__}.")

        # Step 1: Prepare normalized data and weights
        data_prepared, weights_prepared = self._prepare_mle_data(
            data=data,
            weights=weights,
            nan_strategy=nan_strategy,
        )

        # Step 2: Update distribution-specific statistics
        self._mle_update_statistics(data_prepared, weights_prepared, bias_correction)

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

        if sampling_ctx.repetition_idx is None:
            if self.num_repetitions > 1:
                raise ValueError(
                    "Repetition index must be provided in sampling context for leaves with multiple repetitions."
                )
            else:
                sampling_ctx.repetition_idx = torch.zeros(data.shape[0], dtype=torch.long, device=data.device)


        if is_mpe:
            # Get mode of distribution as MPE
            samples = self.mode.unsqueeze(0)
            if sampling_ctx.repetition_idx is not None and samples.ndim == 4:
                samples = samples.repeat(n_samples, 1, 1, 1).detach()
                # repetition_idx shape: (n_samples,)
                repetition_idx = sampling_ctx.repetition_idx[instance_mask]

                r_idxs = repetition_idx.view(-1, 1, 1, 1).expand(-1, samples.shape[1], samples.shape[2], -1)

                # Gather samples according to repetition index
                samples = torch.gather(samples, dim=-1, index=r_idxs).squeeze(-1)

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
            if self.is_conditional:
                # Get evidence
                evidence = data[instance_mask][:, self.scope.evidence]
                dist = self.conditional_distribution(evidence)
                samples = dist.sample((1,)).squeeze(0)  # Distribution parameters already contain batch dim
            else:
                dist = self.distribution
                # Sample n_samples from distribution
                samples = dist.sample((n_samples,))


            if sampling_ctx.repetition_idx is not None and samples.ndim == 4:
                # repetition_idx shape: (n_samples,)
                repetition_idx = sampling_ctx.repetition_idx[instance_mask]

                r_idxs = repetition_idx.view(-1, 1, 1, 1).expand(-1, samples.shape[1], samples.shape[2], -1)

                # Gather samples according to repetition index
                samples = torch.gather(samples, dim=-1, index=r_idxs).squeeze(-1)

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

        c_idxs = sampling_ctx.channel_index[instance_mask].unsqueeze(-1)

        # Index the channel_index to get the correct samples for each scope
        samples = samples.gather(dim=2, index=c_idxs).squeeze(2)

        # Ensure, that no data is overwritten
        if data[samples_mask].isfinite().any():
            raise RuntimeError("Data already contains values at the specified mask. This should not happen.")

        # Update data inplace - place samples at correct scope positions (vectorized)
        # samples[:, feat_idx] should go to data[:, scope.query[feat_idx]]
        # Only write where the mask is True for that specific position
        
        # Get row indices for instances that need sampling
        row_indices = instance_mask.nonzero(as_tuple=True)[0]  # (n_instances,)
        
        # Create scope indices tensor
        scope_idx = torch.tensor(self.scope.query, dtype=torch.long, device=data.device)
        
        # Expand to create all (row, col) index pairs
        # rows: (n_instances, out_features) - row index repeated for each feature
        # cols: (n_instances, out_features) - scope indices repeated for each instance
        rows = row_indices.unsqueeze(1).expand(-1, len(scope_idx))
        cols = scope_idx.unsqueeze(0).expand(n_samples, -1)
        
        # Get mask subset for scope positions only
        mask_subset = samples_mask[instance_mask][:, self.scope.query]  # (n_instances, out_features)
        
        # Apply mask and flatten for single vectorized assignment
        data[rows[mask_subset], cols[mask_subset]] = samples[mask_subset].to(data.dtype)

        return data

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
        if self.is_conditional:
            raise RuntimeError(
                f"Marginalization not supported for conditional leaf {self.__class__.__name__}."
            )

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
