from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable

import torch
from torch import Tensor, nn

from spflow.meta.data.scope import Scope
from spflow.modules.base import Module
from spflow.utils.cache import Cache, cached
from spflow.utils.leaves import apply_nan_strategy, _prepare_mle_weights, parse_leaf_args
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context


class LeafModule(Module, ABC):
    def __init__(
        self,
        scope: Scope | int | list[int],
        out_channels: int = None,
        num_repetitions: int = 1,
        params: list[Tensor | None] | None = None,
        parameter_network: nn.Module = None,
        validate_args: bool | None = True,
    ):
        """Base class for leaf distribution modules.

        Args:
            scope: Variable scope (Scope, int, or list[int]).
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions (for 3D event shapes).
            params: List of parameter tensors (can include None to trigger random init).
            parameter_network: Optional neural network for parameter generation.
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
        self.parameter_network = parameter_network
        self._validate_args = validate_args

    @property
    def is_conditional(self):
        """Indicates if the leaf uses a parameter network for conditional parameters."""
        return self.parameter_network is not None

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

        Subclasses should implement this method to construct the distribution
        from the provided parameters.

        Args:
            params: Dictionary of distribution parameters.

        Returns:
            torch.distributions.Distribution constructed from the parameters.
        """
        return self._torch_distribution_class(validate_args=self._validate_args, **params)  # type: ignore[call-arg]

    def conditional_distribution(self, evidence: Tensor) -> torch.distributions.Distribution:
        """Generates torch.distributions object conditionally based on evidence.

        Subclasses should override this method to construct distribution from parameter network output.

        Args:
            evidence: Evidence tensor for conditioning.

        Returns:
            torch.distributions.Distribution constructed from conditional parameters.
        """
        if evidence is None:
            raise ValueError("Evidence tensor must be provided for conditional distribution.")
        return self.__make_distribution(self.parameter_network(evidence))

    @property
    @abstractmethod
    def _supported_value(self) -> float:
        """Returns a value in the support of the distribution (for NaN imputation)."""
        pass

    @abstractmethod
    def params(self) -> Dict[str, Tensor]:
        """Returns the parameters of the distribution."""
        pass

    @abstractmethod
    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute distribution-specific statistics and update parameters.

        Args:
            data: Scope-filtered data.
            weights: Normalized weights.
            bias_correction: Apply bias correction.
        """
        pass

    def _compute_parameter_estimates(
        self, data: Tensor, weights: Tensor, bias_correction: bool
    ) -> Dict[str, Tensor]:
        """Compute raw MLE parameter estimates without broadcasting.

        Used internally by both simple and KMeans clustering paths.
        Subclasses should override this method for better efficiency when supporting KMeans.

        Args:
            data: Scope-filtered data.
            weights: Normalized weights.
            bias_correction: Apply bias correction.

        Returns:
            Dictionary mapping parameter names to raw estimates (shape: out_features).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _compute_parameter_estimates() "
            "to support use_kmeans=True. Either implement this method or use use_kmeans=False."
        )

    def _set_mle_parameters(self, params_dict: Dict[str, Tensor]) -> None:
        """Set MLE-estimated parameters.

        This method handles the assignment of estimated parameters, accounting for both
        direct nn.Parameter objects and property-based parameters with custom setters.

        Subclasses can override this method to explicitly specify how parameters should
        be set, which improves code clarity and serves as documentation.

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

        # Step 2: Apply NaN strategy (drop/impute)
        scoped_data, normalized_weights = apply_nan_strategy(nan_strategy, scoped_data, self.device, weights)

        # Step 3: Prepare weights for broadcasting
        # Convert from (batch, 1) to (batch, 1, 1, ...) for proper broadcasting
        # with multi-dimensional data
        normalized_weights_flat = normalized_weights.squeeze(-1)
        mle_weights = _prepare_mle_weights(scoped_data, normalized_weights_flat)

        return scoped_data, mle_weights

    def _update_parameters_with_kmeans(
        self, data: Tensor, weights: Tensor, bias_correction: bool
    ) -> None:
        """Update parameters using KMeans clustering followed by MLE for each cluster.

        Clusters data into `out_channels` clusters per repetition, then estimates parameters
        for each cluster separately using `_compute_parameter_estimates()`.

        Args:
            data: Scope-filtered data.
            weights: Normalized weights.
            bias_correction: Apply bias correction.
        """
        from fast_pytorch_kmeans import KMeans

        # Get parameter names from the distribution
        param_names = list(self.params().keys())

        # Create empty tensors for all parameters
        new_params = {name: torch.empty_like(getattr(self, name)) for name in param_names}

        # Iterate over repetitions
        for rep_idx in range(self.num_repetitions):
            # Run KMeans to cluster data into out_channels clusters
            kmeans = KMeans(
                n_clusters=self.out_channels, mode="euclidean", init_method="kmeans++"
            )
            cluster_ids = kmeans.fit_predict(data)

            # For each cluster/channel, filter data and estimate parameters
            for channel_idx in range(self.out_channels):
                # Filter data and weights by cluster membership
                cluster_mask = cluster_ids == channel_idx
                cluster_data = data[cluster_mask]
                cluster_weights = weights[cluster_mask]

                if cluster_data.size(0) == 0:
                    # Empty cluster - use global statistics as fallback
                    estimates = self._compute_parameter_estimates(data, weights, bias_correction)
                else:
                    estimates = self._compute_parameter_estimates(
                        cluster_data, cluster_weights, bias_correction
                    )

                # Assign estimated parameters to the specific channel and repetition
                for param_name, param_value in estimates.items():
                    new_params[param_name][:, channel_idx, rep_idx] = param_value

        # Assign all parameters using the helper method
        self._set_mle_parameters(new_params)

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Optional[Tensor] = None,
        bias_correction: bool = True,
        nan_strategy: str | Callable | None = None,
        use_kmeans: bool = False,
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
            use_kmeans: If True, cluster data using KMeans before estimation.
                        Runs KMeans for each repetition with out_channels clusters.
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
        if use_kmeans:
            self._update_parameters_with_kmeans(data_prepared, weights_prepared, bias_correction)
        else:
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
            else:
                dist = self.distribution

            # Sample from distribution
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
