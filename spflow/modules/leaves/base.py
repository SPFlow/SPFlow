from __future__ import annotations

from abc import ABC
from typing import Optional, Dict, Callable

import torch
from torch import Tensor

from spflow.distributions.base import Distribution
from spflow.meta.data.scope import Scope
from spflow.modules.base import Module
from spflow.utils.cache import Cache, cached
from spflow.utils.leaves import apply_nan_strategy
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context


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
    def distribution(self) -> Distribution:
        """Returns the underlying distribution object."""
        self._distribution

    @property
    def _supported_value(self):
        """Returns the supported values of the distribution."""
        return self.distribution._supported_value

    def params(self) -> Dict[str, Tensor]:
        """Returns the parameters of the distribution."""
        return self.distribution.params()

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
        return self.distribution.device

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
        mle_weights = self._prepare_mle_weights(scoped_data, normalized_weights_flat)

        return scoped_data, mle_weights

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
            preprocess_data: Select scope-relevant features.
        """
        # Step 1: Prepare normalized data and weights
        data_prepared, weights_prepared = self._prepare_mle_data(
            data=data,
            weights=weights,
            nan_strategy=nan_strategy,
        )

        # Step 2: Update distribution-specific statistics (implemented by distribution)
        self.distribution._mle_update_statistics(data_prepared, weights_prepared, bias_correction)

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
