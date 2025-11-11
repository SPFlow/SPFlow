from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional, Dict, Any

import torch
from torch import Tensor

from spflow.meta.data.scope import Scope
from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow.modules.module import Module
from spflow.utils.cache import Cache, init_cache
from spflow.utils.leaf import apply_nan_strategy
import time


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
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of this distribution.

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional PyTorch tensor containing sample instances.
                Each row is regarded as a sample.

        Returns:
            Two dimensional PyTorch tensor indicating for each instance, whether they are part of the support (True) or not (False).
        """

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(data)

        valid = torch.ones_like(data, dtype=torch.bool)

        # check only first entry of num_leaf node dim since all leaf node repetition have the same support

        valid[~nan_mask] = self.distribution.support.check(data)[..., [0]][~nan_mask]

        # check for infinite values
        valid[~nan_mask & valid] &= ~data[~nan_mask & valid].isinf()

        return valid

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

    def _handle_mle_edge_cases(self, param_est: Tensor) -> Tensor:
        """Handle edge cases in parameter estimation (zeros and NaNs).

        Replaces zero values and NaNs with 1e-8 to avoid numerical issues.

        Args:
            param_est: The estimated parameter tensor.

        Returns:
            Parameter tensor with edge cases handled.
        """
        if torch.any(zero_mask := torch.isclose(param_est, torch.tensor(0.0))):
            param_est[zero_mask] = torch.tensor(1e-8)
        if torch.any(nan_mask := torch.isnan(param_est)):
            param_est[nan_mask] = torch.tensor(1e-8)
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
            param_est = param_est.unsqueeze(1).repeat(1, self.out_channels)
        elif len(self.event_shape) == 3:
            param_est = param_est.unsqueeze(1).unsqueeze(1).repeat(1, self.out_channels, self.num_repetitions)

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

    # ToDo: Remove?
    def get_partial_module(self, indices: list[int]) -> Module:
        r"""Returns a partial module with the specified indices."""

        new_query = []
        for i in indices:
            new_query.append(self.scope.query[i])
        new_scope = Scope(new_query, self.scope.evidence)
        return self.__class__(scope=new_scope, out_channels=self.out_channels)

    @property
    def feature_to_scope(self) -> list[Scope]:
        """Returns a list of scopes corresponding to the features in the leaf module."""
        return [Scope([i]) for i in self.scope.query]

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
        # initialize cache
        cache = init_cache(cache)

        # select relevant data for scope
        if preprocess_data:
            data = data[:, self.scope.query]

        # apply NaN strategy
        scope_data, weights = apply_nan_strategy(nan_strategy, data, self, weights, check_support)

        # Forward to the actual distribution
        self.distribution.maximum_likelihood_estimation(scope_data, weights, bias_correction)

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
