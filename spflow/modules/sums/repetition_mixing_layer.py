from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterCombinationError
from spflow.modules.module import Module
from spflow.modules.sums import Sum
from spflow.utils.cache import Cache, cached
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context


class RepetitionMixingLayer(Sum):
    """Mixing layer for RAT-SPN region nodes.

    Specialized sum node for RAT-SPNs. Creates mixtures over input channels.
    Extends Sum with RAT-SPN specific optimizations.
    """

    def __init__(
        self,
        inputs: Module,
        out_channels: int | None = None,
        num_repetitions: int = 1,
        weights: Tensor | None = None,
    ) -> None:
        """Initialize mixing layer for RAT-SPN.

        Args:
            inputs: Input module to mix over channels.
            out_channels: Number of output mixture components.
            num_repetitions: Number of parallel repetitions.
            weights: Initial mixing weights (if None, randomly initialized).
        """
        super().__init__(inputs, out_channels, num_repetitions, weights)

        if self.out_shape.channels != self.inputs.out_shape.channels:
            raise ValueError("out_channels must match the out_channels of the input module.")

        self.sum_dim = 2

    @property
    def feature_to_scope(self) -> np.ndarray:
        return self.inputs.feature_to_scope

    def _process_weights_parameter(
        self,
        inputs: Module | list[Module],
        weights: Tensor | None,
        out_channels: int | None,
        num_repetitions: int | None,
    ) -> tuple[Tensor | None, int | None, int | None]:
        if weights is None:
            return weights, out_channels, num_repetitions

        if out_channels is not None:
            raise InvalidParameterCombinationError(
                f"Cannot specify both 'out_channels' and 'weights' for 'Sum' module."
            )

        weight_dim = weights.dim()
        if weight_dim == 1:
            weights = weights.view(1, -1, 1)
        elif weight_dim == 2:
            weights = weights.view(1, weights.shape[0], weights.shape[1])
        elif weight_dim == 3:
            pass
        else:
            raise ValueError(
                f"Weights for 'RepetitionMixingLayer' must be a 1D, 2D, or 3D tensor but was {weight_dim}D."
            )

        inferred_num_repetitions = weights.shape[-1]
        if num_repetitions is not None and (
            num_repetitions != 1 and num_repetitions != inferred_num_repetitions
        ):
            raise InvalidParameterCombinationError(
                f"Cannot specify 'num_repetitions' that does not match weights shape for 'Sum' module. "
                f"Was {num_repetitions} but weights shape indicates {inferred_num_repetitions}."
            )
        num_repetitions = inferred_num_repetitions

        out_channels = weights.shape[1]

        return weights, out_channels, num_repetitions

    def _get_weights_shape(self) -> tuple[int, int, int]:
        return (
            self.in_shape.features,
            self.out_shape.channels,
            self.out_shape.repetitions,
        )

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        """Generate samples by choosing mixture components.

        Args:
            num_samples: Number of samples to generate.
            data: Data tensor to fill with samples.
            is_mpe: Whether to perform most probable explanation (MPE) sampling.
            cache: Cache for storing intermediate computations.
            sampling_ctx: Sampling context for managing sampling state.

        Returns:
            Tensor: Generated samples.
        """

        # Handle num_samples case (create empty data tensor)
        if data is None:
            if num_samples is None:
                num_samples = 1
            data = torch.full((num_samples, len(self.scope.query)), torch.nan, device=self.device)

        # Initialize sampling context if not provided
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0], data.device)

        logits = self.logits

        logits = logits.unsqueeze(0).expand(
            sampling_ctx.channel_index.shape[0], -1, -1, -1
        )  # shape [b , n_features , in_c, out_c]

        # Check if we have cached input log-likelihoods to compute posterior
        if (
            cache is not None
            and "log_likelihood" in cache
            and cache["log_likelihood"].get(self.inputs) is not None
        ):
            # Compute log posterior by reweighing logits with input lls
            input_lls = cache["log_likelihood"][self.inputs]
            log_prior = logits
            # Only unsqueeze if input_lls has fewer dims than log_prior
            if input_lls.dim() < log_prior.dim():
                input_lls = input_lls.unsqueeze(3)
            log_posterior = log_prior + input_lls
            log_posterior = log_posterior.log_softmax(dim=3)
            logits = log_posterior

        if is_mpe:
            # Take the argmax of the logits to obtain the most probable index
            repetition_idx = torch.argmax(logits.sum(-2), dim=-1).squeeze(-1)
        else:
            # Sample from categorical distribution defined by weights to obtain indices for repetitions
            # sum up the input channel for distribution
            repetition_idx = torch.distributions.Categorical(logits=logits.sum(-2)).sample()

        sampling_ctx.repetition_idx = repetition_idx

        # Sample from input module
        self.inputs.sample(
            data=data,
            is_mpe=is_mpe,
            cache=cache,
            sampling_ctx=sampling_ctx,
        )

        return data

    @cached
    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log likelihood via weighted log-sum-exp.

        Args:
            data: Input data tensor.
            cache: Cache for storing intermediate computations.

        Returns:
            Tensor: Computed log likelihood values.
        """

        batch_size = data.shape[0]
        ll = self.inputs.log_likelihood(
            data,
            cache=cache,
        )

        log_weights = self.log_weights.unsqueeze(0)  # shape: (1, F, IC, OC)

        # Weighted log-likelihoods
        weighted_lls = ll + log_weights  # shape: (B, F, OC, R) + (1, F, OC, R) = (B, F, R, OC)

        # Sum over input channels (sum_dim + 1 since here the batch dimension is the first dimension)
        output = torch.logsumexp(weighted_lls, dim=self.sum_dim + 1)  # shape: (B, F, OC, R)

        # Since modules always have R as last dimension, we need to set it to 1 as Mixing mixes over it
        num_repetitions_after_mixing = 1
        return output.view(batch_size, self.out_shape.features, self.out_shape.channels, num_repetitions_after_mixing)

    def expectation_maximization(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> None:
        """Perform expectation-maximization step.

        Args:
            data: Input data tensor.
            cache: Optional cache dictionary with log-likelihoods.

        Raises:
            ValueError: If required log-likelihoods are not found in cache.
        """
        if cache is None:
            cache = Cache()

        with torch.no_grad():
            # ----- expectation step -----

            # Get input LLs from cache
            input_lls = cache["log_likelihood"].get(self.inputs)
            if input_lls is None:
                raise ValueError("Input log-likelihoods not found in cache. Call log_likelihood first.")

            # Get module lls from cache
            module_lls = cache["log_likelihood"].get(self)
            if module_lls is None:
                raise ValueError("Module log-likelihoods not found in cache. Call log_likelihood first.")

            log_weights = self.log_weights.unsqueeze(0)
            log_grads = torch.log(module_lls.grad)

            log_expectations = log_weights + log_grads + input_lls - module_lls
            log_expectations = log_expectations.logsumexp(0)  # Sum over batch dimension
            log_expectations = log_expectations.log_softmax(self.sum_dim)  # Normalize

            # ----- maximization step -----
            self.log_weights = log_expectations

        # Recursively call EM on inputs
        self.inputs.expectation_maximization(data, cache=cache)
