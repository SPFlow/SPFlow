from __future__ import annotations

import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor

from spflow.exceptions import InvalidParameterCombinationError, MissingCacheError, ShapeError
from spflow.modules.module import Module
from spflow.modules.sums.sum import Sum
from spflow.utils.cache import Cache, cached
from spflow.utils.sampling_context import SamplingContext, sample_from_logits


class RepetitionMixingLayer(Sum):
    """Mixing layer for RAT-SPN region nodes.

    Specialized sum node for RAT-SPNs. Creates mixtures over input channels.
    Extends Sum with RAT-SPN specific optimizations.
    """

    def __init__(
        self,
        inputs: Module,
        out_channels: int = 1,
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
        out_channels: int,
        num_repetitions: int | None,
    ) -> tuple[Tensor | None, int, int | None]:
        if weights is None:
            return weights, out_channels, num_repetitions

        # If out_channels is not the default (1), user explicitly specified both
        if out_channels != 1:
            raise InvalidParameterCombinationError(
                f"Cannot specify both 'out_channels' and 'weights' for 'Sum' module. "
                f"Use only 'weights' to set the number of output channels."
            )
        weight_dim = weights.dim()
        if weight_dim == 1:
            weights = rearrange(weights, "co -> 1 co 1")
        elif weight_dim == 2:
            weights = rearrange(weights, "co r -> 1 co r")
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

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
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
        sampling_ctx.validate_sampling_context(
            num_samples=data.shape[0],
            num_features=self.out_shape.features,
            num_channels=self.out_shape.channels,
            num_repetitions=self.out_shape.repetitions,
            allowed_feature_widths=(1, self.out_shape.features),
        )
        sampling_ctx.broadcast_feature_width(target_features=self.out_shape.features, allow_from_one=True)

        batch_size = int(sampling_ctx.channel_index.shape[0])
        logits = repeat(self.logits, "f co r -> b f co r", b=batch_size)

        # Check if we have cached input log-likelihoods to compute posterior
        if "log_likelihood" in cache and cache["log_likelihood"].get(self.inputs) is not None:
            # Compute log posterior by reweighing logits with input lls
            input_lls = cache["log_likelihood"][self.inputs]
            log_prior = logits
            # Only unsqueeze if input_lls has fewer dims than log_prior
            if input_lls.dim() < log_prior.dim():
                input_lls = rearrange(input_lls, "... -> ... 1")
            log_posterior = log_prior + input_lls
            log_posterior = log_posterior.log_softmax(dim=3)
            logits = log_posterior

        rep_logits = logits.sum(-2)
        rep_samples = sample_from_logits(
            logits=rep_logits,
            dim=-1,
            is_mpe=sampling_ctx.is_mpe,
            is_differentiable=sampling_ctx.is_differentiable,
            hard=sampling_ctx.hard,
            tau=sampling_ctx.tau,
        )
        if rep_logits.shape[1] != 1:
            raise ShapeError(
                "RepetitionMixingLayer.sample currently requires feature width 1 when choosing repetition routes, "
                f"got {rep_logits.shape[1]}."
            )
        if sampling_ctx.is_differentiable:
            repetition_index = rep_samples[:, 0, :]
        else:
            repetition_index = rep_samples[:, 0]

        sampling_ctx.repetition_index = repetition_index

        # Sample from input module
        self.inputs._sample(
            data=data,
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

        ll = self.inputs.log_likelihood(
            data,
            cache=cache,
        )

        log_weights = rearrange(self.log_weights, "f co r -> 1 f co r")

        # Weighted log-likelihoods
        weighted_lls = ll + log_weights  # shape: (B, F, OC, R) + (1, F, OC, R) = (B, F, R, OC)

        # Sum over input channels (sum_dim + 1 since here the batch dimension is the first dimension)
        output = torch.logsumexp(weighted_lls, dim=self.sum_dim + 1)  # shape: (B, F, OC, R)

        # Repetition axis is mixed out, so re-introduce singleton repetition dimension.
        return rearrange(output, "b f co -> b f co 1")

    def _expectation_maximization_step(
        self,
        data: Tensor,
        bias_correction: bool = True,
        *,
        cache: Cache,
    ) -> None:
        """Perform expectation-maximization step.

        Args:
            data: Input data tensor.
            bias_correction: Whether to apply bias correction in leaf updates.
            cache: Optional cache dictionary with log-likelihoods.

        Raises:
            MissingCacheError: If required log-likelihoods are not found in cache.
        """
        with torch.no_grad():
            # ----- expectation step -----

            # Get input LLs from cache
            input_lls = cache["log_likelihood"].get(self.inputs)
            if input_lls is None:
                raise MissingCacheError(
                    "Input log-likelihoods not found in cache. Call log_likelihood first."
                )

            # Get module lls from cache
            module_lls = cache["log_likelihood"].get(self)
            if module_lls is None:
                raise MissingCacheError(
                    "Module log-likelihoods not found in cache. Call log_likelihood first."
                )

            log_weights = rearrange(self.log_weights, "f co r -> 1 f co r")
            log_grads = torch.log(module_lls.grad)

            log_expectations = log_weights + log_grads + input_lls - module_lls
            log_expectations = log_expectations.logsumexp(0)  # Sum over batch dimension
            log_expectations = log_expectations.log_softmax(self.sum_dim)  # Normalize

            # ----- maximization step -----
            self.log_weights = log_expectations

        # Recursively call EM on inputs
        self.inputs._expectation_maximization_step(data, bias_correction=bias_correction, cache=cache)
