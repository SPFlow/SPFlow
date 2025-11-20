"""Mixing layer for RAT-SPN region nodes.

Specialized sum node for RAT-SPNs creating mixture distributions over
input channels. Extends base Sum with RAT-SPN specific optimizations.
"""

from __future__ import annotations

import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta.data import Scope
from spflow.modules.base import Module
from spflow.modules.sums.sum import Sum
from spflow.utils.cache import Cache, cached
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context


class MixingLayer(Sum):
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
        sum_dim: int | None = 1,
    ) -> None:
        """Initialize mixing layer for RAT-SPN.

        Args:
            inputs: Input module to mix over channels.
            out_channels: Number of output mixture components.
            num_repetitions: Number of parallel repetitions.
            weights: Initial mixing weights (if None, randomly initialized).
            sum_dim: Dimension over which to perform mixing.
        """
        super().__init__(inputs, out_channels, num_repetitions, weights, sum_dim)
        if not inputs:
            raise ValueError("'Sum' requires at least one input to be specified.")

        if weights is not None:
            if out_channels is not None:
                raise InvalidParameterCombinationError(
                    f"Cannot specify both 'out_channels' and 'weights' for 'Sum' module."
                )

            out_channels = weights.shape[2]

        if out_channels < 1:
            raise ValueError(
                f"Number of nodes for 'Sum' must be greater of equal to 1 but was {out_channels}."
            )

        self.inputs = inputs

        # Single input, sum over in_channel dimension
        self.sum_dim = sum_dim
        self._out_features = self.inputs.out_features
        self._out_channels_total = out_channels

        if out_channels != inputs.out_channels:
            raise ValueError("out_channels must match the out_channels of the input module.")
        if self._out_features != 1:
            raise ValueError(
                "MixingLayer represents the first layer of the RatSPN, so it must have a single output feature."
            )

        self.num_repetitions = num_repetitions

        # sum up all repetitions
        self._in_channels = self.num_repetitions

        self.weights_shape = (self._out_features, self._in_channels, self._out_channels_total)

        self.scope = self.inputs.scope

        # If weights are not provided, initialize them randomly
        if weights is None:
            weights = (
                # weights has shape (n_nodes, n_scopes, n_inputs) to prevent permutation at ll and sample
                torch.rand(self.weights_shape)
                + 1e-08
            )  # avoid zeros

            # Normalize
            weights /= torch.sum(weights, axis=self.sum_dim, keepdims=True)

        # Register unnormalized log-probabilities for weights as torch parameters
        self.logits = torch.nn.Parameter()

        # Initialize weights (which sets self.logits under the hood accordingly)
        self.weights = weights

    @property
    def out_features(self) -> int:
        return self._out_features

    @property
    def out_channels(self) -> int:
        return self._out_channels_total

    @property
    def feature_to_scope(self) -> list[Scope]:
        return self.inputs.feature_to_scope

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
            log_posterior = log_prior + input_lls.unsqueeze(3)
            log_posterior = log_posterior.log_softmax(dim=2)
            logits = log_posterior

        if is_mpe:
            # Take the argmax of the logits to obtain the most probable index
            repetition_idx = torch.argmax(logits.sum(-1), dim=-1).squeeze(-1)
        else:
            # Sample from categorical distribution defined by weights to obtain indices for repetitions
            # sum up the input channel for distribution
            repetition_idx = torch.distributions.Categorical(logits=logits.sum(-1)).sample()

        # get repetition index for the given channels
        # repetition_idx = repetition_idx.gather(dim=1,index=sampling_ctx.channel_index).squeeze()

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
        weighted_lls = (
            ll.permute(0, 1, 3, 2) + log_weights
        )  # shape: (B, F, R, OC) + (1, F, IC, OC) = (B, F, R = IC, OC)

        # Sum over input channels (sum_dim + 1 since here the batch dimension is the first dimension)
        output = torch.logsumexp(weighted_lls, dim=self.sum_dim + 1)  # shape: (B, F, OC, R)

        # Since modules always have R as last dimension, we need to set it to 1 as Mixing mixes over it
        num_repetitions_after_mixing = 1
        return output.view(batch_size, self.out_features, self.out_channels, num_repetitions_after_mixing)
