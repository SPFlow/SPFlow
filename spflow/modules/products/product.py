from __future__ import annotations

import torch
from torch import Tensor

from spflow.meta.data import Scope
from spflow.modules.base import Module
from spflow.modules.ops.cat import Cat
from spflow.utils.cache import Cache, cached
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context


class Product(Module):
    """Product node implementing factorization via conditional independence.

    Computes joint distribution as product of child distributions. Multiple
    inputs are automatically concatenated along the feature dimension.

    Attributes:
        inputs (Module): Input module(s), concatenated if multiple.
    """

    def __init__(self, inputs: Module | list[Module]) -> None:
        """Initialize product node.

        Args:
            inputs: Single module or list of modules (concatenated along features).
        """
        super().__init__()

        # If inputs is a list, ensure concatenation along the feature dimension
        if isinstance(inputs, list):
            if len(inputs) == 1:
                self.inputs = inputs[0]
            else:
                self.inputs = Cat(inputs=inputs, dim=1)
        else:
            self.inputs = inputs

        # Scope of this product module is equal to the scope of its only input
        self.scope = self.inputs.scope
        self.num_repetitions = self.inputs.num_repetitions

    @property
    def out_channels(self) -> int:
        return self.inputs.out_channels

    @property
    def out_features(self) -> int:
        return 1

    @property
    def feature_to_scope(self) -> list[Scope]:
        return [Scope.join_all(self.inputs.feature_to_scope)]

    @cached
    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log likelihood by summing child log-likelihoods across features.

        Args:
            data: Input data tensor.
            cache: Optional cache for storing intermediate results.

        Returns:
            Tensor: Log likelihood values.
        """
        # compute child log-likelihoods
        ll = self.inputs.log_likelihood(
            data,
            cache=cache,
        )

        # multiply children (sum in log-space)
        result = torch.sum(ll, dim=1, keepdim=True)

        return result

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        """Generate samples by delegating to input module.

        Args:
            num_samples: Number of samples to generate.
            data: Optional data tensor to fill with samples.
            is_mpe: Whether to perform most probable explanation.
            cache: Optional cache for storing intermediate results.
            sampling_ctx: Optional sampling context.

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

        # Expand mask and channels to match input module shape
        mask = sampling_ctx.mask.expand(data.shape[0], self.inputs.out_features)
        channel_index = sampling_ctx.channel_index.expand(data.shape[0], self.inputs.out_features)
        sampling_ctx.update(channel_index=channel_index, mask=mask)

        # Delegate to input module for actual sampling
        self.inputs.sample(
            data=data,
            is_mpe=is_mpe,
            cache=cache,
            sampling_ctx=sampling_ctx,
        )
        return data

    def expectation_maximization(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> None:
        """EM step (delegates to input, no learnable parameters).

        Args:
            data: Input data tensor for EM step.
            cache: Optional cache for storing intermediate results.
        """

        # Product has no learnable parameters, delegate to input
        self.inputs.expectation_maximization(data, cache=cache)

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Tensor | None = None,
        cache: Cache | None = None,
    ) -> None:
        """MLE step (delegates to input, no learnable parameters).

        Args:
            data: Input data tensor for MLE step.
            weights: Optional weights for weighted MLE.
            cache: Optional cache for storing intermediate results.
        """

        # Product has no learnable parameters, delegate to input
        self.inputs.maximum_likelihood_estimation(
            data,
            weights=weights,
            cache=cache,
        )

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Product | Module | None:
        """Marginalize out specified random variables.

        Args:
            marg_rvs: List of random variable indices to marginalize.
            prune: Whether to prune unnecessary nodes.
            cache: Optional cache for storing intermediate results.

        Returns:
            Product | Module | None: Marginalized module or None if fully marginalized.
        """

        # compute layer scope (same for all outputs)
        layer_scope = self.scope
        marg_child = None
        mutual_rvs = set(layer_scope.query).intersection(set(marg_rvs))

        # layer scope is being fully marginalized over
        if len(mutual_rvs) == len(layer_scope.query):
            # passing this loop means marginalizing over the whole scope of this branch
            return None

        # node scope is being partially marginalized
        elif mutual_rvs:
            # marginalize child modules
            marg_child_layer = self.inputs.marginalize(marg_rvs, prune=prune, cache=cache)

            # if marginalized child is not None
            if marg_child_layer:
                marg_child = marg_child_layer

        else:
            marg_child = self.inputs

        if marg_child is None:
            return None

        elif prune and marg_child.out_features == 1:
            return marg_child
        else:
            return Product(inputs=marg_child)
