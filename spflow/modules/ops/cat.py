from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.utils.cache import Cache, cached
from spflow.utils.sampling_context import (
    DifferentiableSamplingContext,
    SamplingContext,
)


class Cat(Module):
    def __init__(self, inputs: list[Module], dim: int = -1):
        """Initialize concatenation operation.

        Args:
            inputs: Modules to concatenate.
            dim: Concatenation dimension (0=batch, 1=feature, 2=channel).
        """
        super().__init__()
        self.inputs = nn.ModuleList(inputs)
        self.dim = dim

        if self.dim == 1:
            # Check if all inputs have the same number of channels
            if not all(
                [module.out_shape.channels == self.inputs[0].out_shape.channels for module in self.inputs]
            ):
                raise ValueError("All inputs must have the same number of channels.")

            # Check that all scopes are disjoint
            if not Scope.all_pairwise_disjoint([module.scope for module in self.inputs]):
                raise ValueError("All inputs must have disjoint scopes.")

            # Scope is the join of all input scopes
            self._scope = Scope.join_all([inp.scope for inp in self.inputs])

        elif self.dim == 2:
            # Check if all inputs have the same number of features and scopes
            if not all(
                [module.out_shape.features == self.inputs[0].out_shape.features for module in self.inputs]
            ):
                raise ValueError("All inputs must have the same number of features.")
            if not Scope.all_equal([module.scope for module in self.inputs]):
                raise ValueError("All inputs must have the same scope.")

            # Scope is the same as all inputs
            self._scope = self.inputs[0].scope
        else:
            raise ValueError("Invalid dimension for concatenation.")

        # Shape computation
        self.in_shape = self.inputs[0].out_shape

        if self.dim == 1:
            out_features = sum([module.out_shape.features for module in self.inputs])
            out_channels = self.inputs[0].out_shape.channels
        else:  # dim == 2
            out_features = self.inputs[0].out_shape.features
            out_channels = sum([module.out_shape.channels for module in self.inputs])

        self.out_shape = ModuleShape(out_features, out_channels, self.inputs[0].out_shape.repetitions)

    @property
    def feature_to_scope(self) -> np.ndarray:
        if self.dim == 1:
            # Concatenate along features dimension (axis=0) since we're concatenating features
            return np.concatenate([module.feature_to_scope for module in self.inputs], axis=0)
        else:
            return self.inputs[0].feature_to_scope

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, dim={self.dim}"

    @cached
    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log likelihood by concatenating input log-likelihoods.

        Args:
            data: Input data tensor.
            cache: Optional cache for storing intermediate results.

        Returns:
            Tensor: Concatenated log-likelihood tensor.
        """

        # get log likelihoods for all inputs
        lls = []
        for input_module in self.inputs:
            input_ll = input_module.log_likelihood(data, cache=cache)
            lls.append(input_ll)

        # Concatenate log likelihoods
        output = torch.cat(lls, dim=self.dim)
        return output

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
        is_mpe: bool = False,
    ) -> Tensor:
        """Generate samples by delegating to concatenated inputs.

        Args:
            num_samples: Number of samples to generate.
            data: Optional data tensor to store samples.
            is_mpe: Whether to perform most probable explanation sampling.
            cache: Optional cache for storing intermediate results.
            sampling_ctx: Sampling context for controlling sample generation.

        Returns:
            Tensor: Generated samples tensor.
        """
        # Prepare data tensor

        if self.dim == 1:
            sampling_ctx.require_feature_width(expected_features=self.out_shape.features)
            ranges: list[tuple[int, int]] = []
            feature_offset = 0
            for module in self.inputs:
                num_features = module.out_shape.features
                ranges.append((feature_offset, feature_offset + num_features))
                feature_offset += num_features
            per_module = sampling_ctx.slice_feature_ranges(ranges=ranges)
            channel_index_per_module = [pair[0] for pair in per_module]
            mask_per_module = [pair[1] for pair in per_module]

        elif self.dim == 2:
            sampling_ctx.require_feature_width(expected_features=self.out_shape.features)
            per_module = sampling_ctx.route_channel_offsets(
                child_channel_counts=[int(module.out_shape.channels) for module in self.inputs],
            )
            channel_index_per_module = [pair[0] for pair in per_module]
            mask_per_module = [pair[1] for pair in per_module]

        else:
            raise ValueError("Invalid dimension for concatenation.")

        # Iterate over inputs
        for i in range(len(self.inputs)):
            input_module = self.inputs[i]
            sampling_ctx_copy = sampling_ctx.copy()
            sampling_ctx_copy.update(channel_index=channel_index_per_module[i], mask=mask_per_module[i])
            input_module._sample(
                data=data,
                is_mpe=is_mpe,
                cache=cache,
                sampling_ctx=sampling_ctx_copy,
            )

        return data

    def _rsample(
        self,
        data: Tensor,
        sampling_ctx: DifferentiableSamplingContext,
        cache: Cache,
        is_mpe: bool = False,
    ) -> Tensor:
        """Differentiable structural passthrough for concatenation nodes."""
        sampling_ctx.require_feature_width(expected_features=self.out_shape.features)
        original_channel_probs = sampling_ctx.channel_probs
        original_mask = sampling_ctx.mask

        if self.dim == 1:
            ranges: list[tuple[int, int]] = []
            feature_offset = 0
            for module in self.inputs:
                num_features = module.out_shape.features
                ranges.append((feature_offset, feature_offset + num_features))
                feature_offset += num_features
            per_module = sampling_ctx.slice_feature_prob_ranges(ranges=ranges)
            channel_probs_per_module = [pair[0] for pair in per_module]
            mask_per_module = [pair[1] for pair in per_module]

        elif self.dim == 2:
            child_channel_counts = [int(module.out_shape.channels) for module in self.inputs]
            resolved_channel_probs = sampling_ctx.resolve_channel_probs(
                expected_channels=int(sum(child_channel_counts)),
                module_name=f"{self.__class__.__name__}._rsample",
            )
            if resolved_channel_probs is not sampling_ctx.channel_probs:
                sampling_ctx.update_prob_routing(
                    channel_probs=resolved_channel_probs,
                    mask=sampling_ctx.mask,
                )
            per_module = sampling_ctx.route_channel_prob_offsets(
                child_channel_counts=child_channel_counts,
            )
            channel_probs_per_module = [pair[0] for pair in per_module]
            mask_per_module = [pair[1] for pair in per_module]

        else:
            raise ValueError("Invalid dimension for concatenation.")

        for i, input_module in enumerate(self.inputs):
            sampling_ctx.update_prob_routing(
                channel_probs=channel_probs_per_module[i],
                mask=mask_per_module[i],
            )
            input_module._rsample(
                data=data,
                is_mpe=is_mpe,
                cache=cache,
                sampling_ctx=sampling_ctx,
            )

        sampling_ctx.update_prob_routing(
            channel_probs=original_channel_probs,
            mask=original_mask,
        )
        return data

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Optional["Module"]:
        """Marginalize out specified random variables.

        Args:
            marg_rvs: List of random variable indices to marginalize.
            prune: Whether to prune unnecessary modules after marginalization.
            cache: Optional cache for storing intermediate results.

        Returns:
            Optional[Module]: Marginalized module or None if fully marginalized.
        """

        # compute module scope (same for all outputs)
        module_scope = self.scope

        mutual_rvs = set(module_scope.query).intersection(set(marg_rvs))

        # Node scope is only being partially marginalized
        if mutual_rvs:
            inputs = []
            # marginalize child modules
            for input_module in self.inputs:
                marg_child_module = input_module.marginalize(marg_rvs, prune=prune, cache=cache)

                # if marginalized child is not None
                if marg_child_module:
                    inputs.append(marg_child_module)

            # if all children were marginalized, return None
            if len(inputs) == 0:
                return None

            # if only a single input survived marginalization, return it if pruning is enabled
            if prune and len(inputs) == 1:
                return inputs[0]

            return Cat(inputs=inputs, dim=self.dim)
        else:
            return self
