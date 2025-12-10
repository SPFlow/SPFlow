from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.modules.base import Module
from spflow.utils.cache import Cache, cached
from spflow.utils.sampling_context import (
    SamplingContext,
    init_default_sampling_context,
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
        self.num_repetitions = self.inputs[0].num_repetitions

        if self.dim == 1:
            # Check if all inputs have the same number of channels
            if not all([module.out_channels == self.inputs[0].out_channels for module in self.inputs]):
                raise ValueError("All inputs must have the same number of channels.")

            # Check that all scopes are disjoint
            if not Scope.all_pairwise_disjoint([module.scope for module in self.inputs]):
                raise ValueError("All inputs must have disjoint scopes.")

            # Scope is the join of all input scopes
            self._scope = Scope.join_all([inp.scope for inp in self.inputs])

        elif self.dim == 2:
            # Check if all inputs have the same number of features and scopes
            if not all([module.out_features == self.inputs[0].out_features for module in self.inputs]):
                raise ValueError("All inputs must have the same number of features.")
            if not Scope.all_equal([module.scope for module in self.inputs]):
                raise ValueError("All inputs must have the same scope.")

            # Scope is the same as all inputs
            self._scope = self.inputs[0].scope
        else:
            raise ValueError("Invalid dimension for concatenation.")

        self._infer_shapes()

    def _infer_shapes(self) -> None:
        """Compute and set input/output shapes for Cat module."""
        from spflow.modules.module_shape import ModuleShape

        # Input shape is the first input's output shape
        self._input_shape = self.inputs[0].output_shape
        self._output_shape = ModuleShape(
            self.out_features, self.out_channels, self.num_repetitions
        )


    @property
    def out_features(self) -> int:
        if self.dim == 1:
            return sum([module.out_features for module in self.inputs])
        else:
            return self.inputs[0].out_features

    @property
    def out_channels(self) -> int:
        if self.dim == 2:
            return sum([module.out_channels for module in self.inputs])
        else:
            return self.inputs[0].out_channels

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

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: Optional[SamplingContext] = None,
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
        data = self._prepare_sample_data(num_samples, data)

        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

        if self.dim == 1:
            # When concatenating features (dim=1), we need to split the sampling context
            # for each input module based on which INTERNAL feature indices belong to that module.
            # 
            # IMPORTANT: sampling_ctx.channel_index and mask are indexed by internal feature
            # position (0, 1, 2, ..., total_features-1), NOT by scope indices. Each module's
            # features occupy a contiguous range in the concatenated output.
            channel_index_per_module = []
            mask_per_module = []
            feature_offset = 0
            for module in self.inputs:
                # Get the internal feature indices for this module (contiguous range)
                num_features = module.out_features
                feature_indices = list(range(feature_offset, feature_offset + num_features))
                channel_index_per_module.append(sampling_ctx.channel_index[:, feature_indices])
                mask_per_module.append(sampling_ctx.mask[:, feature_indices])
                feature_offset += num_features

        elif self.dim == 2:
            # Concatenation happens at out_channels
            # Therefore, we need to use modulo to get the correct output_ids
            channel_index_per_module = []
            mask_per_module = []

            # Get split assignments
            split_size = self.out_channels // len(self.inputs)
            split_assignment = sampling_ctx.channel_index // split_size
            for i, _ in enumerate(self.inputs):
                oids = sampling_ctx.channel_index
                oids_mod = oids.remainder(split_size)
                channel_index_per_module.append(oids_mod)
                mask = (split_assignment == i) & sampling_ctx.mask
                mask_per_module.append(mask)

        else:
            raise ValueError("Invalid dimension for concatenation.")

        # Iterate over inputs
        for i in range(len(self.inputs)):
            input_module = self.inputs[i]
            sampling_ctx_copy = sampling_ctx.copy()
            sampling_ctx_copy.update(channel_index=channel_index_per_module[i], mask=mask_per_module[i])

            input_module.sample(
                data=data,
                is_mpe=is_mpe,
                cache=cache,
                sampling_ctx=sampling_ctx_copy,
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
