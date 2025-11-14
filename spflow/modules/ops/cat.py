from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.modules.base import Module
from spflow.utils.cache import Cache, init_cache
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
    def feature_to_scope(self) -> list[Scope]:
        if self.dim == 1:
            scope_list = []
            return [scope_list + module.feature_to_scope for module in self.inputs]
        else:
            return self.inputs[0].feature_to_scope

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, dim={self.dim}"

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
        cache = init_cache(cache)
        log_cache = cache.setdefault("log_likelihood", {})

        # get log likelihoods for all inputs
        lls = []
        for input_module in self.inputs:
            input_ll = input_module.log_likelihood(data, cache=cache)
            log_cache[input_module] = input_ll
            lls.append(input_ll)

        # Concatenate log likelihoods
        output = torch.cat(lls, dim=self.dim)
        log_cache[self] = output
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

        cache = init_cache(cache)
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

        if self.dim == 1:
            # split_size = self.out_features // len(self.inputs)
            # channel_index_per_module = sampling_ctx.channel_index.split(split_size, dim=self.dim)
            # mask_per_module = sampling_ctx.mask.split(split_size, dim=self.dim)
            channel_index_per_module = []
            mask_per_module = []
            for s in self.feature_to_scope:
                query = Scope.join_all(s).query
                channel_index_per_module.append(sampling_ctx.channel_index[:, query])
                mask_per_module.append(sampling_ctx.mask[:, query])

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
        cache = init_cache(cache)

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
