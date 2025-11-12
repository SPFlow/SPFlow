from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from spflow.exceptions import StructureError
from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow.modules.product import Product
from spflow.modules.base_product import BaseProduct
from spflow.modules.module import Module
from spflow.utils.cache import Cache, init_cache


class Factorize(BaseProduct):
    r"""
    Factorize module that applies a factorization to the input features. This module is used to create a factorized
    representation of the input features by splitting them into multiple parts based on the specified depth and
    number of repetitions.
    """

    def __init__(
        self,
        inputs: list[Module] | Module,
        depth: int,
        num_repetitions: int,
    ) -> None:
        r"""Initializes ``Factorize`` module.
        Args:
            inputs: List of Leaf modules.
            depth: The depth of the graph. Necessary to determine the number of features in the output.
            num_repetitions: The number of repetitions.

        """
        super().__init__(inputs=inputs)

        self.depth = depth
        self.num_repetitions = num_repetitions
        indices = self._factorize(
            depth, num_repetitions
        )  # shape: [num_features_in, num_features_out, num_repetitions]
        self.register_buffer("indices", indices.to(torch.get_default_dtype()))

    @property
    def out_channels(self) -> int:
        return self.inputs[0].out_channels

    @property
    def out_features(self) -> int:
        return 2**self.depth

    @property
    def feature_to_scope(self) -> list[Scope]:
        return self.inputs[0].feature_to_scope

    def map_out_channels_to_in_channels(self, output_ids: Tensor) -> Tensor:
        return self.unraveled_channel_indices[output_ids]

    def map_out_mask_to_in_mask(self, mask: Tensor) -> Tensor:
        return mask.unsqueeze(-1).expand(-1, -1, len(self.inputs))

    @property
    def device(self):
        """
        Get the device of the module. Necessary hack since this module has no parameters.

        Returns:
            torch.device: The device on which the module's buffers are located.
        """
        return next(iter(self.buffers())).device

    def _factorize(self, depth, num_repetitions):
        r"""
        Generates a factorization of the input features based on the specified depth and number of repetitions.
        The input features are split into groups, and the groups are shuffled to create a randomized factorization.
        Example:
            input features: [1, 2, 3, 4, 5, 6, 7]
            depth: 2
            num_repetitions: 2
            output features: [{1, 3}, {4, 7}, {2}, {5, 6}], [ {3, 5}, {7}, {1, 6}, {2, 4} ]

        """
        scope = self.inputs[0].scope
        num_features = len(scope.query)
        num_features_out = 2**depth
        if num_features < num_features_out:
            raise StructureError(
                f"Number of input features ({num_features}) must be at least equal to the number of output "
                f"features ({num_features_out}). Consider reducing the depth parameter."
            )
        cardinality = int(np.floor(num_features / num_features_out))
        group_sizes = np.ones(num_features_out, dtype=int) * cardinality
        rest = num_features - cardinality * num_features_out
        for i in range(rest):
            group_sizes[i] += 1
        np.random.shuffle(group_sizes)
        scopes = torch.zeros(num_features, num_features_out, num_repetitions)
        for r in range(num_repetitions):
            idxs = torch.randperm(n=num_features)
            offset = 0
            for o in range(num_features_out):
                group_size = group_sizes[o]
                low = offset
                high = offset + group_size
                offset = high
                if o == num_features_out - 1:
                    high = num_features
                scopes[idxs[low:high], o, r] = 1

        return scopes  # .to(torch.int32)

    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log P(data | module) for factorize.

        Args:
            data: The data tensor.
            cache: Optional cache dictionary.

        Returns:
            Log likelihood tensor.
        """
        # initialize cache
        cache = init_cache(cache)

        lls = self._get_input_log_likelihoods(
            data,
            cache=cache,
        )  # lls[0] shape: [batch_size, num_features, num_channel]
        output = torch.einsum("bicr, ior->bocr", lls[0], self.indices)

        return output

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: Optional[SamplingContext] = None,
    ) -> Tensor:
        """Generate samples from the factorize module.

        Args:
            num_samples: Number of samples to generate.
            data: The data tensor to populate with samples.
            is_mpe: Whether to use maximum probability estimation instead of sampling.
            cache: Optional cache dictionary for intermediate results.
            sampling_ctx: Optional sampling context.

        Returns:
            The data tensor populated with samples.
        """
        # Prepare data tensor
        data = self._prepare_sample_data(num_samples, data)

        # initialize contexts
        cache = init_cache(cache)
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

        # gather indices for specific repetitions
        rep_indices = sampling_ctx.repetition_idx.view(-1, 1, 1, 1).expand(
            -1, self.indices.shape[0], self.indices.shape[1], -1
        )
        indices = (
            self.indices.unsqueeze(0)
            .expand(data.shape[0], -1, -1, -1)
            .to(dtype=torch.long, device=self.device)
        )
        indices = torch.gather(indices, dim=-1, index=rep_indices).squeeze(-1)

        # gather channel indices and mask
        channel_index = torch.sum(sampling_ctx.channel_index.unsqueeze(1) * indices, dim=-1)
        mask = torch.sum(sampling_ctx.mask.unsqueeze(1) * indices, dim=-1).bool()

        sampling_ctx.update(channel_index=channel_index, mask=mask)

        self.inputs[0].sample(
            data=data,
            is_mpe=is_mpe,
            cache=cache,
            sampling_ctx=sampling_ctx,
        )

        return data

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Optional[Product | Module]:
        """Marginalize out specified random variables.

        Args:
            marg_rvs: List of random variables to marginalize over.
            prune: Whether to prune the structure.
            cache: Optional cache dictionary.

        Returns:
            The marginalized module or None if fully marginalized.
        """
        # initialize cache
        cache = init_cache(cache)
        # compute layer scope (same for all outputs)
        layer_scope = self.scope
        marg_child = None
        mutual_rvs = set(layer_scope.query).intersection(set(marg_rvs))

        # layer scope is being fully marginalized over
        if len(mutual_rvs) == len(layer_scope.query):
            # passing this loop means marginalizing over the whole scope of this branch
            pass
        # node scope is being partially marginalized
        elif mutual_rvs:
            # marginalize child modules
            marg_child_layer = self.inputs[0].marginalize(marg_rvs, prune=prune, cache=cache)

            # if marginalized child is not None
            if marg_child_layer:
                marg_child = marg_child_layer

        else:
            marg_child = self.inputs[0]

        if marg_child is None:
            return None

        # Prune: if child has only one feature, factorization is redundant - return child directly
        elif prune and marg_child.out_features == 1:
            return marg_child
        else:
            return Factorize(inputs=[marg_child], depth=self.depth, num_repetitions=self.num_repetitions)
