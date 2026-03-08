"""Factorization module for RAT-SPN architecture.

Creates randomized feature partitions for efficient tensorized computations
in RAT-SPNs. Splits features into groups with multiple random repetitions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from einops import rearrange
from torch import Tensor

from spflow.exceptions import StructureError
from spflow.meta.data import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.ops.cat import Cat
from spflow.modules.products.base_product import BaseProduct
from spflow.modules.products.product import Product
from spflow.utils.cache import Cache, cached
from spflow.utils.sampling_context import SamplingContext


class Factorize(BaseProduct):
    """Factorization module for RAT-SPN feature partitioning.

    Creates 2^depth output features by randomly grouping inputs. Multiple
    repetitions provide different randomizations.

    Attributes:
        depth (int): Depth parameter (output features = 2^depth).
        num_repetitions (int): Number of parallel random factorizations.
        indices (Tensor): Factorization matrix (num_features, 2^depth, num_repetitions).
    """

    def __init__(
        self,
        inputs: list[Module] | Module,
        depth: int,
        num_repetitions: int,
    ) -> None:
        """Initialize factorize module.

        Args:
            inputs: Leaf modules or single module to factorize.
            depth: Depth parameter (output features = 2^depth).
            num_repetitions: Number of parallel randomizations.
        """
        if isinstance(inputs, list):
            if len(inputs) > 1:
                inputs = [Cat(inputs=inputs, dim=1)]
        else:
            inputs = [inputs]
        super().__init__(inputs=inputs)

        self.depth = depth
        indices = self._factorize(
            depth, num_repetitions
        )  # shape: [num_features_in, num_features_out, num_repetitions]
        self.register_buffer("indices", indices.to(torch.get_default_dtype()))
        # Sampling can optionally use a dedicated mapping (defaults to forward mapping).
        self.register_buffer("sampling_indices", self.indices.clone())
        self.register_buffer(
            "sampling_feature_owners",
            self.sampling_indices.argmax(dim=1).to(dtype=torch.long),
        )

        # Shape computation
        self.in_shape = self.inputs[0].out_shape
        self.out_shape = ModuleShape(2**self.depth, self.inputs[0].out_shape.channels, num_repetitions)

    @property
    def feature_to_scope(self) -> np.ndarray:
        f2s_inputs = self.inputs[0].feature_to_scope

        # We need to map the input features to output features based on the factorization given in self.indices
        # self.indices shape: [num_features_in, num_features_out, num_repetitions]
        # f2s_inputs shape: [num_features_in, num_repetitions]
        # f2s_outputs shape: [num_features_out, num_repetitions]
        indices = self.indices.detach().cpu().numpy()
        out = np.empty((self.out_shape.features, self.out_shape.repetitions), dtype=Scope)

        for r in range(self.out_shape.repetitions):
            for o in range(self.out_shape.features):
                mask = indices[:, o, r] > 0
                scopes = f2s_inputs[mask, r]
                out[o, r] = Scope.join_all(scopes)

        return out

    def map_out_channels_to_in_channels(self, output_ids: Tensor) -> Tensor:
        if output_ids.is_floating_point():
            return output_ids.unsqueeze(-2)
        return output_ids.unsqueeze(-1)

    def map_out_mask_to_in_mask(self, mask: Tensor) -> Tensor:
        return mask.unsqueeze(-1)

    @property
    def device(self):
        """Get the device of the module."""
        return next(iter(self.buffers())).device

    def _factorize(self, depth, num_repetitions):
        """Generate randomized factorization matrix.

        Args:
            depth: Depth parameter for factorization.
            num_repetitions: Number of parallel randomizations.
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

        return scopes

    @cached
    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log likelihood via tensor contraction.

        Args:
            data: Input data tensor.
            cache: Optional cache for storing intermediate results.

        Returns:
            Tensor: Computed log likelihood values.
        """
        # initialize cache

        lls = self._get_input_log_likelihoods(
            data,
            cache=cache,
        )  # lls[0] shape: [batch_size, num_features, num_channel, num_repetitions]

        output = torch.einsum("bicr, ior->bocr", lls[0], self.indices)
        return output

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
    ) -> Tensor:
        """Generate samples by delegating to input with mapped indices.

        Args:
            num_samples: Number of samples to generate.
            data: Optional data tensor to store samples.
            is_mpe: Whether to perform most probable explanation.
            cache: Optional cache for storing intermediate results.
            sampling_ctx: Sampling context for controlling sampling behavior.

        Returns:
            Tensor: Generated samples.
        """
        # Prepare data tensor

        sampling_ctx.validate_sampling_context(
            num_samples=data.shape[0],
            num_features=self.out_shape.features,
            num_channels=self.out_shape.channels,
            num_repetitions=self.out_shape.repetitions,
            allowed_feature_widths=(1, self.out_shape.features),
        )
        sampling_ctx.broadcast_feature_width(target_features=self.out_shape.features, allow_from_one=True)
        channel_index, mask = self._route_sampling(sampling_ctx=sampling_ctx)

        child_ctx = sampling_ctx.with_routing(
            channel_index=channel_index,
            mask=mask,
            clone_routing=False,
            clone_repetition=False,
        )

        self.inputs[0]._sample(
            data=data,
            cache=cache,
            sampling_ctx=child_ctx,
        )

        return data

    def _route_sampling(self, *, sampling_ctx: SamplingContext) -> tuple[Tensor, Tensor]:
        """Route sampling without batch-expanded selector tensors."""
        if sampling_ctx.is_differentiable:
            return self._route_sampling_differentiable(sampling_ctx=sampling_ctx)
        return self._route_sampling_integer(sampling_ctx=sampling_ctx)

    def _route_sampling_integer(self, *, sampling_ctx: SamplingContext) -> tuple[Tensor, Tensor]:
        """Gather parent routing using per-repetition output-owner indices."""
        owner_index = self._select_sampling_feature_owners(sampling_ctx=sampling_ctx)
        channel_index = torch.gather(sampling_ctx.channel_index, dim=1, index=owner_index)
        mask = torch.gather(sampling_ctx.mask, dim=1, index=owner_index)
        return channel_index, mask

    def _route_sampling_differentiable(self, *, sampling_ctx: SamplingContext) -> tuple[Tensor, Tensor]:
        """Fuse repetition selection with routing contraction for differentiable sampling."""
        sampling_indices = self.sampling_indices.to(
            device=sampling_ctx.channel_index.device,
            dtype=sampling_ctx.channel_index.dtype,
        )
        repetition_index = sampling_ctx.repetition_index.to(
            device=sampling_ctx.channel_index.device,
            dtype=sampling_ctx.channel_index.dtype,
        )
        channel_index = torch.einsum(
            "ior,boc,br->bic",
            sampling_indices,
            sampling_ctx.channel_index,
            repetition_index,
        )
        mask = (
            torch.einsum(
                "ior,bo,br->bi",
                sampling_indices,
                sampling_ctx.mask.to(dtype=sampling_indices.dtype),
                repetition_index,
            )
            > 0
        )
        return channel_index, mask

    def _select_sampling_feature_owners(self, sampling_ctx: SamplingContext) -> Tensor:
        """Select the per-input output owner for each sample repetition."""
        repetition_index = sampling_ctx.repetition_index
        if repetition_index.ndim == 2:
            repetition_index = repetition_index[:, 0]
        repetition_index = repetition_index.to(device=self.sampling_feature_owners.device, dtype=torch.long)
        owners = self.sampling_feature_owners.index_select(dim=1, index=repetition_index)
        return owners.transpose(0, 1).to(device=sampling_ctx.channel_index.device)

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Optional[Product | Module]:
        """Marginalize out specified random variables.

        Args:
            marg_rvs: List of random variable indices to marginalize.
            prune: Whether to prune redundant factorizations.
            cache: Optional cache for storing intermediate results.

        Returns:
            Optional[Product | Module]: Marginalized module or None if fully marginalized.
        """
        # initialize cache
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
        elif prune and marg_child.out_shape.features == 1:
            return marg_child
        else:
            return Factorize(
                inputs=[marg_child], depth=self.depth, num_repetitions=self.out_shape.repetitions
            )
