"""Interleaved splitting operation for tensor partitioning.

Distributes features in an interleaved pattern across splits using modulo
arithmetic. Promotes feature diversity across branches. Used in RAT-SPN
and similar architectures.
"""

from __future__ import annotations

import numpy as np
import torch
from einops import rearrange
from torch import Tensor

from spflow.modules.module import Module
from spflow.modules.ops.split import Split
from spflow.utils.cache import Cache, cached
from spflow.utils.sampling_context import (
    SamplingContext,
    validate_sampling_context,
)


class SplitInterleaved(Split):
    """Split operation using interleaved feature distribution.

    Distributes features using modulo arithmetic: feature i goes to split i % num_splits.
    Optimized for common cases (2 and 3 splits).

    Example:
        With num_splits=2: [0,1,2,3] -> [0,2], [1,3]
        With num_splits=3: [0,1,2,3,4,5] -> [0,3], [1,4], [2,5]

    Attributes:
        split_masks (list[Tensor]): Boolean masks for each split.
    """

    def __init__(self, inputs: Module, dim: int = 1, num_splits: int | None = 2):
        """Initialize interleaved split operation.

        Args:
            inputs: Input module to split.
            dim: Dimension along which to split.
            num_splits: Number of parts to split into.
        """
        super().__init__(inputs=inputs, dim=dim, num_splits=num_splits)

        num_f = inputs.out_shape.features
        indices = torch.arange(num_f, device=inputs.device) % num_splits

        # Create masks for each split
        self.split_masks = [indices == i for i in range(num_splits)]

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, dim={self.dim}"

    @property
    def feature_to_scope(self) -> np.ndarray:
        """Get feature-to-scope mapping for each split.

        Returns:
            np.ndarray: Array mapping features to scopes for each split.
                        Shape: (num_features_per_split, num_splits, num_repetitions)
        """
        scopes = self.inputs.feature_to_scope
        num_scopes_per_chunk = len(scopes) // self.num_splits
        out = []
        for r in range(self.out_shape.repetitions):
            feature_to_scope_r = []
            for i in range(self.num_splits):
                sub_scopes_r = scopes[i :: self.num_splits, r]
                feature_to_scope_r.append(sub_scopes_r)

            out.append(np.array(feature_to_scope_r).reshape(num_scopes_per_chunk, self.num_splits))

        out = np.stack(out, axis=2)
        return out

    @cached
    def log_likelihood(self, data: Tensor, cache: Cache | None = None) -> list[Tensor]:
        """Compute log likelihoods for each split.

        Args:
            data: Input data tensor.
            cache: Optional cache for storing intermediate results.

        Returns:
            List of log likelihood tensors, one for each split.
        """
        lls = self.inputs.log_likelihood(data, cache=cache)

        # Optimized for common cases
        if self.num_splits == 1:
            return [lls]
        elif self.num_splits == 2:
            return [lls[:, 0::2, ...], lls[:, 1::2, ...]]
        elif self.num_splits == 3:
            return [lls[:, 0::3, ...], lls[:, 1::3, ...], lls[:, 2::3, ...]]
        else:
            return [lls[:, mask, ...] for mask in self.split_masks]

    def merge_split_indices(self, *split_indices: Tensor) -> Tensor:
        """Merge split indices back to original layout (interleaved).

        SplitInterleaved splits features by modulo: [0,1,2,3] -> [0,2], [1,3].
        So we interleave: [left[0], right[0], left[1], right[1], ...].
        """
        stacked = torch.stack(split_indices, dim=2)  # (batch, features_per_split, num_splits)
        return rearrange(stacked, "b f split -> b (f split)")

    def merge_split_tensors(self, *split_tensors: Tensor) -> Tensor:
        """Merge split feature tensors back to original layout (interleaved)."""
        stacked = torch.stack(split_tensors, dim=2)  # (batch, features_per_split, num_splits, ...)
        return rearrange(stacked, "b f split ... -> b (f split) ...")

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
    ) -> Tensor:
        input_features = self.inputs.out_shape.features
        split_features = input_features // self.num_splits
        validate_sampling_context(
            sampling_ctx,
            num_samples=data.shape[0],
            num_features=input_features,
            num_channels=self.out_shape.channels,
            num_repetitions=self.out_shape.repetitions,
            allowed_feature_widths=(1, input_features, split_features),
        )

        before_features = int(sampling_ctx.channel_index.shape[1])
        if before_features == 1 and input_features > 1:
            sampling_ctx.broadcast_feature_width(target_features=input_features, allow_from_one=True)
            before_features = input_features
        sampling_ctx.repeat_split_feature_width(
            num_splits=self.num_splits,
            target_features=input_features,
        )

        # Split-sized adaptation first repeats by split, then interleaves into input order.
        if before_features != input_features:
            channel_index = rearrange(
                sampling_ctx.channel_index,
                "b (split f) -> b (f split)",
                split=self.num_splits,
            )
            mask = rearrange(
                sampling_ctx.mask,
                "b (split f) -> b (f split)",
                split=self.num_splits,
            )
            sampling_ctx.channel_index = channel_index
            sampling_ctx.mask = mask

        self.inputs._sample(
            data=data,
            cache=cache,
            sampling_ctx=sampling_ctx,
        )
        return data
