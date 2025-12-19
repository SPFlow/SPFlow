"""Consecutive splitting operation for tensor partitioning.

Splits features into consecutive chunks. For example, [0,1,2,3] -> [0,1], [2,3].
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor

from spflow.modules.module import Module
from spflow.modules.ops.split import Split
from spflow.utils.cache import Cache, cached
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context


class SplitConsecutive(Split):
    """Split operation using consecutive feature distribution.

    Splits features into consecutive chunks: feature i goes to split i // (num_features / num_splits).

    Example:
        With num_splits=2: [0,1,2,3] -> [0,1], [2,3]
        With num_splits=3: [0,1,2,3,4,5] -> [0,1], [2,3], [4,5]
    """

    def __init__(
        self,
        inputs: Module,
        dim: int = 1,
        num_splits: int | None = 2,
    ):
        """Initialize consecutive split operation.

        Args:
            inputs: Input module to split.
            dim: Dimension along which to split (0=batch, 1=feature, 2=channel).
            num_splits: Number of splits along the given dimension.
        """
        super().__init__(inputs=inputs, dim=dim, num_splits=num_splits)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, dim={self.dim}"

    @property
    def feature_to_scope(self) -> np.ndarray:
        scopes = self.inputs.feature_to_scope
        num_scopes_per_chunk = len(scopes) // self.num_splits
        out = []
        for r in range(self.out_shape.repetitions):
            feature_to_scope_r = []
            for i in range(self.num_splits):
                sub_scopes_r = scopes[i * num_scopes_per_chunk : (i + 1) * num_scopes_per_chunk, r]
                feature_to_scope_r.append(sub_scopes_r)
            out.append(np.array(feature_to_scope_r).reshape(num_scopes_per_chunk, self.num_splits))

        out = np.stack(out, axis=2)
        return out

    @cached
    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> list[Tensor]:
        """Compute log likelihoods for split outputs.

        Args:
            data: Input data tensor.
            cache: Optional cache for storing intermediate computations.

        Returns:
            List of log likelihood tensors, one for each split output.
        """
        lls = self.inputs.log_likelihood(data, cache=cache)
        lls_split = lls.split(self.inputs.out_shape.features // self.num_splits, dim=self.dim)
        return list(lls_split)

    def merge_split_indices(self, *split_indices: Tensor) -> Tensor:
        """Merge split indices back to original layout (consecutive).

        SplitConsecutive splits features consecutively: [0,1,2,3] -> [0,1], [2,3].
        So we concatenate: left_indices, right_indices.
        """
        return torch.cat(split_indices, dim=1)

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Optional[Dict[str, Any]] = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        """Generate samples by delegating to input module.

        SplitConsecutive splits features consecutively: [0,1,2,3,4,5,6,7] ->
        left=[0,1,2,3], right=[4,5,6,7]. When sampling, we may need to expand
        the channel indices by repeating them for each split if they come from
        a parent that operates on the split output features.

        Args:
            num_samples: Number of samples to generate.
            data: Existing data tensor to modify.
            is_mpe: Whether to perform most probable explanation.
            cache: Cache dictionary for intermediate results.
            sampling_ctx: Sampling context for controlling sample generation.

        Returns:
            Tensor containing the generated samples.
        """
        data = self._prepare_sample_data(num_samples, data)
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0], data.device)

        input_features = self.inputs.out_shape.features
        split_features = input_features // self.num_splits

        if sampling_ctx.channel_index.shape[1] == split_features:
            channel_index = sampling_ctx.channel_index.repeat(1, self.num_splits)
            mask = sampling_ctx.mask.repeat(1, self.num_splits)
            sampling_ctx.update(channel_index=channel_index, mask=mask)
        elif sampling_ctx.channel_index.shape[1] == input_features:
            pass
        else:
            mask = sampling_ctx.mask.expand(data.shape[0], input_features)
            channel_index = sampling_ctx.channel_index.expand(data.shape[0], input_features)
            sampling_ctx.update(channel_index=channel_index, mask=mask)

        self.inputs.sample(
            data=data,
            is_mpe=is_mpe,
            cache=cache,
            sampling_ctx=sampling_ctx,
        )
        return data
