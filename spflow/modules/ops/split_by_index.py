"""Index-based splitting operation for tensor partitioning.

Splits features according to user-specified indices. Allows full control
over which features go into which split.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from spflow.modules.module import Module
from spflow.modules.ops.split import Split
from spflow.utils.cache import Cache, cached
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context


class SplitByIndex(Split):
    """Split operation using explicit feature indices.

    Allows full control over which features go into which split by specifying
    exact indices for each split.

    Example:
        With indices=[[0, 1, 4], [2, 3, 5, 6, 7]]: features are split into
        group 1: [0, 1, 4] and group 2: [2, 3, 5, 6, 7]

    Attributes:
        indices: List of lists specifying feature indices for each split.
        inverse_indices: Tensor mapping original positions to split outputs.
    """

    def __init__(
        self,
        inputs: Module,
        indices: Sequence[Sequence[int]] | None = None,
        dim: int = 1,
    ):
        """Initialize index-based split operation.

        Args:
            inputs: Input module to split.
            indices: List of lists specifying feature indices for each split.
                Each inner list contains the feature indices for that split.
                All features must be covered exactly once (no overlap, no gaps).
            dim: Dimension along which to split (0=batch, 1=feature, 2=channel).

        Raises:
            ValueError: If indices are invalid (overlap, gaps, out of bounds).
        """
        if indices is None:
            raise ValueError("indices must be provided for SplitByIndex")

        num_splits = len(indices)
        super().__init__(inputs=inputs, dim=dim, num_splits=num_splits)

        # Convert to list of lists for consistency
        self._indices = [list(idx_group) for idx_group in indices]

        # Validate indices
        self._validate_indices()

        # Pre-compute inverse mapping for merge_split_indices
        # inverse_indices[i] = (split_idx, position_in_split)
        num_features = self.inputs.out_shape.features
        self._inverse_order = self._compute_inverse_order(num_features)

        # Create gather indices for log_likelihood
        self._gather_indices = [
            torch.tensor(idx_group, dtype=torch.long, device=inputs.device)
            for idx_group in self._indices
        ]

    def _validate_indices(self) -> None:
        """Validate that indices cover all features exactly once."""
        num_features = self.inputs.out_shape.features

        # Flatten all indices
        all_indices = []
        for idx_group in self._indices:
            all_indices.extend(idx_group)

        # Check for out of bounds
        for idx in all_indices:
            if idx < 0 or idx >= num_features:
                raise ValueError(
                    f"Index {idx} is out of bounds for input with {num_features} features."
                )

        # Check for duplicates (overlapping)
        if len(all_indices) != len(set(all_indices)):
            raise ValueError("Indices contain duplicates. Each feature must appear exactly once.")

        # Check all features are covered
        if set(all_indices) != set(range(num_features)):
            missing = set(range(num_features)) - set(all_indices)
            raise ValueError(f"Indices do not cover all features. Missing: {missing}")

    def _compute_inverse_order(self, num_features: int) -> Tensor:
        """Compute inverse mapping from split outputs back to original order.

        Returns:
            Tensor of shape (num_features,) where inverse_order[i] gives the
            position in the concatenated split outputs that corresponds to
            original feature i.
        """
        inverse_order = torch.zeros(num_features, dtype=torch.long)
        offset = 0
        for idx_group in self._indices:
            for pos, orig_idx in enumerate(idx_group):
                inverse_order[orig_idx] = offset + pos
            offset += len(idx_group)
        return inverse_order

    @property
    def indices(self) -> list[list[int]]:
        """Get the feature indices for each split."""
        return self._indices

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, dim={self.dim}, indices={self._indices}"

    @property
    def feature_to_scope(self) -> np.ndarray:
        """Get feature-to-scope mapping for each split.

        Returns:
            np.ndarray: Array mapping features to scopes for each split.
                        Shape: (num_features_per_split, num_splits, num_repetitions)
        """
        scopes = self.inputs.feature_to_scope
        max_split_size = max(len(idx_group) for idx_group in self._indices)
        out = []

        for r in range(self.out_shape.repetitions):
            feature_to_scope_r = []
            for idx_group in self._indices:
                # Gather scopes for this split's indices
                sub_scopes_r = scopes[idx_group, r]
                # Pad if necessary to match max_split_size
                if len(sub_scopes_r) < max_split_size:
                    padding = np.full(max_split_size - len(sub_scopes_r), -1)
                    sub_scopes_r = np.concatenate([sub_scopes_r, padding])
                feature_to_scope_r.append(sub_scopes_r)

            out.append(np.array(feature_to_scope_r).T)  # Transpose to (features, splits)

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

        # Gather features for each split
        result = []
        for gather_idx in self._gather_indices:
            # Move gather indices to same device as lls
            if gather_idx.device != lls.device:
                gather_idx = gather_idx.to(lls.device)
            # Index along feature dimension (dim=1)
            split_lls = lls.index_select(dim=1, index=gather_idx)
            result.append(split_lls)

        return result

    def merge_split_indices(self, *split_indices: Tensor) -> Tensor:
        """Merge split indices back to original layout.

        Takes channel indices for each split and combines them into
        indices matching the original (unsplit) feature layout.

        Args:
            *split_indices: Channel index tensors for each split.

        Returns:
            Merged indices matching the input module's feature layout.
        """
        # Concatenate all split indices
        concat_indices = torch.cat(split_indices, dim=1)

        # Reorder to original feature order using inverse mapping
        inverse_order = self._inverse_order.to(concat_indices.device)

        # Create output tensor
        batch_size = concat_indices.shape[0]
        num_features = self.inputs.out_shape.features
        result = torch.zeros(batch_size, num_features, dtype=concat_indices.dtype, device=concat_indices.device)

        # Scatter back to original positions
        for split_idx, idx_group in enumerate(self._indices):
            offset = sum(len(self._indices[i]) for i in range(split_idx))
            for pos, orig_idx in enumerate(idx_group):
                result[:, orig_idx] = concat_indices[:, offset + pos]

        return result

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Optional[Dict[str, Any]] = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        """Generate samples by delegating to input module.

        SplitByIndex may receive channel indices for split features that need
        to be expanded to the full input feature count.

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

        # Check if we need to expand the sampling context
        ctx_features = sampling_ctx.channel_index.shape[1]

        if ctx_features == input_features:
            # Already full size, no expansion needed
            pass
        else:
            # Need to expand - assume context matches first split size and repeat
            # This handles the case when parent module operates on split outputs
            first_split_size = len(self._indices[0])
            if ctx_features == first_split_size:
                # Repeat for each split (assuming equal split sizes for simplicity)
                channel_index = sampling_ctx.channel_index.repeat(1, self.num_splits)
                mask = sampling_ctx.mask.repeat(1, self.num_splits)

                # Truncate if we repeated too much
                if channel_index.shape[1] > input_features:
                    channel_index = channel_index[:, :input_features]
                    mask = mask[:, :input_features]

                sampling_ctx.update(channel_index=channel_index, mask=mask)
            else:
                # Expand to full size
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
