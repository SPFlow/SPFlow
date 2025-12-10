"""Alternating splitting operation for tensor partitioning.

Distributes features in an alternating pattern across splits using modulo
arithmetic. Promotes feature diversity across branches. Used in RAT-SPN
and similar architectures.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from spflow.modules.base import Module
from spflow.modules.ops.split import Split
from spflow.utils.cache import Cache, cached


class SplitAlternate(Split):
    """Split operation using alternating feature distribution.

    Distributes features using modulo arithmetic: feature i goes to split i % num_splits.
    Optimized for common cases (2 and 3 splits).

    Attributes:
        split_masks (list[Tensor]): Boolean masks for each split.
    """

    def __init__(self, inputs: Module, dim: int = 1, num_splits: int | None = 2):
        """Initialize alternating split operation.

        Args:
            inputs: Input module to split.
            dim: Dimension along which to split.
            num_splits: Number of parts to split into.
        """
        super().__init__(inputs=inputs, dim=dim, num_splits=num_splits)

        num_f = inputs.out_features
        indices = torch.arange(num_f, device=inputs.device) % num_splits

        # Create masks for each split
        self.split_masks = [indices == i for i in range(num_splits)]

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, dim={self.dim}"

    @property
    def feature_to_scope(self) -> np.ndarray:
        """
        Get feature-to-scope mapping for each split.

        Returns:
            np.ndarray: Array mapping features to scopes for each split.
                        Shape: (num_features_per_split, num_splits, num_repetitions)

        """
        scopes = self.inputs.feature_to_scope
        num_scopes_per_chunk = len(scopes) // self.num_splits
        out = []
        for r in range(self.num_repetitions):
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

        # get log likelihoods for all inputs
        lls = self.inputs.log_likelihood(data, cache=cache)

        # For computational speed up hard code the loglikelihoods for most common cases: Num splits = 2 and 3
        # For general cases, we use the split masks to get the log likelihoods for each split
        if self.num_splits == 1:
            return [lls]
        elif self.num_splits == 2:
            return [lls[:, 0::2, ...], lls[:, 1::2, ...]]
        elif self.num_splits == 3:
            return [lls[:, 0::3, ...], lls[:, 1::3, ...], lls[:, 2::3, ...]]
        else:
            return [lls[:, mask, ...] for mask in self.split_masks]
