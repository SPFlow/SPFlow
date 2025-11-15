"""Alternating splitting operation for tensor partitioning.

Distributes features in an alternating pattern across splits using modulo
arithmetic. Promotes feature diversity across branches. Used in RAT-SPN
and similar architectures.
"""

from __future__ import annotations

import torch
from torch import Tensor

from spflow.meta.data import Scope
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
    def feature_to_scope(self) -> list[Scope]:
        scopes = self.inputs[0].feature_to_scope
        feature_to_scope = []
        for i in range(self.num_splits):
            sub_scopes = scopes[i :: self.num_splits]
            feature_to_scope.append(sub_scopes)
        return feature_to_scope

    def _apply(self, fn):
        # Apply the function to the module and its split masks
        super()._apply(fn)
        self.split_masks = [fn(mask) for mask in self.split_masks]
        return self

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
        lls = self.inputs[0].log_likelihood(data, cache=cache)

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
