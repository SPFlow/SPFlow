from __future__ import annotations

import torch
from torch import Tensor

from spflow.meta.data import Scope
from spflow.modules.base import Module
from spflow.modules.ops.split import Split
from spflow.utils.cache import Cache, init_cache


class SplitAlternate(Split):
    def __init__(self, inputs: Module, dim: int = 1, num_splits: int | None = 2):
        """
        Split a single module along a given dimension. This implementation splits the features in an alternating manner.
        Example:
            If num_splits=2, the features are split as follows:
            - Input features: [0, 1, 2, 3, 4, 5, ...]
            - Split 0: features 0, 2, 4, ...
            - Split 1: features 1, 3, 5, ...

            If num_splits=3, the features are split as follows:
            - Input features: [0, 1, 2, 3, 4, 5, ...]
            - Split 0: features 0, 3, 6, ...
            - Split 1: features 1, 4, 7, ...
            - Split 2: features 2, 5, 8, ...


        Args:
            inputs:
            dim: Concatenation dimension. Note: dim=0: batch, dim=1: feature, dim=2: channel.
            num_splits: Number of splits along the given dimension.
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

    def log_likelihood(self, data: Tensor, cache: Cache | None = None) -> list[Tensor]:
        cache = init_cache(cache)
        log_cache = cache.setdefault("log_likelihood", {})

        # get log likelihoods for all inputs
        lls = self.inputs[0].log_likelihood(data, cache=cache)
        log_cache[self.inputs[0]] = lls

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
