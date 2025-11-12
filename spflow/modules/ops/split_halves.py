from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from spflow.meta.data import Scope
from spflow.modules.module import Module
from spflow.modules.ops.split import Split
from spflow.utils.cache import Cache, init_cache


class SplitHalves(Split):  # ToDo: make abstract and implement concrete classes
    def __init__(
        self,
        inputs: Module,
        dim: int = 1,
        num_splits: int | None = 2,
        split_func: Callable[[torch.Tensor], list[torch.Tensor]] | None = None,
    ):
        """
        Split a single module along a given dimension. This implementation splits the features consecutively.
        Example:
            If num_splits=2, the features are split as follows:
            - Input features: [0, 1, 2, 3, 4, 5]
            - Split 0: features [0, 1, 2]
            - Split 1: features [3, 4, 5]
            If num_splits=3, the features are split as follows:
            - Input features: [0, 1, 2, 3, 4, 5]
            - Split 0: features [0, 1]
            - Split 1: features [2, 3]
            - Split 2: features [4, 5]


        Args:
            inputs:
            dim: Concatenation dimension. Note: dim=0: batch, dim=1: feature, dim=2: channel.
            num_splits: Number of splits along the given dimension.
        """
        super().__init__(inputs=inputs, dim=dim, num_splits=num_splits)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, dim={self.dim}"

    @property
    def feature_to_scope(self) -> list[Scope]:
        scopes = self.inputs[0].feature_to_scope
        num_scopes_per_chunk = len(scopes) // self.num_splits
        feature_to_scope = []
        for i in range(self.num_splits):
            sub_scopes = scopes[i * num_scopes_per_chunk : (i + 1) * num_scopes_per_chunk]
            feature_to_scope.append(sub_scopes)
        return feature_to_scope

    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> list[Tensor]:
        cache = init_cache(cache)
        log_cache = cache.setdefault("log_likelihood", {})

        # get log likelihoods for all inputs
        lls = self.inputs[0].log_likelihood(data, cache=cache)
        log_cache[self.inputs[0]] = lls

        lls_split = lls.split(self.inputs[0].out_features // self.num_splits, dim=self.dim)

        return list(lls_split)
