from __future__ import annotations

from torch import Tensor

from spflow.meta.data import Scope
from spflow.modules.base import Module
from spflow.modules.ops.split import Split
from spflow.utils.cache import Cache, init_cache


class SplitHalves(Split):
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
        """Compute log likelihoods for split outputs.

        Args:
            data: Input data tensor.
            cache: Optional cache for storing intermediate computations.

        Returns:
            List of log likelihood tensors, one for each split output.
        """
        cache = init_cache(cache)
        log_cache = cache.setdefault("log_likelihood", {})

        # get log likelihoods for all inputs
        lls = self.inputs[0].log_likelihood(data, cache=cache)
        log_cache[self.inputs[0]] = lls

        lls_split = lls.split(self.inputs[0].out_features // self.num_splits, dim=self.dim)

        return list(lls_split)
