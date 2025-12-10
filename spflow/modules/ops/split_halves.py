from __future__ import annotations

import numpy as np
from torch import Tensor

from spflow.modules.base import Module
from spflow.modules.ops.split import Split
from spflow.utils.cache import Cache, cached


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
    def feature_to_scope(self) -> np.ndarray:
        scopes = self.inputs.feature_to_scope
        num_scopes_per_chunk = len(scopes) // self.num_splits
        out = []
        for r in range(self.num_repetitions):
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

        # get log likelihoods for all inputs
        lls = self.inputs.log_likelihood(data, cache=cache)

        lls_split = lls.split(self.inputs.out_features // self.num_splits, dim=self.dim)

        return list(lls_split)
