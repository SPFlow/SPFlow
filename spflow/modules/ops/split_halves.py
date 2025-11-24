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
        # Split operations don't change the feature-to-scope mapping,
        # just reorganize the channel structure. Delegate to input.
        return self.inputs[0].feature_to_scope
            

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
        lls = self.inputs[0].log_likelihood(data, cache=cache)

        lls_split = lls.split(self.inputs[0].out_features // self.num_splits, dim=self.dim)

        return list(lls_split)
