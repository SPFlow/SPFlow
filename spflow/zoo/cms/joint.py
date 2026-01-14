"""Wrapper that exposes joint log-likelihood as a single feature.

Some modules (e.g. multivariate leaves like :class:`~spflow.modules.leaves.CLTree`)
return a log-likelihood tensor with a feature axis, where the joint score is
obtained by summing across features.

This wrapper provides a consistent "root-like" view where
``log_likelihood(data)`` returns shape ``(batch, 1, channels, repetitions)``.

Notes:
    This wrapper performs a *tensor reduction* (sum over the feature axis) and is
    not meant to imply any additional probabilistic independence assumptions.
    In particular, it is not a "Product node" / factorization; it simply changes
    how the score is exposed. Sampling and marginalization are delegated to the
    wrapped module unchanged.
"""

from __future__ import annotations

import numpy as np
from torch import Tensor

from spflow.meta.data.scope import Scope
from spflow.modules.module_shape import ModuleShape
from spflow.modules.wrapper.base import Wrapper
from spflow.utils.cache import Cache, cached


class JointLogLikelihood(Wrapper):
    """Expose a wrapped module's joint log-likelihood as a single feature."""

    def __init__(self, module):
        super().__init__(module)
        self.out_shape = ModuleShape(1, module.out_shape.channels, module.out_shape.repetitions)

    @property
    def feature_to_scope(self) -> np.ndarray:
        out = []
        for r in range(self.out_shape.repetitions):
            joined = Scope.join_all(self.module.feature_to_scope[:, r])
            out.append(np.array([[joined]]))
        return np.concatenate(out, axis=1)

    @cached
    def log_likelihood(self, data: Tensor, cache: Cache | None = None) -> Tensor:
        ll = self.module.log_likelihood(data, cache=cache)
        return ll.sum(dim=1, keepdim=True)

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx=None,
    ) -> Tensor:
        return self.module.sample(
            num_samples=num_samples,
            data=data,
            is_mpe=is_mpe,
            cache=cache,
            sampling_ctx=sampling_ctx,
        )

    def marginalize(self, marg_rvs: list[int], prune: bool = True, cache: Cache | None = None):
        child = self.module.marginalize(marg_rvs=marg_rvs, prune=prune, cache=cache)
        if child is None:
            return None
        if child.out_shape.features == 1:
            return child
        return JointLogLikelihood(child)
