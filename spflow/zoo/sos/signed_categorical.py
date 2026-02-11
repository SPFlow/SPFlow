from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from einops import repeat
from torch import Tensor, nn

from spflow.exceptions import ShapeError, UnsupportedOperationError
from spflow.meta.data.scope import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.utils.cache import Cache


class SignedCategorical(Module):
    """Signed discrete leaf with unconstrained real-valued state weights.

    This module is intended for SOS-style non-monotonic components where the
    per-state values are not probabilities (they can be negative).

    For each scoped variable ``x`` in ``{0, ..., K-1}``, the leaf returns
    ``w[x]`` and therefore represents a real-valued function rather than a
    normalized distribution.
    """

    def __init__(
        self,
        scope: Scope | int | Iterable[int],
        out_channels: int = 1,
        num_repetitions: int = 1,
        K: int | None = None,
        weights: Tensor | None = None,
    ) -> None:
        super().__init__()

        if not isinstance(scope, Scope):
            scope = Scope(scope)
        self.scope = scope.copy()

        if K is None and weights is None:
            raise ValueError("Either 'K' or 'weights' must be provided for SignedCategorical.")

        if weights is not None:
            weights = torch.as_tensor(weights, dtype=torch.get_default_dtype())
            if weights.dim() != 4:
                raise ShapeError(
                    f"SignedCategorical weights must be 4D (F,C,R,K), got shape {tuple(weights.shape)}."
                )
            inferred_K = int(weights.shape[-1])
            K = inferred_K if K is None else K

        assert K is not None
        if K <= 0:
            raise ValueError(f"K must be >= 1, got {K}.")
        self.K = int(K)

        features = len(self.scope.query)
        self.in_shape = ModuleShape(features=features, channels=1, repetitions=1)
        self.out_shape = ModuleShape(features=features, channels=out_channels, repetitions=num_repetitions)

        self.weights_shape = (
            self.out_shape.features,
            self.out_shape.channels,
            self.out_shape.repetitions,
            self.K,
        )

        if weights is None:
            weights = 2.0 * torch.rand(self.weights_shape, dtype=torch.get_default_dtype()) - 1.0

        if tuple(weights.shape) != tuple(self.weights_shape):
            raise ShapeError(
                f"Invalid weights shape for SignedCategorical: got {tuple(weights.shape)}, "
                f"expected {self.weights_shape}."
            )

        self.weights = nn.Parameter(weights)

    @property
    def feature_to_scope(self) -> np.ndarray:
        scopes = np.empty((self.out_shape.features, self.out_shape.repetitions), dtype=object)
        for i, rv in enumerate(self.scope.query):
            for r in range(self.out_shape.repetitions):
                scopes[i, r] = Scope([rv])
        return scopes

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, K={self.K}, " f"weights={self.weights_shape}"

    def log_likelihood(self, data: Tensor, cache: Cache | None = None) -> Tensor:  # type: ignore[override]
        raise UnsupportedOperationError(
            "SignedCategorical does not define log_likelihood() because outputs may be negative. "
            "Use SOCS signed evaluation utilities."
        )

    def expectation_maximization(
        self,
        data: Tensor,
        bias_correction: bool = True,
        cache: Cache | None = None,
    ) -> None:
        raise UnsupportedOperationError("SignedCategorical does not support expectation-maximization.")

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Tensor | None = None,
        cache: Cache | None = None,
    ) -> None:
        raise UnsupportedOperationError("SignedCategorical does not support maximum-likelihood estimation.")

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Module | None:
        if set(self.scope.query).intersection(marg_rvs):
            return None
        return self

    def signed_logabs_and_sign(self, data: Tensor, cache: Cache | None = None) -> tuple[Tensor, Tensor]:
        """Evaluate this leaf in ``(log|.|, sign)`` form.

        Args:
            data: Tensor of shape (B, D) with integer-valued states in scoped columns.
            cache: Optional traversal cache.

        Returns:
            Tuple ``(logabs, sign)`` with shape ``(B, F, C, R)``.
        """
        if cache is None:
            cache = Cache()

        cached = cache.get("signed_logabs_and_sign", self)
        if cached is not None:
            return cached

        if data.dim() != 2:
            raise ShapeError(f"Expected data to be 2D (B,D), got shape {tuple(data.shape)}.")

        data_q = data[:, self.scope.query]
        if torch.isnan(data_q).any():
            raise UnsupportedOperationError(
                "SignedCategorical signed evaluation does not support NaN evidence."
            )

        idx = data_q.to(dtype=torch.long)
        if (idx < 0).any() or (idx >= self.K).any():
            raise ValueError(f"SignedCategorical state index out of bounds for K={self.K}.")

        batch_size = idx.shape[0]
        num_channels = self.out_shape.channels
        num_repetitions = self.out_shape.repetitions
        singleton_size = 1

        w = repeat(self.weights, "f c r k -> b f c r k", b=batch_size)  # (B,F,C,R,K)
        gather_idx = repeat(
            idx,
            "b f -> b f c r one",
            c=num_channels,
            r=num_repetitions,
            one=singleton_size,
        )

        vals = w.gather(dim=4, index=gather_idx).squeeze(4)  # (B,F,C,R)
        logabs = torch.log(torch.abs(vals).clamp_min(1e-30))
        sign = torch.sign(vals).to(dtype=torch.int8)

        out = (logabs, sign)
        cache.set("signed_logabs_and_sign", self, out)
        return out

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx=None,
    ) -> Tensor:
        raise UnsupportedOperationError(
            "SignedCategorical.sample() is not supported. "
            "Convert to a monotone proposal first (e.g., via build_abs_weight_proposal)."
        )
