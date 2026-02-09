from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import Tensor

from spflow.exceptions import ShapeError, UnsupportedOperationError
from spflow.modules.module import Module
from spflow.utils.inner_product import triple_product_scalar
from spflow.utils.cache import Cache
from spflow.modules.sos.socs import _signed_eval
from spflow.utils.sampling_context import SamplingContext


class ExpSOCS(Module):
    """Product of a monotone PC with a SOCS (µSOCS / ExpSOS).

    This implements the model used in the reference implementation (`ExpSOS`) and
    Definition 5 (µSOCS) in the SOS/SOCS paper, specialized to real-valued circuits:

        c(x) = m(x) * Σ_i c_i(x)^2
        p(x) = c(x) / Z,   Z = ∫ m(x) * Σ_i c_i(x)^2 dx

    where `m` is a monotone (non-negative) probabilistic circuit and each `c_i` may
    contain signed parameters (e.g., `SignedSum`).

    Notes:
        - Currently supports only scalar-output circuits (out_shape == (1,1,1)) for
          both the monotone circuit and all components.
        - Exact normalization uses a triple-product dynamic program
          `∫ c_i(x) c_i(x) m(x) dx`.
        - Sampling is not implemented.
    """

    def __init__(self, *, monotone: Module, components: list[Module]) -> None:
        super().__init__()
        if len(components) < 1:
            raise ValueError("ExpSOCS requires at least one component.")

        if tuple(monotone.out_shape) != (1, 1, 1):
            raise ShapeError(
                "ExpSOCS currently requires monotone.out_shape == (1,1,1); "
                f"got {tuple(monotone.out_shape)}."
            )

        for c in components:
            if tuple(c.out_shape) != (1, 1, 1):
                raise ShapeError(
                    "ExpSOCS currently requires component.out_shape == (1,1,1); " f"got {tuple(c.out_shape)}."
                )
            if c.scope != monotone.scope:
                raise ShapeError("ExpSOCS requires monotone and all components to have identical scope.")

        self.monotone = monotone
        self.components = torch.nn.ModuleList(components)
        self.scope = monotone.scope
        self.in_shape = monotone.in_shape
        self.out_shape = monotone.out_shape

    @property
    def feature_to_scope(self) -> np.ndarray:
        return cast(Module, self.monotone).feature_to_scope

    def _log_partition(self, cache: Cache) -> Tensor:
        cached = cache.get("exp_socs_logZ", self)
        if cached is not None:
            cache.extras["exp_socs_logZ"] = cached
            return cast(Tensor, cached)

        z_parts = []
        for comp in self.components:
            z_parts.append(
                triple_product_scalar(cast(Module, comp), cast(Module, comp), self.monotone, cache=cache)
            )
        Z = torch.stack(z_parts).sum()
        Z = torch.clamp(Z, min=0.0)
        logZ = torch.log(Z.clamp_min(1e-30))

        cache.set("exp_socs_logZ", self, logZ)
        cache.extras["exp_socs_logZ"] = logZ
        return logZ

    def log_likelihood(self, data: Tensor, cache: Cache | None = None) -> Tensor:  # type: ignore[override]
        if cache is None:
            cache = Cache()

        log_m = self.monotone.log_likelihood(data, cache=cache)  # (B,1,1,1)

        comp_terms = []
        for comp in self.components:
            logabs, _sign = _signed_eval(cast(Module, comp), data, cache)
            comp_terms.append(2.0 * logabs)
        stacked = torch.stack(comp_terms, dim=0)  # (r, B, 1, 1, 1)
        log_c2 = torch.logsumexp(stacked, dim=0)  # (B,1,1,1)

        logZ = self._log_partition(cache).to(dtype=log_c2.dtype, device=log_c2.device)
        return log_m + log_c2 - logZ

    def expectation_maximization(
        self,
        data: Tensor,
        bias_correction: bool = True,
        cache: Cache | None = None,
    ) -> None:
        raise UnsupportedOperationError("ExpSOCS does not support expectation-maximization.")

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Tensor | None = None,
        cache: Cache | None = None,
    ) -> None:
        raise UnsupportedOperationError("ExpSOCS does not support maximum-likelihood estimation.")

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Module | None:
        mono = self.monotone.marginalize(marg_rvs, prune=prune, cache=cache)
        if mono is None:
            return None

        comps: list[Module] = []
        for comp in self.components:
            m = cast(Module, comp).marginalize(marg_rvs, prune=prune, cache=cache)
            if m is None:
                return None
            comps.append(m)

        return ExpSOCS(monotone=cast(Module, mono), components=comps)

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        raise UnsupportedOperationError("ExpSOCS.sample() is not supported.")
