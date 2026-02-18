from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import torch
from einops import reduce
from torch import Tensor

from spflow.exceptions import ShapeError, UnsupportedOperationError
from spflow.meta.data.scope import Scope
from spflow.modules.module import Module
from spflow.modules.ops.cat import Cat
from spflow.modules.products.product import Product
from spflow.modules.sums.signed_sum import SignedSum
from spflow.modules.sums.sum import Sum
from spflow.utils.cache import Cache, cached
from spflow.utils.inner_product import inner_product_matrix, log_self_inner_product_scalar
from spflow.utils.sampling_context import SamplingContext, validate_sampling_context


def _is_signed_categorical(module: Module) -> bool:
    return module.__class__.__name__ == "SignedCategorical" and hasattr(module, "signed_logabs_and_sign")


def _contains_signed_sum(module: Module) -> bool:
    for m in module.modules():
        if isinstance(m, SignedSum) or _is_signed_categorical(cast(Module, m)):
            return True
    return False


def _signed_eval(module: Module, data: Tensor, cache: Cache) -> tuple[Tensor, Tensor]:
    """Evaluate `module` as a real function in (log|·|, sign) form.

    Returns:
        logabs, sign of shape (B, F, C, R).
    """
    cached = cache.get("signed_eval", module)
    if cached is not None:
        return cached

    if hasattr(module, "signed_logabs_and_sign"):
        out = module.signed_logabs_and_sign(data, cache=cache)  # type: ignore[attr-defined]
        cache.set("signed_eval", module, out)
        return out

    # Leaves and monotone internal modules: use log_likelihood as log-abs and sign=+1.
    if isinstance(module, (Sum,)):
        logv = module.log_likelihood(data, cache=cache)
        sign = torch.ones_like(logv, dtype=torch.int8)
        out = (logv, sign)
        cache.set("signed_eval", module, out)
        return out

    if isinstance(module, Cat):
        parts = [_signed_eval(cast(Module, child), data, cache) for child in module.inputs]
        logabs = torch.cat([p[0] for p in parts], dim=module.dim)
        sign = torch.cat([p[1] for p in parts], dim=module.dim)
        out = (logabs, sign)
        cache.set("signed_eval", module, out)
        return out

    if isinstance(module, Product):
        child_logabs, child_sign = _signed_eval(cast(Module, module.inputs), data, cache)
        # Multiply over features => add log-abs, multiply signs
        logabs = torch.sum(child_logabs, dim=1, keepdim=True)
        sign = torch.prod(child_sign.to(dtype=torch.int16), dim=1, keepdim=True).to(dtype=torch.int8)
        out = (logabs, sign)
        cache.set("signed_eval", module, out)
        return out

    # Default: non-negative module
    logv = module.log_likelihood(data, cache=cache)
    sign = torch.ones_like(logv, dtype=torch.int8)
    out = (logv, sign)
    cache.set("signed_eval", module, out)
    return out


class SOCS(Module):
    """Sum of Compatible Squares (SOCS) wrapper module.

    Represents a non-negative density of the form:

        c(x) = Σ_i c_i(x)^2
        p(x) = c(x) / Z, where Z = ∫ c(x) dx = Σ_i ∫ c_i(x)^2 dx

    Notes:
        - `log_likelihood()` is supported for signed components built with `SignedSum`.
        - `sample()` is supported only when all components are standard monotone SPFlow PCs
          (i.e., do not contain `SignedSum`), using a Metropolis–Hastings independence sampler.
    """

    def __init__(self, components: list[Module]) -> None:
        super().__init__()
        if len(components) < 1:
            raise ValueError("SOCS requires at least one component.")

        # Validate scope equality and compatible output shapes.
        scope = components[0].scope
        out_shape0 = components[0].out_shape
        for c in components:
            if not Scope.all_equal([scope, c.scope]):
                raise ShapeError("All SOCS components must have identical scope.")
            if tuple(c.out_shape) != tuple(out_shape0):
                raise ShapeError(
                    "All SOCS components must have identical out_shape; "
                    f"got {tuple(out_shape0)} vs {tuple(c.out_shape)}."
                )

        self.components = torch.nn.ModuleList(components)
        self.scope = scope
        self.in_shape = components[0].in_shape
        self.out_shape = components[0].out_shape

    @property
    def feature_to_scope(self) -> np.ndarray:
        return cast(Module, self.components[0]).feature_to_scope

    def _log_partition(self, cache: Cache) -> Tensor:
        """Compute log Z per output entry (shape: (F, C, R))."""
        cached = cache.get("socs_logZ", self)
        if cached is not None:
            # Keep a convenient handle for downstream inspection/debugging.
            cache.extras["socs_logZ"] = cached
            return cast(Tensor, cached)

        # Z[f,c,r] = Σ_i ∫ c_{i,f,c,r}(x)^2 dx;
        # each component contributes the diagonal of its self inner-product.
        z_parts = []
        for comp in self.components:
            k = inner_product_matrix(cast(Module, comp), cast(Module, comp), cache=cache)  # (F, C, C, R)
            diag = torch.diagonal(k, dim1=1, dim2=2)  # (F, R, C)
            z_parts.append(diag.permute(0, 2, 1))  # (F, C, R)

        Z = reduce(torch.stack(z_parts, dim=0), "n f c r -> f c r", "sum")
        logZ = torch.log(torch.clamp(Z, min=1e-30))
        cache.set("socs_logZ", self, logZ)
        cache.extras["socs_logZ"] = logZ
        return logZ

    @cached
    def log_likelihood(self, data: Tensor, cache: Cache | None = None) -> Tensor:  # type: ignore[override]
        # log c(x) = log Σ_i exp(2 log|c_i(x)|) (elementwise over output entries)
        comp_terms = []
        for comp in self.components:
            logabs, _sign = _signed_eval(cast(Module, comp), data, cache)
            comp_terms.append(2.0 * logabs)
        stacked = torch.stack(comp_terms, dim=0)  # (r, B, F, C, R)
        log_c = torch.logsumexp(stacked, dim=0)  # (B, F, C, R)

        logZ = self._log_partition(cache).to(dtype=log_c.dtype, device=log_c.device).unsqueeze(0)
        return log_c - logZ

    def _expectation_maximization_step(
        self,
        data: Tensor,
        bias_correction: bool = True,
        *,
        cache: Cache,
    ) -> None:
        raise UnsupportedOperationError("SOCS does not support expectation-maximization.")

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Module | None:
        # Marginalize each component and rebuild SOCS if possible.
        new_components: list[Module] = []
        for comp in self.components:
            m = cast(Module, comp).marginalize(marg_rvs, prune=prune, cache=cache)
            if m is None:
                return None
            new_components.append(m)

        if prune and len(new_components) == 1:
            # Keep SOCS wrapper; semantics differ from the raw component.
            return SOCS(new_components)
        return SOCS(new_components)

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
    ) -> Tensor:
        data = self._prepare_sample_data(num_samples=num_samples, data=data)

        if is_mpe:
            raise UnsupportedOperationError("SOCS.mpe() is not supported (use MAP on components if needed).")

        # Only unconditional sampling for now (all NaNs)
        if torch.isfinite(data).any():
            raise UnsupportedOperationError(
                "SOCS.sample() does not support conditional sampling with evidence yet."
            )

        if tuple(self.out_shape) != (1, 1, 1):
            raise UnsupportedOperationError(
                "SOCS.sample() currently supports only scalar-output circuits "
                "(out_shape.features==1, out_shape.channels==1, out_shape.repetitions==1)."
            )

        return super().sample(
            num_samples=None,
            data=data,
            is_mpe=is_mpe,
            cache=cache,
        )

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
        is_mpe: bool = False,
    ) -> Tensor:
        if is_mpe:
            raise UnsupportedOperationError("SOCS.mpe() is not supported (use MAP on components if needed).")

        # Only unconditional sampling for now (all NaNs)
        if torch.isfinite(data).any():
            raise UnsupportedOperationError(
                "SOCS.sample() does not support conditional sampling with evidence yet."
            )

        if tuple(self.out_shape) != (1, 1, 1):
            raise UnsupportedOperationError(
                "SOCS.sample() currently supports only scalar-output circuits "
                "(out_shape.features==1, out_shape.channels==1, out_shape.repetitions==1)."
            )

        num_samples = data.shape[0]

        # Mixture over components with weights proportional to Z_i
        logZs = torch.stack(
            [log_self_inner_product_scalar(cast(Module, c), cache=cache) for c in self.components]
        )
        comp_idx = torch.distributions.Categorical(logits=logZs).sample((num_samples,))

        # MCMC settings (can be overridden via cache.extras when cache is provided).
        cache_extras = cache.extras
        steps_after_burn_in = int(cache_extras.get("socs_mcmc_steps", cache_extras.get("socs_mh_steps", 50)))
        burn_in = int(cache_extras.get("socs_mcmc_burn_in", cache_extras.get("socs_mh_burn_in", 10)))
        if steps_after_burn_in < 1:
            raise ValueError("socs_mcmc_steps must be >= 1.")
        if burn_in < 0:
            raise ValueError("socs_mcmc_burn_in must be >= 0.")
        total_steps = burn_in + steps_after_burn_in

        def _joint_ll(mod: Module, x: Tensor) -> Tensor:
            # Do not reuse the traversal cache across different MCMC states.
            ll = mod.log_likelihood(x, cache=None)
            return reduce(ll, "b f 1 1 -> b", "sum")

        def _log_target_signed(mod: Module, x: Tensor) -> Tensor:
            # Same: the Cache implementation is per-module (not per-data), so it must
            # not be re-used when evaluating different x values in MCMC.
            eval_cache = Cache()
            logabs, _sign = _signed_eval(mod, x, eval_cache)
            return reduce(2.0 * logabs, "b f 1 1 -> b", "sum")

        # Sample per-component with an independence MH kernel:
        # target π(x) ∝ c_i(x)^2, proposal q(x) from a monotone PC (either c_i itself or abs-weight proxy).
        out = data.clone()
        for i, comp in enumerate(self.components):
            mask = comp_idx == i
            if not mask.any():
                continue
            n_i = int(mask.sum().item())
            comp_mod = cast(Module, comp)
            has_signed = _contains_signed_sum(comp_mod)
            if has_signed:
                # Local import to avoid a circular import: learn.build_socs -> SOCS.
                from spflow.learn.build_socs import build_abs_weight_proposal

                proposal = build_abs_weight_proposal(comp_mod)
            else:
                proposal = comp_mod

            x = proposal.sample(num_samples=n_i)
            log_q_x = _joint_ll(proposal, x)
            log_t_x = _log_target_signed(comp_mod, x)
            for _t in range(total_steps):
                x_p = proposal.sample(num_samples=n_i)
                log_q_p = _joint_ll(proposal, x_p)
                log_t_p = _log_target_signed(comp_mod, x_p)

                log_alpha = (log_t_p - log_t_x) + (log_q_x - log_q_p)
                u = torch.log(torch.rand_like(log_alpha))
                accept = u < torch.minimum(log_alpha, torch.zeros_like(log_alpha))
                x = torch.where(accept.unsqueeze(1), x_p, x)
                log_q_x = torch.where(accept, log_q_p, log_q_x)
                log_t_x = torch.where(accept, log_t_p, log_t_x)

            out[mask] = x

        return out
