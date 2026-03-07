from __future__ import annotations

import math
from collections.abc import Iterable, Iterator

import torch
from einops import rearrange, reduce
from torch import Tensor

from spflow.exceptions import InvalidParameterError, UnsupportedOperationError
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.bernoulli import Bernoulli
from spflow.modules.leaves.categorical import Categorical
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.module import Module


def as_scope(scope: Scope | int | Iterable[int] | None) -> Scope:
    """Normalize a scope-like argument to a Scope."""
    return Scope.as_scope(scope)


def reduce_log_likelihood(
    ll: Tensor,
    *,
    channel_agg: str,
    repetition_agg: str,
) -> Tensor:
    """Reduce a SPFlow log-likelihood tensor to per-sample scalar log-likelihoods.

    SPFlow modules typically return log-likelihoods shaped like:
        (batch, features, channels, repetitions)

    This function:
    - sums over features (log-space product),
    - aggregates repetitions and channels (mixture-like reduction), and
    - returns a 1D tensor of shape (batch,).

    Args:
        ll: Log-likelihood tensor.
        channel_agg: How to aggregate multiple channels ("logmeanexp", "logsumexp", "first").
        repetition_agg: How to aggregate multiple repetitions ("logmeanexp", "logsumexp", "first").

    Returns:
        Per-sample log-likelihood tensor of shape (batch,).
    """
    if ll.dim() == 2:
        ll = rearrange(ll, "b f -> b f 1 1")
    elif ll.dim() == 3:
        ll = rearrange(ll, "b f c -> b f c 1")
    elif ll.dim() != 4:
        raise InvalidParameterError(f"Unexpected log-likelihood shape {tuple(ll.shape)}.")

    if ll.shape[0] == 0:
        return ll.new_zeros((0,))

    ll = reduce(ll, "b f c r -> b c r", "sum")

    def reduce_over(t: Tensor, dim: int, method: str) -> Tensor:
        if t.shape[dim] == 1:
            return t.squeeze(dim)
        if method == "first":
            return t.select(dim, 0)
        if method == "logsumexp":
            return torch.logsumexp(t, dim=dim)
        if method == "logmeanexp":
            return torch.logsumexp(t, dim=dim) - math.log(t.shape[dim])
        raise InvalidParameterError(f"Unknown reduction method '{method}'.")

    ll = reduce_over(ll, dim=-1, method=repetition_agg)  # (B, C)
    ll = reduce_over(ll, dim=-1, method=channel_agg)  # (B,)
    return ll


def iter_modules(module: Module) -> Iterator[Module]:
    """Yield all modules in a probabilistic circuit (pre-order)."""
    yield module

    inputs = None
    if hasattr(module, "inputs"):
        try:
            inputs = module.inputs
        except AttributeError:
            inputs = None

    if inputs is None:
        return

    if isinstance(inputs, Module):
        yield from iter_modules(inputs)
        return

    if hasattr(inputs, "__iter__") and inputs.__class__.__name__ == "ModuleList":
        for child in inputs:
            yield from iter_modules(child)
        return

    if isinstance(inputs, (list, tuple)):
        for child in inputs:
            yield from iter_modules(child)
        return

    raise UnsupportedOperationError(
        f"Unsupported module 'inputs' container type {type(inputs)} for {module.__class__.__name__}."
    )


def infer_discrete_domains(
    model: Module,
    scope: Scope,
) -> dict[int, Tensor]:
    """Infer per-variable discrete domains for exact enumeration.

    Currently supports:
    - :class:`spflow.modules.leaves.Bernoulli` -> {0, 1}
    - :class:`spflow.modules.leaves.Categorical` -> {0, ..., K-1}

    Args:
        model: SPFlow probabilistic circuit.
        scope: Variables to infer domains for (query variables).

    Returns:
        Mapping from variable index to a 1D tensor containing all possible values.

    Raises:
        UnsupportedOperationError: If a required domain cannot be inferred or is inconsistent.
    """
    scope = as_scope(scope)
    target_rvs = set(scope.query)
    domains: dict[int, Tensor] = {}
    domain_sizes: dict[int, set[int]] = {rv: set() for rv in target_rvs}

    for module in iter_modules(model):
        if not isinstance(module, LeafModule):
            continue

        leaf_scope = module.scope
        rvs = set(leaf_scope.query).intersection(target_rvs)
        if not rvs:
            continue

        if isinstance(module, Bernoulli):
            values = torch.tensor([0.0, 1.0], dtype=torch.get_default_dtype(), device=module.device)
            for rv in rvs:
                domain_sizes[rv].add(2)
                domains.setdefault(rv, values)
            continue

        if isinstance(module, Categorical):
            k = int(module.K)
            if k < 1:
                raise UnsupportedOperationError(f"Categorical leaf has invalid K={k} for exact enumeration.")
            values = torch.arange(k, dtype=torch.get_default_dtype(), device=module.device)
            for rv in rvs:
                domain_sizes[rv].add(k)
                domains.setdefault(rv, values)
            continue

        raise UnsupportedOperationError(
            f"Exact enumeration supports only Bernoulli and Categorical leaves; got {module.__class__.__name__}."
        )

    missing = [rv for rv in target_rvs if rv not in domains]
    if missing:
        raise UnsupportedOperationError(
            f"Could not infer discrete domains for variables {sorted(missing)}; "
            "exact enumeration requires Bernoulli/Categorical leaves covering these RVs."
        )

    inconsistent = [rv for rv, sizes in domain_sizes.items() if len(sizes) > 1]
    if inconsistent:
        detail = {rv: sorted(domain_sizes[rv]) for rv in inconsistent}
        raise UnsupportedOperationError(f"Inconsistent inferred domain sizes: {detail}.")

    return domains


def fork_rng(seed: int | None, device: torch.device) -> torch.random.fork_rng:
    """Create an RNG fork context for deterministic sampling (best-effort)."""
    if seed is None:
        return torch.random.fork_rng(devices=[], enabled=False)

    devices: list[int] = []
    if device.type == "cuda":
        devices = [device.index] if device.index is not None else []

    ctx = torch.random.fork_rng(devices=devices, enabled=True)
    # Caller enters/exits context; we seed after entering.
    return ctx
