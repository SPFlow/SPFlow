from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterError, UnsupportedOperationError
from spflow.measures._utils import (
    as_scope,
    fork_rng,
    infer_discrete_domains,
    reduce_log_likelihood,
)
from spflow.meta.data.scope import Scope
from spflow.modules.module import Module


def entropy(
    model: Module,
    scope: Scope | int | Iterable[int],
    *,
    method: str = "mc",
    num_samples: int = 10_000,
    seed: int | None = None,
    channel_agg: str = "logmeanexp",
    repetition_agg: str = "logmeanexp",
) -> Tensor:
    """Estimate the entropy H(X) (in nats) for a subset of variables.

    The returned value is in nats (natural logarithm base), consistent with SPFlow
    log-likelihood conventions.

    Args:
        model: SPFlow probabilistic circuit.
        scope: Variables X to compute entropy for.
        method: "mc" (Monte Carlo) or "exact" (enumeration for tiny discrete domains).
        num_samples: Number of samples for Monte Carlo estimation.
        seed: Optional seed for best-effort deterministic sampling.
        channel_agg: How to aggregate multiple channels ("logmeanexp", "logsumexp", "first").
        repetition_agg: How to aggregate multiple repetitions ("logmeanexp", "logsumexp", "first").

    Returns:
        Scalar tensor containing H(X) in nats.
    """
    scope = as_scope(scope)
    if scope.empty():
        raise InvalidParameterError("entropy scope must be non-empty.")

    if method not in ("mc", "exact"):
        raise InvalidParameterError(f"Unknown method '{method}'. Use 'mc' or 'exact'.")

    if method == "exact":
        domains = infer_discrete_domains(model, scope)
        rvs = list(scope.query)
        values = [domains[rv] for rv in rvs]
        grid = torch.cartesian_prod(*values)  # (N, |rvs|)
        if grid.dim() == 1:
            grid = grid.unsqueeze(1)

        d = len(model.scope.query)
        data = torch.full((grid.shape[0], d), torch.nan, device=model.device, dtype=torch.get_default_dtype())
        for j, rv in enumerate(rvs):
            data[:, rv] = grid[:, j]

        ll = reduce_log_likelihood(
            model.log_likelihood(data),
            channel_agg=channel_agg,
            repetition_agg=repetition_agg,
        )  # (N,)

        mask = torch.isfinite(ll)
        p_log_p = torch.zeros_like(ll)
        p_log_p[mask] = torch.exp(ll[mask]) * ll[mask]
        return -p_log_p.sum()

    if num_samples < 1:
        raise InvalidParameterError("num_samples must be >= 1 for Monte Carlo entropy.")

    with fork_rng(seed, model.device) as _:
        if seed is not None:
            torch.manual_seed(seed)
        samples = model.sample(num_samples=num_samples)

    d = samples.shape[1]
    evidence = torch.full((num_samples, d), torch.nan, device=samples.device, dtype=samples.dtype)
    evidence[:, list(scope.query)] = samples[:, list(scope.query)]
    ll = reduce_log_likelihood(
        model.log_likelihood(evidence),
        channel_agg=channel_agg,
        repetition_agg=repetition_agg,
    )
    return -ll.mean()


def mutual_information(
    model: Module,
    x_scope: Scope | int | Iterable[int],
    y_scope: Scope | int | Iterable[int],
    *,
    method: str = "mc",
    num_samples: int = 10_000,
    seed: int | None = None,
    channel_agg: str = "logmeanexp",
    repetition_agg: str = "logmeanexp",
) -> Tensor:
    """Estimate mutual information I(X;Y) (in nats)."""
    x_scope = as_scope(x_scope)
    y_scope = as_scope(y_scope)
    if set(x_scope.query).intersection(y_scope.query):
        raise InvalidParameterError("x_scope and y_scope must be disjoint for mutual_information.")

    if method == "exact":
        h_x = entropy(
            model,
            x_scope,
            method="exact",
            channel_agg=channel_agg,
            repetition_agg=repetition_agg,
        )
        h_y = entropy(
            model,
            y_scope,
            method="exact",
            channel_agg=channel_agg,
            repetition_agg=repetition_agg,
        )
        h_xy = entropy(
            model,
            Scope(list(x_scope.query) + list(y_scope.query)),
            method="exact",
            channel_agg=channel_agg,
            repetition_agg=repetition_agg,
        )
        return h_x + h_y - h_xy

    if method != "mc":
        raise InvalidParameterError(f"Unknown method '{method}'. Use 'mc' or 'exact'.")

    if num_samples < 1:
        raise InvalidParameterError("num_samples must be >= 1 for Monte Carlo mutual_information.")

    with fork_rng(seed, model.device) as _:
        if seed is not None:
            torch.manual_seed(seed)
        samples = model.sample(num_samples=num_samples)

    d = samples.shape[1]
    x_rvs = list(x_scope.query)
    y_rvs = list(y_scope.query)
    xy_rvs = x_rvs + y_rvs

    def ll_for(rvs: list[int]) -> Tensor:
        ev = torch.full((num_samples, d), torch.nan, device=samples.device, dtype=samples.dtype)
        ev[:, rvs] = samples[:, rvs]
        return reduce_log_likelihood(
            model.log_likelihood(ev),
            channel_agg=channel_agg,
            repetition_agg=repetition_agg,
        )

    ll_xy = ll_for(xy_rvs)
    ll_x = ll_for(x_rvs)
    ll_y = ll_for(y_rvs)
    return (ll_xy - ll_x - ll_y).mean()


def conditional_mutual_information(
    model: Module,
    x_scope: Scope | int | Iterable[int],
    y_scope: Scope | int | Iterable[int],
    z_scope: Scope | int | Iterable[int],
    *,
    method: str = "mc",
    num_samples: int = 10_000,
    seed: int | None = None,
    channel_agg: str = "logmeanexp",
    repetition_agg: str = "logmeanexp",
) -> Tensor:
    """Estimate conditional mutual information I(X;Y|Z) (in nats)."""
    x_scope = as_scope(x_scope)
    y_scope = as_scope(y_scope)
    z_scope = as_scope(z_scope)

    all_rvs = list(x_scope.query) + list(y_scope.query) + list(z_scope.query)
    if len(set(all_rvs)) != len(all_rvs):
        raise InvalidParameterError("x_scope, y_scope, and z_scope must be pairwise disjoint.")

    if method == "exact":
        h_z = entropy(
            model,
            z_scope,
            method="exact",
            channel_agg=channel_agg,
            repetition_agg=repetition_agg,
        )
        h_xz = entropy(
            model,
            Scope(list(x_scope.query) + list(z_scope.query)),
            method="exact",
            channel_agg=channel_agg,
            repetition_agg=repetition_agg,
        )
        h_yz = entropy(
            model,
            Scope(list(y_scope.query) + list(z_scope.query)),
            method="exact",
            channel_agg=channel_agg,
            repetition_agg=repetition_agg,
        )
        h_xyz = entropy(
            model,
            Scope(list(x_scope.query) + list(y_scope.query) + list(z_scope.query)),
            method="exact",
            channel_agg=channel_agg,
            repetition_agg=repetition_agg,
        )
        return h_xz + h_yz - h_z - h_xyz

    if method != "mc":
        raise InvalidParameterError(f"Unknown method '{method}'. Use 'mc' or 'exact'.")

    if num_samples < 1:
        raise InvalidParameterError(
            "num_samples must be >= 1 for Monte Carlo conditional_mutual_information."
        )

    with fork_rng(seed, model.device) as _:
        if seed is not None:
            torch.manual_seed(seed)
        samples = model.sample(num_samples=num_samples)

    d = samples.shape[1]
    x_rvs = list(x_scope.query)
    y_rvs = list(y_scope.query)
    z_rvs = list(z_scope.query)
    xyz_rvs = x_rvs + y_rvs + z_rvs
    xz_rvs = x_rvs + z_rvs
    yz_rvs = y_rvs + z_rvs

    def ll_for(rvs: list[int]) -> Tensor:
        ev = torch.full((num_samples, d), torch.nan, device=samples.device, dtype=samples.dtype)
        ev[:, rvs] = samples[:, rvs]
        return reduce_log_likelihood(
            model.log_likelihood(ev),
            channel_agg=channel_agg,
            repetition_agg=repetition_agg,
        )

    ll_xyz = ll_for(xyz_rvs)
    ll_z = ll_for(z_rvs)
    ll_xz = ll_for(xz_rvs)
    ll_yz = ll_for(yz_rvs)

    # I(X;Y|Z) = E[log p(x,y,z) + log p(z) - log p(x,z) - log p(y,z)]
    return (ll_xyz + ll_z - ll_xz - ll_yz).mean()


__all__ = [
    "entropy",
    "mutual_information",
    "conditional_mutual_information",
]
