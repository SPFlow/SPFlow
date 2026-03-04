"""Latent-stat extraction helpers for APC encoders."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spflow.exceptions import ShapeError, UnsupportedOperationError
from spflow.modules.leaves import Bernoulli, Binomial, Categorical, Normal
from spflow.modules.leaves.leaf import LeafModule
from spflow.utils.sampling_context import LeafParamRecord, SamplingContext
from spflow.zoo.apc.encoders.base import LatentStats


def _expect_matrix(param: Tensor, *, name: str, batch_size: int, latent_dim: int) -> Tensor:
    if param.dim() != 2:
        raise ShapeError(
            f"Expected latent parameter '{name}' to have shape ({batch_size}, {latent_dim}), got {tuple(param.shape)}."
        )
    if tuple(param.shape) != (batch_size, latent_dim):
        raise ShapeError(
            f"Expected latent parameter '{name}' to have shape ({batch_size}, {latent_dim}), got {tuple(param.shape)}."
        )
    return param


def _probs_from_logits_or_probs(
    params: dict[str, Tensor],
    *,
    batch_size: int,
    latent_dim: int,
    name: str,
) -> Tensor:
    if "logits" in params:
        logits = params["logits"]
        logits = _expect_matrix(logits, name="logits", batch_size=batch_size, latent_dim=latent_dim)
        return torch.sigmoid(logits)
    if "probs" in params:
        probs = params["probs"]
        probs = _expect_matrix(probs, name="probs", batch_size=batch_size, latent_dim=latent_dim)
        return probs
    raise UnsupportedOperationError(f"{name} latent extraction requires 'logits' or 'probs' parameters.")


def _validate_record_for_latents(record: LeafParamRecord, *, batch_size: int, latent_dim: int) -> None:
    if record.active_mask.dim() != 2:
        raise ShapeError(f"Latent record active_mask must have rank 2, got rank {record.active_mask.dim()}.")
    if tuple(record.active_mask.shape) != (batch_size, latent_dim):
        raise ShapeError(
            f"Latent record active_mask shape mismatch: expected ({batch_size}, {latent_dim}), "
            f"got {tuple(record.active_mask.shape)}."
        )
    if not bool(record.active_mask.all()):
        raise UnsupportedOperationError(
            "Latent record contains inactive entries; APC latent stats require fully sampled latent scope."
        )


def find_unique_latent_record(
    sampling_ctx: SamplingContext,
    *,
    leaf_id: int,
    scope_cols: tuple[int, ...],
    batch_size: int,
    latent_dim: int,
) -> LeafParamRecord:
    """Find and validate a unique latent leaf-parameter record."""
    matches = [
        record
        for record in sampling_ctx.leaf_param_records_for(leaf_id)
        if tuple(record.scope_cols) == scope_cols
    ]
    if len(matches) != 1:
        raise ShapeError(
            f"Expected exactly one latent leaf-parameter record for scope {scope_cols}, found {len(matches)}."
        )
    record = matches[0]
    _validate_record_for_latents(record, batch_size=batch_size, latent_dim=latent_dim)
    return record


def latent_stats_from_leaf_record(
    *,
    leaf: LeafModule,
    record: LeafParamRecord,
    batch_size: int,
    latent_dim: int,
) -> LatentStats:
    """Compute exact latent stats and KL from a routed latent leaf-parameter record."""
    if isinstance(leaf, Normal):
        if "loc" not in record.params:
            raise UnsupportedOperationError("Normal latent extraction requires a 'loc' parameter.")
        mu = _expect_matrix(record.params["loc"], name="loc", batch_size=batch_size, latent_dim=latent_dim)
        if "log_scale" in record.params:
            log_scale = _expect_matrix(
                record.params["log_scale"], name="log_scale", batch_size=batch_size, latent_dim=latent_dim
            )
            logvar = 2.0 * log_scale
            sigma2 = torch.exp(logvar)
        elif "scale" in record.params:
            scale = _expect_matrix(
                record.params["scale"], name="scale", batch_size=batch_size, latent_dim=latent_dim
            )
            eps = torch.finfo(mu.dtype).eps
            sigma2 = scale.pow(2).clamp_min(eps)
            logvar = sigma2.log()
        else:
            raise UnsupportedOperationError(
                "Normal latent extraction requires 'scale' or 'log_scale' parameter."
            )
        kld_per_sample = 0.5 * (mu.pow(2) + sigma2 - 1.0 - logvar).sum(dim=1)
        return LatentStats(mu=mu, logvar=logvar, kld_per_sample=kld_per_sample, decode_latent=mu)

    if isinstance(leaf, Bernoulli):
        p = _probs_from_logits_or_probs(
            record.params,
            batch_size=batch_size,
            latent_dim=latent_dim,
            name="Bernoulli",
        )
        eps = torch.finfo(p.dtype).eps
        p = p.clamp(min=eps, max=1.0 - eps)
        one_minus_p = (1.0 - p).clamp(min=eps, max=1.0 - eps)
        log_half = math.log(0.5)
        kld_dim = p * (p.log() - log_half) + one_minus_p * (one_minus_p.log() - log_half)
        var = (p * one_minus_p).clamp_min(eps)
        decode = (p >= 0.5).to(dtype=p.dtype)
        return LatentStats(
            mu=p,
            logvar=var.log(),
            kld_per_sample=kld_dim.sum(dim=1),
            decode_latent=decode,
        )

    if isinstance(leaf, Binomial):
        if "total_count" not in record.params:
            raise UnsupportedOperationError("Binomial latent extraction requires 'total_count' parameter.")
        p = _probs_from_logits_or_probs(
            record.params,
            batch_size=batch_size,
            latent_dim=latent_dim,
            name="Binomial",
        )
        eps = torch.finfo(p.dtype).eps
        p = p.clamp(min=eps, max=1.0 - eps)
        one_minus_p = (1.0 - p).clamp(min=eps, max=1.0 - eps)
        total_count = _expect_matrix(
            record.params["total_count"],
            name="total_count",
            batch_size=batch_size,
            latent_dim=latent_dim,
        ).to(dtype=p.dtype)
        if bool((total_count < 0).any()):
            raise ShapeError("Binomial total_count must be non-negative.")
        log_half = math.log(0.5)
        bernoulli_kl = p * (p.log() - log_half) + one_minus_p * (one_minus_p.log() - log_half)
        mu = total_count * p
        var = (total_count * p * one_minus_p).clamp_min(eps)
        decode = torch.floor((total_count + 1.0) * p).clamp(min=0.0)
        decode = torch.minimum(decode, total_count)
        return LatentStats(
            mu=mu,
            logvar=var.log(),
            kld_per_sample=(total_count * bernoulli_kl).sum(dim=1),
            decode_latent=decode,
        )

    if isinstance(leaf, Categorical):
        if "logits" in record.params:
            logits = record.params["logits"]
            if logits.dim() != 3 or logits.shape[:2] != (batch_size, latent_dim):
                raise ShapeError(
                    "Categorical logits must have shape "
                    f"({batch_size}, {latent_dim}, K), got {tuple(logits.shape)}."
                )
            probs = torch.softmax(logits, dim=-1)
        elif "probs" in record.params:
            probs = record.params["probs"]
            if probs.dim() != 3 or probs.shape[:2] != (batch_size, latent_dim):
                raise ShapeError(
                    "Categorical probs must have shape "
                    f"({batch_size}, {latent_dim}, K), got {tuple(probs.shape)}."
                )
            eps = torch.finfo(probs.dtype).eps
            probs = probs.clamp_min(eps)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
        else:
            raise UnsupportedOperationError(
                "Categorical latent extraction requires 'logits' or 'probs' parameter."
            )

        num_classes = int(probs.shape[-1])
        if num_classes <= 0:
            raise ShapeError("Categorical latent extraction requires at least one class.")
        eps = torch.finfo(probs.dtype).eps
        probs = probs.clamp(min=eps, max=1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
        kld_dim = (probs * (probs.log() + math.log(float(num_classes)))).sum(dim=-1)

        k_values = torch.arange(num_classes, device=probs.device, dtype=probs.dtype)
        mu = (probs * k_values.view(1, 1, -1)).sum(dim=-1)
        second_moment = (probs * (k_values.pow(2).view(1, 1, -1))).sum(dim=-1)
        var = (second_moment - mu.pow(2)).clamp_min(eps)
        decode = torch.argmax(probs, dim=-1).to(dtype=probs.dtype)
        return LatentStats(
            mu=mu,
            logvar=var.log(),
            kld_per_sample=kld_dim.sum(dim=1),
            decode_latent=decode,
        )

    raise UnsupportedOperationError(
        f"Latent stats extraction is only supported for Normal, Bernoulli, Binomial, Categorical. Got {type(leaf)}."
    )
