"""Einet-based APC joint encoder over data and latent variables."""

from __future__ import annotations

from typing import Callable, Literal

import torch
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterError, ShapeError
from spflow.modules.leaves import Normal
from spflow.modules.leaves.leaf import LeafModule
from spflow.zoo.apc.encoders.base import LatentStats
from spflow.zoo.einet import Einet

LeafFactory = Callable[[list[int], int, int], LeafModule]


def _default_normal_leaf(scope_indices: list[int], out_channels: int, num_repetitions: int) -> LeafModule:
    """Create a normal leaf over a scope block."""
    return Normal(
        scope=scope_indices,
        out_channels=out_channels,
        num_repetitions=num_repetitions,
    )


class EinetJointEncoder(nn.Module):
    """Joint APC encoder using an Einet over concatenated features [x, z]."""

    def __init__(
        self,
        *,
        num_x_features: int,
        latent_dim: int,
        num_sums: int = 10,
        num_leaves: int = 10,
        depth: int = 1,
        num_repetitions: int = 5,
        layer_type: Literal["einsum", "linsum"] = "linsum",
        structure: Literal["top-down", "bottom-up"] = "top-down",
        x_leaf_factory: LeafFactory | None = None,
        z_leaf_factory: LeafFactory | None = None,
        posterior_stat_samples: int = 4,
        posterior_var_floor: float = 1e-6,
    ) -> None:
        super().__init__()

        if num_x_features <= 0:
            raise InvalidParameterError(f"num_x_features must be >= 1, got {num_x_features}.")
        if latent_dim <= 0:
            raise InvalidParameterError(f"latent_dim must be >= 1, got {latent_dim}.")
        if posterior_stat_samples <= 0:
            raise InvalidParameterError(f"posterior_stat_samples must be >= 1, got {posterior_stat_samples}.")
        if posterior_var_floor <= 0.0:
            raise InvalidParameterError(f"posterior_var_floor must be > 0, got {posterior_var_floor}.")
        if structure != "top-down":
            raise InvalidParameterError(
                "EinetJointEncoder requires structure='top-down' because APC encoding/decoding uses sampling."
            )

        self.num_x_features = num_x_features
        self.latent_dim = latent_dim
        self.posterior_stat_samples = posterior_stat_samples
        self.posterior_var_floor = posterior_var_floor

        self._x_cols = list(range(num_x_features))
        self._z_cols = list(range(num_x_features, num_x_features + latent_dim))

        x_leaf_factory = x_leaf_factory or _default_normal_leaf
        z_leaf_factory = z_leaf_factory or _default_normal_leaf

        x_leaf = x_leaf_factory(self._x_cols, num_leaves, num_repetitions)
        self._validate_leaf_scope(leaf=x_leaf, expected_scope=self._x_cols, role="x")

        z_leaf = z_leaf_factory(self._z_cols, num_leaves, num_repetitions)
        self._validate_leaf_scope(leaf=z_leaf, expected_scope=self._z_cols, role="z")

        self.pc = Einet(
            leaf_modules=[x_leaf, z_leaf],
            num_classes=1,
            num_sums=num_sums,
            num_leaves=num_leaves,
            depth=depth,
            num_repetitions=num_repetitions,
            layer_type=layer_type,
            structure=structure,
        )

    @staticmethod
    def _validate_leaf_scope(*, leaf: LeafModule, expected_scope: list[int], role: str) -> None:
        if not isinstance(leaf, LeafModule):
            raise InvalidParameterError(
                f"{role}_leaf_factory must return LeafModule instances, got {type(leaf)}."
            )
        scope_query = list(leaf.scope.query)
        if set(scope_query) != set(expected_scope) or len(scope_query) != len(expected_scope):
            raise InvalidParameterError(
                f"{role}_leaf_factory returned scope {scope_query}, expected scope {expected_scope}."
            )

    def _flatten_x(self, x: Tensor) -> Tensor:
        if x.dim() < 2:
            raise ShapeError(f"x must have at least 2 dimensions, got shape {tuple(x.shape)}.")
        x_flat = x.reshape(x.shape[0], -1)
        if x_flat.shape[1] != self.num_x_features:
            raise ShapeError(
                f"Expected x to have {self.num_x_features} flattened features, got {x_flat.shape[1]}."
            )
        return x_flat

    def _flatten_z(self, z: Tensor) -> Tensor:
        if z.dim() < 2:
            raise ShapeError(f"z must have at least 2 dimensions, got shape {tuple(z.shape)}.")
        z_flat = z.reshape(z.shape[0], -1)
        if z_flat.shape[1] != self.latent_dim:
            raise ShapeError(f"Expected z to have latent_dim={self.latent_dim}, got {z_flat.shape[1]}.")
        return z_flat

    @staticmethod
    def _evidence_dtype(*, x_flat: Tensor | None, z_flat: Tensor | None) -> torch.dtype:
        if x_flat is not None and x_flat.is_floating_point():
            return x_flat.dtype
        if z_flat is not None and z_flat.is_floating_point():
            return z_flat.dtype
        return torch.get_default_dtype()

    def _build_evidence(
        self,
        *,
        x_flat: Tensor | None,
        z_flat: Tensor | None,
        num_samples: int | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        if x_flat is None and z_flat is None and num_samples is None:
            raise InvalidParameterError("num_samples must be provided when x_flat and z_flat are None.")

        inferred_batch = num_samples
        if x_flat is not None:
            inferred_batch = x_flat.shape[0]
        if z_flat is not None:
            if inferred_batch is None:
                inferred_batch = z_flat.shape[0]
            elif z_flat.shape[0] != inferred_batch:
                raise ShapeError(
                    f"x and z batch sizes must match, got {inferred_batch} and {z_flat.shape[0]}."
                )

        if inferred_batch is None:
            raise RuntimeError("Failed to infer batch size for evidence construction.")

        if device is None:
            if x_flat is not None:
                device = x_flat.device
            elif z_flat is not None:
                device = z_flat.device
            else:
                device = self.pc.device

        dtype = self._evidence_dtype(x_flat=x_flat, z_flat=z_flat)

        if x_flat is None:
            x_flat = torch.full((inferred_batch, self.num_x_features), torch.nan, device=device, dtype=dtype)
        else:
            x_flat = x_flat.to(device=device, dtype=dtype)

        if z_flat is None:
            z_flat = torch.full((inferred_batch, self.latent_dim), torch.nan, device=device, dtype=dtype)
        else:
            z_flat = z_flat.to(device=device, dtype=dtype)

        return torch.cat([x_flat, z_flat], dim=1)

    @staticmethod
    def _flatten_ll(ll: Tensor) -> Tensor:
        if ll.dim() < 1:
            raise ShapeError(f"Expected log-likelihood with batch dimension, got shape {tuple(ll.shape)}.")
        ll_flat = ll.reshape(ll.shape[0], -1)
        if ll_flat.shape[1] != 1:
            raise ShapeError(
                f"Expected scalar log-likelihood per sample, got trailing shape {tuple(ll_flat.shape[1:])}."
            )
        return ll_flat[:, 0]

    def _posterior_sample(self, x_flat: Tensor, *, mpe: bool, tau: float) -> Tensor:
        evidence = self._build_evidence(x_flat=x_flat, z_flat=None)
        if mpe:
            joint = self.pc.sample(data=evidence, is_mpe=True)
        else:
            joint = self.pc.rsample(data=evidence, is_mpe=False, tau=tau, hard=True)
        return joint[:, self._z_cols]

    def encode(
        self,
        x: Tensor,
        *,
        mpe: bool = False,
        tau: float = 1.0,
        return_latent_stats: bool = False,
    ) -> Tensor | tuple[LatentStats, Tensor]:
        x_flat = self._flatten_x(x)
        z = self._posterior_sample(x_flat, mpe=mpe, tau=tau)
        if return_latent_stats:
            return self.latent_stats(x_flat, tau=tau), z
        return z

    def decode(
        self,
        z: Tensor,
        *,
        x: Tensor | None = None,
        mpe: bool = False,
        tau: float = 1.0,
        fill_evidence: bool = False,
    ) -> Tensor:
        z_flat = self._flatten_z(z)
        x_flat = None if x is None else self._flatten_x(x)
        evidence = self._build_evidence(x_flat=x_flat, z_flat=z_flat)

        if mpe:
            joint = self.pc.sample(data=evidence, is_mpe=True)
        else:
            joint = self.pc.rsample(data=evidence, is_mpe=False, tau=tau, hard=True)

        x_rec = joint[:, self._x_cols]
        if fill_evidence and x_flat is not None:
            finite_mask = torch.isfinite(x_flat)
            x_rec = torch.where(finite_mask, x_flat.to(x_rec.dtype), x_rec)
        return x_rec

    def joint_log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        x_flat = self._flatten_x(x)
        z_flat = self._flatten_z(z)
        evidence = self._build_evidence(x_flat=x_flat, z_flat=z_flat)
        ll = self.pc.log_likelihood(evidence)
        return self._flatten_ll(ll)

    def log_likelihood_x(self, x: Tensor) -> Tensor:
        x_flat = self._flatten_x(x)
        evidence = self._build_evidence(x_flat=x_flat, z_flat=None)
        ll = self.pc.log_likelihood(evidence)
        return self._flatten_ll(ll)

    def sample_prior_z(self, num_samples: int, *, tau: float = 1.0) -> Tensor:
        if num_samples <= 0:
            raise InvalidParameterError(f"num_samples must be >= 1, got {num_samples}.")
        evidence = self._build_evidence(
            x_flat=None, z_flat=None, num_samples=num_samples, device=self.pc.device
        )
        joint = self.pc.rsample(
            data=evidence,
            is_mpe=False,
            tau=tau,
            hard=True,
        )
        return joint[:, self._z_cols]

    def latent_stats(self, x: Tensor, *, tau: float = 1.0) -> LatentStats:
        x_flat = self._flatten_x(x)

        samples = [
            self._posterior_sample(x_flat, mpe=False, tau=tau) for _ in range(self.posterior_stat_samples)
        ]
        stacked = torch.stack(samples, dim=0)  # (K, B, latent_dim)

        mu = stacked.mean(dim=0)
        var = stacked.var(dim=0, unbiased=False).clamp_min(self.posterior_var_floor)
        logvar = var.log()
        return LatentStats(mu=mu, logvar=logvar)
