"""Shared APC joint-encoder utilities for Einet and Conv-PC backends."""

from __future__ import annotations

from torch import Tensor, nn
import torch

from spflow.exceptions import InvalidParameterError, ShapeError
from spflow.modules.leaves import Normal
from spflow.modules.leaves.leaf import LeafModule
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext
from spflow.zoo.apc.encoders.base import LatentStats
from spflow.zoo.apc.encoders.latent_stats import (
    find_unique_latent_record,
    latent_stats_from_leaf_record,
)


class JointPcEncoderBase(nn.Module):
    """Shared logic for APC encoders backed by a joint PC over ``[X, Z]``."""

    num_x_features: int
    latent_dim: int
    pc: nn.Module
    _x_cols: list[int]
    _z_cols: list[int]
    _z_leaf: LeafModule

    @staticmethod
    def _validate_leaf_scope(*, leaf: LeafModule, expected_scope: list[int], role: str) -> None:
        """Validate that a leaf covers exactly the expected variable scope."""
        if not isinstance(leaf, LeafModule):
            raise InvalidParameterError(
                f"{role}_leaf_factory must return LeafModule instances, got {type(leaf)}."
            )
        scope_query = list(leaf.scope.query)
        if set(scope_query) != set(expected_scope) or len(scope_query) != len(expected_scope):
            raise InvalidParameterError(
                f"{role}_leaf_factory returned scope {scope_query}, expected scope {expected_scope}."
            )

    @staticmethod
    def _evidence_dtype(*, x_flat: Tensor | None, z_flat: Tensor | None) -> torch.dtype:
        """Select an evidence dtype from provided tensors, falling back to default dtype."""
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
        """Build a joint evidence tensor over ``[X, Z]`` with ``NaN`` for missing blocks."""
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
                device = self.pc.device  # type: ignore[attr-defined]

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
        """Normalize PC log-likelihood outputs to shape ``(B,)``."""
        if ll.dim() < 1:
            raise ShapeError(f"Expected log-likelihood with batch dimension, got shape {tuple(ll.shape)}.")
        ll_flat = ll.reshape(ll.shape[0], -1)
        if ll_flat.shape[1] != 1:
            raise ShapeError(
                f"Expected scalar log-likelihood per sample, got trailing shape {tuple(ll_flat.shape[1:])}."
            )
        return ll_flat[:, 0]

    def _sampling_num_repetitions(self) -> int:
        """Resolve repetition width used by the top-down sampling context."""
        reps = getattr(self.pc, "num_repetitions", None)
        if reps is not None:
            return int(reps)
        out_shape = getattr(self.pc, "out_shape", None)
        if out_shape is not None and hasattr(out_shape, "repetitions"):
            return int(out_shape.repetitions)
        raise RuntimeError("Unable to determine PC repetition count for sampling context.")

    def _sample_joint(
        self,
        *,
        evidence: Tensor,
        mpe: bool,
        tau: float,
        return_sampling_ctx: bool,
    ) -> tuple[Tensor, SamplingContext | None]:
        """Sample the joint PC with differentiable routing semantics."""
        cache = Cache()
        # Populate evidence-conditioned likelihood cache before top-down sampling.
        self.pc.log_likelihood(evidence, cache=cache)  # type: ignore[attr-defined]
        batch_size = evidence.shape[0]
        channel_index = torch.ones(
            (batch_size, 1, 1), dtype=torch.get_default_dtype(), device=evidence.device
        )
        mask = torch.full((batch_size, 1), True, dtype=torch.bool, device=evidence.device)
        repetition_index = torch.zeros(
            (batch_size, self._sampling_num_repetitions()),
            dtype=torch.get_default_dtype(),
            device=evidence.device,
        )
        repetition_index[:, 0] = 1.0
        sampling_ctx: SamplingContext | None = SamplingContext(
            channel_index=channel_index,
            mask=mask,
            device=evidence.device,
            repetition_index=repetition_index,
            is_mpe=mpe,
            is_differentiable=True,
            tau=tau,
            return_leaf_params=return_sampling_ctx,
        )
        joint = self.pc._sample(data=evidence, cache=cache, sampling_ctx=sampling_ctx)  # type: ignore[attr-defined]
        if return_sampling_ctx:
            return joint, sampling_ctx
        return joint, None

    def _posterior_sample(self, x_flat: Tensor, *, mpe: bool, tau: float) -> Tensor:
        """Sample ``z ~ p(Z|X=x)``."""
        evidence = self._build_evidence(x_flat=x_flat, z_flat=None)
        joint, _ = self._sample_joint(
            evidence=evidence,
            mpe=mpe,
            tau=tau,
            return_sampling_ctx=False,
        )
        return joint[:, self._z_cols]

    def _posterior_sample_with_ctx(
        self,
        x_flat: Tensor,
        *,
        mpe: bool,
        tau: float,
    ) -> tuple[Tensor, SamplingContext]:
        """Sample ``z ~ p(Z|X=x)`` and return sampling context."""
        evidence = self._build_evidence(x_flat=x_flat, z_flat=None)
        joint, sampling_ctx = self._sample_joint(
            evidence=evidence,
            mpe=mpe,
            tau=tau,
            return_sampling_ctx=True,
        )
        if sampling_ctx is None:
            raise RuntimeError("Expected sampling context when return_sampling_ctx=True.")
        return joint[:, self._z_cols], sampling_ctx

    def _latent_stats_from_leaf_params(self, sampling_ctx: SamplingContext, batch_size: int) -> LatentStats:
        """Extract exact latent stats and KL from routed latent leaf parameters."""
        record = find_unique_latent_record(
            sampling_ctx,
            leaf_id=id(self._z_leaf),
            scope_cols=tuple(self._z_cols),
            batch_size=batch_size,
            latent_dim=self.latent_dim,
        )
        return latent_stats_from_leaf_record(
            leaf=self._z_leaf,
            record=record,
            batch_size=batch_size,
            latent_dim=self.latent_dim,
        )

    # Hooks implemented by concrete backends.
    def _flatten_x(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def _flatten_z(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def _reshape_x_like(self, x_flat: Tensor, x_like: Tensor | None) -> Tensor:
        return x_flat

    def encode(
        self,
        x: Tensor,
        *,
        mpe: bool = False,
        tau: float = 1.0,
        return_latent_stats: bool = False,
    ) -> Tensor | tuple[LatentStats, Tensor]:
        """Encode observations into latent samples."""
        x_flat = self._flatten_x(x)
        if not return_latent_stats:
            return self._posterior_sample(x_flat, mpe=mpe, tau=tau)

        z_first, sampling_ctx = self._posterior_sample_with_ctx(x_flat, mpe=mpe, tau=tau)
        stats = self._latent_stats_from_leaf_params(sampling_ctx=sampling_ctx, batch_size=z_first.shape[0])
        if mpe:
            z_samples = z_first
        elif isinstance(self._z_leaf, Normal):
            # Keep historical Normal behavior for stochastic encoding parity.
            z_samples = tau * torch.randn_like(stats.mu) * torch.exp(0.5 * stats.logvar) + stats.mu
        else:
            # For non-Normal leaves, expose traversal samples directly.
            z_samples = z_first
        return stats, z_samples

    def decode(
        self,
        z: Tensor,
        *,
        x: Tensor | None = None,
        mpe: bool = False,
        tau: float = 1.0,
        fill_evidence: bool = False,
    ) -> Tensor:
        """Decode latents by sampling/imputing the ``X`` block given ``Z`` evidence."""
        z_flat = self._flatten_z(z)
        x_flat = None if x is None else self._flatten_x(x)
        evidence = self._build_evidence(x_flat=x_flat, z_flat=z_flat)
        joint, _ = self._sample_joint(evidence=evidence, mpe=mpe, tau=tau, return_sampling_ctx=False)

        x_rec_flat = joint[:, self._x_cols]
        if fill_evidence and x_flat is not None:
            finite_mask = torch.isfinite(x_flat)
            x_rec_flat = torch.where(finite_mask, x_flat.to(x_rec_flat.dtype), x_rec_flat)
        return self._reshape_x_like(x_rec_flat, x)

    def joint_log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        """Compute per-sample joint log-likelihood ``log p(x, z)``."""
        x_flat = self._flatten_x(x)
        z_flat = self._flatten_z(z)
        evidence = self._build_evidence(x_flat=x_flat, z_flat=z_flat)
        ll = self.pc.log_likelihood(evidence)  # type: ignore[attr-defined]
        return self._flatten_ll(ll)

    def log_likelihood_x(self, x: Tensor) -> Tensor:
        """Compute per-sample marginal log-likelihood ``log p(x)``."""
        x_flat = self._flatten_x(x)
        evidence = self._build_evidence(x_flat=x_flat, z_flat=None)
        ll = self.pc.log_likelihood(evidence)  # type: ignore[attr-defined]
        return self._flatten_ll(ll)

    def sample_prior_z(self, num_samples: int, *, tau: float = 1.0) -> Tensor:
        """Sample latent variables from the model prior over ``Z``."""
        if num_samples <= 0:
            raise InvalidParameterError(f"num_samples must be >= 1, got {num_samples}.")
        evidence = self._build_evidence(
            x_flat=None, z_flat=None, num_samples=num_samples, device=self.pc.device  # type: ignore[attr-defined]
        )
        joint, _ = self._sample_joint(evidence=evidence, mpe=False, tau=tau, return_sampling_ctx=False)
        return joint[:, self._z_cols]

    def latent_stats(self, x: Tensor, *, tau: float = 1.0) -> LatentStats:
        """Return exact latent stats from routed latent leaf parameters."""
        x_flat = self._flatten_x(x)
        z_first, sampling_ctx = self._posterior_sample_with_ctx(x_flat, mpe=False, tau=tau)
        return self._latent_stats_from_leaf_params(sampling_ctx=sampling_ctx, batch_size=z_first.shape[0])
