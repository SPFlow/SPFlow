"""Shared interfaces and latent-stat containers for APC encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import Tensor


@dataclass(frozen=True)
class LatentStats:
    """Latent posterior statistics used in APC training.

    Attributes:
        mu: Posterior mean-like moment, typically shaped ``(B, latent_dim)``.
        logvar: Posterior log-variance-like moment with same shape as ``mu``.
        kld_per_sample: Exact posterior KL to the fixed prior, shaped ``(B,)``.
        decode_latent: Deterministic latent for decode-time MPE behavior.
    """

    mu: Tensor
    logvar: Tensor
    kld_per_sample: Tensor
    decode_latent: Tensor


@runtime_checkable
class ApcEncoder(Protocol):
    """Protocol implemented by APC-compatible probabilistic-circuit encoders.

    Implementations expose posterior sampling (``encode``), evidence-conditioned
    reconstruction (``decode``), likelihood queries, and latent-moment extraction.
    """

    num_x_features: int
    latent_dim: int

    def encode(
        self,
        x: Tensor,
        *,
        mpe: bool = False,
        tau: float = 1.0,
        return_latent_stats: bool = False,
    ) -> Tensor | tuple[LatentStats, Tensor]:
        """Encode observations into latent samples z (shape: batch x latent_dim)."""

    def decode(
        self,
        z: Tensor,
        *,
        x: Tensor | None = None,
        mpe: bool = False,
        tau: float = 1.0,
        fill_evidence: bool = False,
    ) -> Tensor:
        """Decode latent values to data-space samples or reconstructions."""

    def joint_log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        """Compute log p(x, z) as a per-sample tensor (shape: batch,)."""

    def log_likelihood_x(self, x: Tensor) -> Tensor:
        """Compute log p(x) by marginalizing latent variables (shape: batch,)."""

    def sample_prior_z(self, num_samples: int, *, tau: float = 1.0) -> Tensor:
        """Sample latent variables from the encoder prior."""

    def latent_stats(self, x: Tensor, *, tau: float = 1.0) -> LatentStats:
        """Return latent posterior statistics for ``x``."""
