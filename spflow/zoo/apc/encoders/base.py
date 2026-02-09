"""Shared interfaces and latent-stat containers for APC encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import Tensor


@dataclass(frozen=True)
class LatentStats:
    """Latent distribution moments used in KL regularization.

    Attributes:
        mu: Approximate posterior mean, typically shaped ``(B, latent_dim)``.
        logvar: Approximate posterior log-variance with the same shape as ``mu``.
    """

    mu: Tensor
    logvar: Tensor


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
        """Return latent Gaussian statistics (mu, logvar) for x."""
