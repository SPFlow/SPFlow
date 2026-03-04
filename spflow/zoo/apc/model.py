"""High-level APC model orchestration.

This module combines an APC encoder (tractable probabilistic circuit over ``X,Z``)
with an optional neural decoder and exposes the paper-style composite objective:

``total = w_rec * rec + w_kld * kld + w_nll * nll``.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from spflow.exceptions import InvalidParameterError, UnsupportedOperationError
from spflow.zoo.apc.debug_trace import trace_tensor
from spflow.zoo.apc.config import ApcConfig
from spflow.zoo.apc.encoders.base import ApcEncoder, LatentStats


class AutoencodingPC(nn.Module):
    """APC model combining a tractable encoder and an optional decoder.

    If ``decoder`` is ``None``, decoding is delegated to the encoder's
    evidence-conditioned ``decode`` method.
    """

    def __init__(
        self,
        encoder: ApcEncoder,
        decoder: nn.Module | None,
        config: ApcConfig,
    ) -> None:
        """Initialize an APC model.

        Args:
            encoder: APC-compatible encoder implementation.
            decoder: Optional neural decoder mapping ``z -> x``.
            config: APC model and loss configuration.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

    def encode(self, x: Tensor, *, mpe: bool = False, tau: float | None = None) -> Tensor:
        """Encode observed data into latent samples.

        Args:
            x: Observation tensor.
            mpe: Whether to use deterministic MPE routing.
            tau: Optional sampling temperature override.

        Returns:
            Latent samples ``z``.
        """
        trace_tensor("apc.encode.x_in", x)
        tau_eff = self.config.sample_tau if tau is None else tau
        z = self.encoder.encode(x, mpe=mpe, tau=tau_eff)  # type: ignore[assignment]
        if isinstance(z, Tensor):
            trace_tensor("apc.encode.z_samples", z)
        return z  # type: ignore[return-value]

    def decode(self, z: Tensor, *, mpe: bool = False, tau: float | None = None) -> Tensor:
        """Decode latents into reconstructions/samples in data space.

        Args:
            z: Latent samples.
            mpe: Whether to use deterministic MPE routing when using encoder decode.
            tau: Optional sampling temperature override.

        Returns:
            Reconstructed/sample ``x`` tensor. Any output scaling is
            caller-managed; this method does not apply range transforms.
        """
        trace_tensor("apc.decode.z_in", z)
        tau_eff = self.config.sample_tau if tau is None else tau
        if self.decoder is None:
            x_rec = self.encoder.decode(z, mpe=mpe, tau=tau_eff)
            trace_tensor("apc.decode.x_rec", x_rec)
            return x_rec
        x_rec = self.decoder(z)
        trace_tensor("apc.decode.x_rec", x_rec)
        return x_rec

    def reconstruct(self, x: Tensor, *, mpe: bool = False, tau: float | None = None) -> Tensor:
        """Reconstruct ``x`` by encoding to ``z`` and decoding back to data space."""
        z = self.encode(x, mpe=mpe, tau=tau)
        return self.decode(z, mpe=mpe, tau=tau)

    def sample_x(self, num_samples: int, *, tau: float | None = None) -> Tensor:
        """Sample synthetic observations by sampling ``z`` and decoding."""
        z = self.sample_z(num_samples=num_samples, tau=tau)
        return self.decode(z, mpe=False, tau=tau)

    def sample_z(self, num_samples: int, *, tau: float | None = None) -> Tensor:
        """Sample latents from the encoder prior."""
        tau_eff = self.config.sample_tau if tau is None else tau
        return self.encoder.sample_prior_z(num_samples=num_samples, tau=tau_eff)

    @staticmethod
    def _flatten_tensor(tensor: Tensor) -> Tensor:
        """Flatten all non-batch axes into a single feature axis."""
        if tensor.dim() < 2:
            raise InvalidParameterError(
                f"Expected tensor with batch dimension and at least one feature axis, got shape {tuple(tensor.shape)}."
            )
        return rearrange(tensor, "b ... -> b (...)")

    def _reconstruction_loss(self, x: Tensor, x_rec: Tensor) -> Tensor:
        """Compute reconstruction loss with feature-sum / batch-mean reduction.

        Input preprocessing is intentionally caller-controlled. This method
        compares ``x`` and ``x_rec`` directly.
        """
        x_flat = self._flatten_tensor(x)
        x_rec_flat = self._flatten_tensor(x_rec)

        if x_flat.shape != x_rec_flat.shape:
            raise InvalidParameterError(
                f"Reconstruction shape mismatch: x has {tuple(x_flat.shape)}, x_rec has {tuple(x_rec_flat.shape)}."
            )
        batch_size = x_flat.shape[0]

        if self.config.rec_loss == "mse":
            return F.mse_loss(x_rec_flat, x_flat, reduction="sum") / batch_size
        if self.config.rec_loss == "bce":
            return F.binary_cross_entropy(x_rec_flat, x_flat, reduction="sum") / batch_size
        raise InvalidParameterError(f"Unsupported rec_loss '{self.config.rec_loss}'.")

    @staticmethod
    def _kld_from_stats(stats: LatentStats) -> Tensor:
        """Compute mean KL divergence from exact encoder-provided per-sample KL."""
        if not isinstance(stats.kld_per_sample, Tensor):
            raise InvalidParameterError("LatentStats.kld_per_sample must be a Tensor.")
        if stats.kld_per_sample.dim() != 1:
            raise InvalidParameterError(
                "LatentStats.kld_per_sample must have shape (batch,), "
                f"got {tuple(stats.kld_per_sample.shape)}."
            )
        if stats.mu.shape[0] != stats.kld_per_sample.shape[0]:
            raise InvalidParameterError(
                "Latent stats batch mismatch: "
                f"mu batch {stats.mu.shape[0]} vs kld_per_sample batch {stats.kld_per_sample.shape[0]}."
            )
        return stats.kld_per_sample.mean()

    @staticmethod
    def _float_scalar_zero(x: Tensor) -> Tensor:
        """Create a floating scalar zero on ``x``'s device."""
        dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
        return torch.zeros((), device=x.device, dtype=dtype)

    def loss_components(self, x: Tensor) -> dict[str, Tensor]:
        """Compute APC loss components and intermediate tensors.

        Args:
            x: Observation tensor.

        Returns:
            Dictionary with scalar terms ``rec``, ``kld``, ``nll``, ``total``
            and helpful intermediates ``z``, ``x_rec``, ``mu``, ``logvar``.
        """
        tau_eff = self.config.sample_tau
        weights = self.config.loss_weights

        need_stats = (weights.kld > 0.0) or self.config.train_decode_mpe
        need_z_samples = (weights.rec > 0.0) or (weights.nll > 0.0)

        stats: LatentStats | None = None
        z_samples: Tensor | None = None
        components: dict[str, Tensor] = {}

        if need_stats or need_z_samples:
            encoded = self.encoder.encode(
                x,
                mpe=False,
                tau=tau_eff,
                return_latent_stats=need_stats,
            )
            if need_stats:
                if (
                    not isinstance(encoded, tuple)
                    or len(encoded) != 2
                    or not isinstance(encoded[0], LatentStats)
                    or not isinstance(encoded[1], Tensor)
                ):
                    raise UnsupportedOperationError(
                        "Encoder must return (LatentStats, z_samples) when return_latent_stats=True."
                    )
                stats = encoded[0]
                z_samples = encoded[1]
                components["mu"] = stats.mu
                components["logvar"] = stats.logvar
            else:
                if not isinstance(encoded, Tensor):
                    raise UnsupportedOperationError(
                        "Encoder must return latent samples as a Tensor when return_latent_stats=False."
                    )
                z_samples = encoded

        if z_samples is not None:
            components["z"] = z_samples

        rec = self._float_scalar_zero(x)
        if weights.rec > 0.0:
            if z_samples is None:
                raise RuntimeError("Reconstruction loss requested but latent samples were not computed.")
            if self.config.train_decode_mpe:
                if stats is None:
                    raise UnsupportedOperationError(
                        "train_decode_mpe=True requires encoder latent stats; "
                        "disable train_decode_mpe or use an encoder that supports latent stats."
                    )
                if not isinstance(stats.decode_latent, Tensor):
                    raise UnsupportedOperationError(
                        "train_decode_mpe=True requires LatentStats.decode_latent to be a Tensor."
                    )
                z_to_decode = stats.decode_latent
            else:
                z_to_decode = z_samples
            x_rec = self.decode(z_to_decode, mpe=False, tau=tau_eff)
            rec = self._reconstruction_loss(x, x_rec)
            components["x_rec"] = x_rec

        nll = self._float_scalar_zero(x)
        if weights.nll > 0.0:
            if self.config.nll_x_and_z:
                if z_samples is None:
                    raise RuntimeError("Joint NLL requested but latent samples were not computed.")
                lls = self.joint_log_likelihood(x, z_samples)
            else:
                lls = self.log_likelihood_x(x)
            nll = -lls.sum() / x.shape[0]

        kld = self._float_scalar_zero(x)
        if weights.kld > 0.0:
            if stats is None:
                raise UnsupportedOperationError(
                    "KL loss requires encoder latent stats, but they are unavailable. "
                    "Set loss_weights.kld=0 or use an encoder that supports return_latent_stats=True."
                )
            kld = self._kld_from_stats(stats)

        total = weights.rec * rec + weights.kld * kld + weights.nll * nll
        components["rec"] = rec
        components["kld"] = kld
        components["nll"] = nll
        components["total"] = total
        return components

    def loss(self, x: Tensor) -> Tensor:
        """Return only the weighted total APC loss."""
        return self.loss_components(x)["total"]

    def log_likelihood_x(self, x: Tensor) -> Tensor:
        """Compute encoder marginal log-likelihood ``log p(x)`` per sample."""
        return self.encoder.log_likelihood_x(x)

    def joint_log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        """Compute encoder joint log-likelihood ``log p(x, z)`` per sample."""
        return self.encoder.joint_log_likelihood(x, z)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Alias for :math:`loss_components` to integrate with training loops."""
        return self.loss_components(x)

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.config.latent_dim}, rec_loss={self.config.rec_loss}, "
            f"weights=(rec={self.config.loss_weights.rec}, kld={self.config.loss_weights.kld}, "
            f"nll={self.config.loss_weights.nll})"
        )
