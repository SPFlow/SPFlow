"""APC orchestrator model skeleton.

The full implementation is completed in later APC tasks.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from spflow.exceptions import InvalidParameterError
from spflow.zoo.apc.config import ApcConfig
from spflow.zoo.apc.encoders.base import ApcEncoder, LatentStats


class AutoencodingPC(nn.Module):
    """APC model combining a tractable encoder and a pluggable decoder."""

    def __init__(
        self,
        encoder: ApcEncoder,
        decoder: nn.Module | None,
        config: ApcConfig,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

    def encode(self, x: Tensor, *, mpe: bool = False, tau: float | None = None) -> Tensor:
        tau_eff = self.config.sample_tau if tau is None else tau
        return self.encoder.encode(x, mpe=mpe, tau=tau_eff)  # type: ignore[return-value]

    def decode(self, z: Tensor, *, mpe: bool = False, tau: float | None = None) -> Tensor:
        tau_eff = self.config.sample_tau if tau is None else tau
        if self.decoder is None:
            return self.encoder.decode(z, mpe=mpe, tau=tau_eff)
        return self.decoder(z)

    def reconstruct(self, x: Tensor, *, mpe: bool = False, tau: float | None = None) -> Tensor:
        z = self.encode(x, mpe=mpe, tau=tau)
        return self.decode(z, mpe=mpe, tau=tau)

    def sample_x(self, num_samples: int, *, tau: float | None = None) -> Tensor:
        z = self.sample_z(num_samples=num_samples, tau=tau)
        return self.decode(z, mpe=False, tau=tau)

    def sample_z(self, num_samples: int, *, tau: float | None = None) -> Tensor:
        tau_eff = self.config.sample_tau if tau is None else tau
        return self.encoder.sample_prior_z(num_samples=num_samples, tau=tau_eff)

    @staticmethod
    def _flatten_tensor(tensor: Tensor) -> Tensor:
        if tensor.dim() < 2:
            raise InvalidParameterError(
                f"Expected tensor with batch dimension and at least one feature axis, got shape {tuple(tensor.shape)}."
            )
        return tensor.reshape(tensor.shape[0], -1)

    def _reconstruction_loss(self, x: Tensor, x_rec: Tensor) -> Tensor:
        x_flat = self._flatten_tensor(x)
        x_rec_flat = self._flatten_tensor(x_rec)

        if x_flat.shape != x_rec_flat.shape:
            raise InvalidParameterError(
                f"Reconstruction shape mismatch: x has {tuple(x_flat.shape)}, x_rec has {tuple(x_rec_flat.shape)}."
            )

        observed = torch.isfinite(x_flat)
        if not observed.any():
            return x_flat.new_zeros(())

        x_obs = x_flat[observed]
        x_rec_obs = x_rec_flat[observed]

        if self.config.rec_loss == "mse":
            return F.mse_loss(x_rec_obs, x_obs, reduction="mean")
        if self.config.rec_loss == "bce":
            x_rec_obs = x_rec_obs.clamp_min(1e-6).clamp_max(1.0 - 1e-6)
            return F.binary_cross_entropy(x_rec_obs, x_obs, reduction="mean")
        raise InvalidParameterError(f"Unsupported rec_loss '{self.config.rec_loss}'.")

    @staticmethod
    def _kld_from_stats(stats: LatentStats) -> Tensor:
        if stats.mu.shape != stats.logvar.shape:
            raise InvalidParameterError(
                f"Latent stats shape mismatch: mu {tuple(stats.mu.shape)} vs logvar {tuple(stats.logvar.shape)}."
            )
        reduce_dims = tuple(range(1, stats.mu.dim()))
        if len(reduce_dims) == 0:
            raise InvalidParameterError("Latent stats must include at least one latent dimension.")
        kld_per_sample = 0.5 * (stats.mu.pow(2) + stats.logvar.exp() - 1.0 - stats.logvar).sum(
            dim=reduce_dims
        )
        return kld_per_sample.mean()

    def loss_components(self, x: Tensor) -> dict[str, Tensor]:
        tau = self.config.sample_tau
        latent_out = self.encoder.encode(x, mpe=False, tau=tau, return_latent_stats=True)
        if not isinstance(latent_out, tuple) or len(latent_out) != 2:
            raise InvalidParameterError("Encoder must return (LatentStats, z) when return_latent_stats=True.")
        stats, z = latent_out
        if not isinstance(stats, LatentStats):
            raise InvalidParameterError(
                f"Encoder returned invalid latent stats type: {type(stats)}. Expected LatentStats."
            )

        x_rec = self.decode(z, mpe=False, tau=tau)

        rec = self._reconstruction_loss(x=x, x_rec=x_rec)
        kld = self._kld_from_stats(stats)
        nll = -self.encoder.joint_log_likelihood(x, z).mean()

        w = self.config.loss_weights
        total = w.rec * rec + w.kld * kld + w.nll * nll

        return {
            "rec": rec,
            "kld": kld,
            "nll": nll,
            "total": total,
            "z": z,
            "x_rec": x_rec,
            "mu": stats.mu,
            "logvar": stats.logvar,
        }

    def loss(self, x: Tensor) -> Tensor:
        return self.loss_components(x)["total"]

    def log_likelihood_x(self, x: Tensor) -> Tensor:
        return self.encoder.log_likelihood_x(x)

    def joint_log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        return self.encoder.joint_log_likelihood(x, z)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return self.loss_components(x)

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.config.latent_dim}, rec_loss={self.config.rec_loss}, "
            f"weights=(rec={self.config.loss_weights.rec}, kld={self.config.loss_weights.kld}, "
            f"nll={self.config.loss_weights.nll})"
        )
