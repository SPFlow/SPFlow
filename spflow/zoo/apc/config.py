"""Configuration objects for Autoencoding Probabilistic Circuits (APC)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from spflow.exceptions import InvalidParameterError


@dataclass(frozen=True)
class ApcLossWeights:
    """Loss term weights for APC training."""

    rec: float = 1.0
    kld: float = 1.0
    nll: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (("rec", self.rec), ("kld", self.kld), ("nll", self.nll)):
            if value < 0.0:
                raise InvalidParameterError(f"ApcLossWeights.{name} must be >= 0, got {value}.")


@dataclass(frozen=True)
class ApcConfig:
    """Core APC model configuration."""

    latent_dim: int
    rec_loss: Literal["mse", "bce"] = "mse"
    sample_tau: float = 1.0
    loss_weights: ApcLossWeights = field(default_factory=ApcLossWeights)

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise InvalidParameterError(f"latent_dim must be >= 1, got {self.latent_dim}.")
        if self.sample_tau <= 0.0:
            raise InvalidParameterError(f"sample_tau must be > 0, got {self.sample_tau}.")


@dataclass(frozen=True)
class ApcTrainConfig:
    """Lightweight APC trainer configuration."""

    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise InvalidParameterError(f"epochs must be >= 1, got {self.epochs}.")
        if self.batch_size <= 0:
            raise InvalidParameterError(f"batch_size must be >= 1, got {self.batch_size}.")
        if self.learning_rate <= 0.0:
            raise InvalidParameterError(f"learning_rate must be > 0, got {self.learning_rate}.")
        if self.weight_decay < 0.0:
            raise InvalidParameterError(f"weight_decay must be >= 0, got {self.weight_decay}.")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise InvalidParameterError(f"grad_clip_norm must be > 0 when set, got {self.grad_clip_norm}.")
