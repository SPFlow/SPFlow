"""Decoder modules for APC models."""

from __future__ import annotations

import math
from typing import Literal

from torch import Tensor
from torch import nn
from torch.nn import functional as F

from spflow.exceptions import InvalidParameterError


class MLPDecoder1D(nn.Module):
    """MLP decoder mapping latent vectors to 1D reconstructions."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        out_activation: Literal["identity", "tanh", "sigmoid"] = "identity",
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise InvalidParameterError(f"latent_dim must be >= 1, got {latent_dim}.")
        if output_dim <= 0:
            raise InvalidParameterError(f"output_dim must be >= 1, got {output_dim}.")
        if len(hidden_dims) == 0:
            raise InvalidParameterError("hidden_dims must contain at least one layer size.")
        if any(h <= 0 for h in hidden_dims):
            raise InvalidParameterError(f"hidden_dims must be positive, got {hidden_dims}.")

        layers: list[nn.Module] = []
        in_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.1))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        if out_activation == "tanh":
            layers.append(nn.Tanh())
        elif out_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif out_activation != "identity":
            raise InvalidParameterError(
                "out_activation must be one of {'identity', 'tanh', 'sigmoid'}, " f"got '{out_activation}'."
            )

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        z = z.reshape(z.shape[0], -1)
        if z.shape[1] != self.latent_dim:
            raise InvalidParameterError(f"Expected latent feature size {self.latent_dim}, got {z.shape[1]}.")
        return self.net(z)


class ConvDecoder2D(nn.Module):
    """Convolutional decoder mapping latent vectors to image-shaped outputs."""

    def __init__(
        self,
        latent_dim: int,
        output_shape: tuple[int, int, int],
        base_channels: int = 128,
        num_upsamples: int = 2,
        out_activation: Literal["identity", "tanh", "sigmoid"] = "identity",
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise InvalidParameterError(f"latent_dim must be >= 1, got {latent_dim}.")
        if len(output_shape) != 3:
            raise InvalidParameterError(
                f"output_shape must be (channels, height, width), got {output_shape}."
            )
        channels, height, width = output_shape
        if channels <= 0 or height <= 0 or width <= 0:
            raise InvalidParameterError(
                "output_shape entries must be positive, "
                f"got (channels={channels}, height={height}, width={width})."
            )
        if base_channels <= 0:
            raise InvalidParameterError(f"base_channels must be >= 1, got {base_channels}.")
        if num_upsamples < 0:
            raise InvalidParameterError(f"num_upsamples must be >= 0, got {num_upsamples}.")

        scale = 2**num_upsamples
        start_h = max(1, math.ceil(height / scale))
        start_w = max(1, math.ceil(width / scale))

        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.start_h = start_h
        self.start_w = start_w
        self.num_upsamples = num_upsamples
        self.proj = nn.Linear(latent_dim, base_channels * start_h * start_w)

        blocks: list[nn.Module] = []
        in_channels = base_channels
        for _ in range(num_upsamples):
            out_channels = max(channels, in_channels // 2)
            blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(0.1),
                )
            )
            in_channels = out_channels
        self.upsample = nn.Sequential(*blocks)
        self.out_conv = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)

        if out_activation == "identity":
            self.out_activation: nn.Module = nn.Identity()
        elif out_activation == "tanh":
            self.out_activation = nn.Tanh()
        elif out_activation == "sigmoid":
            self.out_activation = nn.Sigmoid()
        else:
            raise InvalidParameterError(
                "out_activation must be one of {'identity', 'tanh', 'sigmoid'}, " f"got '{out_activation}'."
            )

    def forward(self, z: Tensor) -> Tensor:
        z = z.reshape(z.shape[0], -1)
        if z.shape[1] != self.latent_dim:
            raise InvalidParameterError(f"Expected latent feature size {self.latent_dim}, got {z.shape[1]}.")

        x = self.proj(z)
        x = x.reshape(z.shape[0], -1, self.start_h, self.start_w)
        x = self.upsample(x)
        x = self.out_conv(x)

        target_h = self.output_shape[1]
        target_w = self.output_shape[2]
        if x.shape[-2] != target_h or x.shape[-1] != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)

        return self.out_activation(x)
