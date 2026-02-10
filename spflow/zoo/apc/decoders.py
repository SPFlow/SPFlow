"""Neural decoder modules used by APC models."""

from __future__ import annotations

import math
from typing import Literal

from torch import Tensor
from torch import nn
from torch.nn import functional as F

from spflow.exceptions import InvalidParameterError


class _Residual(nn.Module):
    """Single residual block used by the reference-style image decoder."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_hiddens: int,
        num_residual_hiddens: int,
        bn: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(num_residual_hiddens))
            layers.insert(5, nn.BatchNorm2d(num_hiddens))
        self._block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self._block(x)


class _ResidualStack(nn.Module):
    """Residual stack used by the reference-style image decoder."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        bn: bool,
    ) -> None:
        super().__init__()
        self._layers = nn.ModuleList(
            [
                _Residual(
                    in_channels=in_channels,
                    num_hiddens=num_hiddens,
                    num_residual_hiddens=num_residual_hiddens,
                    bn=bn,
                )
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._layers:
            x = layer(x)
        return F.relu(x)


class MLPDecoder1D(nn.Module):
    """MLP decoder mapping latent vectors to flat feature vectors.

    The module expects latent input shaped ``(B, latent_dim)`` (or reshape-compatible)
    and returns reconstructed vectors shaped ``(B, output_dim)``.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        out_activation: Literal["identity", "tanh", "sigmoid"] = "identity",
    ) -> None:
        """Initialize an MLP decoder for 1D/tabular reconstructions.

        Args:
            latent_dim: Size of the latent representation.
            output_dim: Number of output reconstruction features.
            hidden_dims: Width of hidden layers.
            out_activation: Final output activation.
        """
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
        """Decode latent vectors into reconstruction vectors.

        Args:
            z: Latent tensor of shape ``(B, latent_dim)`` (or reshape-compatible).

        Returns:
            Tensor of shape ``(B, output_dim)``.
        """
        z = z.reshape(z.shape[0], -1)
        if z.shape[1] != self.latent_dim:
            raise InvalidParameterError(f"Expected latent feature size {self.latent_dim}, got {z.shape[1]}.")
        return self.net(z)


class ConvDecoder2D(nn.Module):
    """Convolutional decoder mapping latent vectors to image-shaped outputs.

    The decoder projects ``z`` to a coarse feature map, upsamples through small
    convolutional blocks, and resizes to the exact configured output image size.
    """

    def __init__(
        self,
        latent_dim: int,
        output_shape: tuple[int, int, int],
        base_channels: int = 128,
        num_upsamples: int = 2,
        out_activation: Literal["identity", "tanh", "sigmoid"] = "identity",
    ) -> None:
        """Initialize a convolutional image decoder.

        Args:
            latent_dim: Size of the latent representation.
            output_shape: Target output shape ``(channels, height, width)``.
            base_channels: Initial projected channel count.
            num_upsamples: Number of nearest-neighbor upsampling blocks.
            out_activation: Final output activation.
        """
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
        """Decode latent vectors into image-shaped reconstructions.

        Args:
            z: Latent tensor of shape ``(B, latent_dim)`` (or reshape-compatible).

        Returns:
            Tensor of shape ``(B, C, H, W)`` matching ``output_shape``.
        """
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
            # Projection + discrete upsampling may overshoot/undershoot by one pixel.
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)

        return self.out_activation(x)


class NeuralDecoder2D(nn.Module):
    """Reference-style neural 2D decoder used by APC Conv-PC setups."""

    def __init__(
        self,
        latent_dim: int,
        output_shape: tuple[int, int, int],
        *,
        num_hidden: int = 64,
        num_res_hidden: int = 16,
        num_res_layers: int = 2,
        num_scales: int = 2,
        bn: bool = True,
        out_activation: Literal["identity", "linear", "tanh", "sigmoid"] = "tanh",
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
        if num_hidden <= 0 or num_res_hidden <= 0 or num_res_layers <= 0:
            raise InvalidParameterError(
                "num_hidden, num_res_hidden, and num_res_layers must be >= 1, "
                f"got ({num_hidden}, {num_res_hidden}, {num_res_layers})."
            )
        if num_scales < 2:
            raise InvalidParameterError(f"num_scales must be >= 2, got {num_scales}.")
        scale_divisor = 2**num_scales
        if height % scale_divisor != 0 or width % scale_divisor != 0:
            raise InvalidParameterError(
                "output spatial size must be divisible by 2**num_scales for NeuralDecoder2D. "
                f"Got (height={height}, width={width}, num_scales={num_scales})."
            )
        if num_hidden % 2 != 0:
            raise InvalidParameterError(
                f"num_hidden must be even for final channel halving, got {num_hidden}."
            )

        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.first_h = height // scale_divisor
        self.first_w = width // scale_divisor

        self.linear = nn.Linear(latent_dim, self.first_h * self.first_w * num_hidden)
        self._conv_1 = nn.Conv2d(
            in_channels=num_hidden,
            out_channels=num_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._residual_stack = _ResidualStack(
            in_channels=num_hidden,
            num_hiddens=num_hidden,
            num_residual_layers=num_res_layers,
            num_residual_hiddens=num_res_hidden,
            bn=bn,
        )
        self.scales = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=num_hidden,
                    out_channels=num_hidden,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
                for _ in range(num_scales - 2)
            ]
        )
        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hidden,
            out_channels=num_hidden // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hidden // 2,
            out_channels=channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        if out_activation in {"identity", "linear"}:
            self.out_activation: nn.Module = nn.Identity()
        elif out_activation == "tanh":
            self.out_activation = nn.Tanh()
        elif out_activation == "sigmoid":
            self.out_activation = nn.Sigmoid()
        else:
            raise InvalidParameterError(
                "out_activation must be one of {'identity', 'linear', 'tanh', 'sigmoid'}, "
                f"got '{out_activation}'."
            )

    def forward(self, z: Tensor) -> Tensor:
        z = z.reshape(z.shape[0], -1)
        if z.shape[1] != self.latent_dim:
            raise InvalidParameterError(f"Expected latent feature size {self.latent_dim}, got {z.shape[1]}.")

        x = self.linear(z)
        x = x.view(x.shape[0], -1, self.first_h, self.first_w)
        x = self._conv_1(x)
        x = self._residual_stack(x)

        for scale in self.scales:
            x = scale(x)
            x = F.relu(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)
        x = self._conv_trans_2(x)

        target_h = self.output_shape[1]
        target_w = self.output_shape[2]
        if x.shape[-2] != target_h or x.shape[-1] != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)

        return self.out_activation(x)
