"""Einet-based APC joint encoder over data and latent variables.

The encoder builds a joint PC over concatenated variables ``[X, Z]`` using two
leaf modules:
- one leaf that covers all observed ``X`` columns,
- one leaf that covers all latent ``Z`` columns.
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
from einops import rearrange
from torch import Tensor

from spflow.exceptions import InvalidParameterError, ShapeError
from spflow.modules.leaves import Normal
from spflow.modules.leaves.leaf import LeafModule
from spflow.zoo.apc.encoders.joint_pc_base import JointPcEncoderBase
from spflow.zoo.einet import Einet

LeafFactory = Callable[[list[int], int, int], LeafModule]


def _default_normal_leaf(scope_indices: list[int], out_channels: int, num_repetitions: int) -> LeafModule:
    """Create a Normal leaf with reference-style ``(mu, logvar)`` init."""
    event_shape = (len(scope_indices), out_channels, num_repetitions)
    loc = torch.randn(event_shape)
    logvar = torch.randn(event_shape)
    scale = torch.exp(0.5 * logvar)
    return Normal(
        scope=scope_indices,
        out_channels=out_channels,
        num_repetitions=num_repetitions,
        loc=loc,
        scale=scale,
    )


class EinetJointEncoder(JointPcEncoderBase):
    """Joint APC encoder using an Einet over concatenated features ``[x, z]``."""

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
        x_leaf_factory: LeafFactory | None = None,
        z_leaf_factory: LeafFactory | None = None,
    ) -> None:
        """Initialize a joint Einet-based APC encoder.

        Args:
            num_x_features: Number of flattened data features in ``X``.
            latent_dim: Number of latent dimensions in ``Z``.
            num_sums: Number of sum units per sum layer.
            num_leaves: Number of leaf channels/components.
            depth: Number of internal Einet product/sum stages.
            num_repetitions: Number of repetitions/channels in the circuit.
            layer_type: Internal sum-product implementation type.
            x_leaf_factory: Factory to create the ``X`` leaf module.
            z_leaf_factory: Factory to create the ``Z`` leaf module.
        """
        super().__init__()

        if num_x_features <= 0:
            raise InvalidParameterError(f"num_x_features must be >= 1, got {num_x_features}.")
        if latent_dim <= 0:
            raise InvalidParameterError(f"latent_dim must be >= 1, got {latent_dim}.")

        self.num_x_features = num_x_features
        self.latent_dim = latent_dim

        self._x_cols = list(range(num_x_features))
        self._z_cols = list(range(num_x_features, num_x_features + latent_dim))

        x_leaf_factory = x_leaf_factory or _default_normal_leaf
        z_leaf_factory = z_leaf_factory or _default_normal_leaf

        x_leaf = x_leaf_factory(self._x_cols, num_leaves, num_repetitions)
        self._validate_leaf_scope(leaf=x_leaf, expected_scope=self._x_cols, role="x")

        z_leaf = z_leaf_factory(self._z_cols, num_leaves, num_repetitions)
        self._validate_leaf_scope(leaf=z_leaf, expected_scope=self._z_cols, role="z")
        self._z_leaf = z_leaf

        self.pc = Einet(
            leaf_modules=[x_leaf, z_leaf],
            num_classes=1,
            num_sums=num_sums,
            num_leaves=num_leaves,
            depth=depth,
            num_repetitions=num_repetitions,
            layer_type=layer_type,
            structure="top-down",
        )

    def _flatten_x(self, x: Tensor) -> Tensor:
        """Flatten ``x`` to ``(B, num_x_features)`` and validate dimensionality."""
        if x.dim() < 2:
            raise ShapeError(f"x must have at least 2 dimensions, got shape {tuple(x.shape)}.")
        x_flat = rearrange(x, "b ... -> b (...)")
        if x_flat.shape[1] != self.num_x_features:
            raise ShapeError(
                f"Expected x to have {self.num_x_features} flattened features, got {x_flat.shape[1]}."
            )
        return x_flat

    def _flatten_z(self, z: Tensor) -> Tensor:
        """Flatten ``z`` to ``(B, latent_dim)`` and validate dimensionality."""
        if z.dim() < 2:
            raise ShapeError(f"z must have at least 2 dimensions, got shape {tuple(z.shape)}.")
        z_flat = rearrange(z, "b ... -> b (...)")
        if z_flat.shape[1] != self.latent_dim:
            raise ShapeError(f"Expected z to have latent_dim={self.latent_dim}, got {z_flat.shape[1]}.")
        return z_flat
