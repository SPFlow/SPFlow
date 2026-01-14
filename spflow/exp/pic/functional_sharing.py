"""Functional sharing utilities for Probabilistic Integral Circuits.

This module provides neural network components for functional sharing in PICs,
as described in Section 3.3 of the NeurIPS 2024 paper:
"Scaling Continuous Latent Variable Models as Probabilistic Integral Circuits"

Functional sharing reduces the number of parameters and speeds up QPC materialization
by sharing MLPs across multiple PIC units.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FourierFeatures(nn.Module):
    """Fourier feature encoding layer for positional encoding.

    Maps low-dimensional inputs to higher-dimensional features using
    random Fourier features, which helps MLPs learn high-frequency functions.

    From paper Eq. 5: FF : R^I → R^M

    Attributes:
        B: Random frequency matrix (not trained).
        scale: Frequency scaling factor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 1.0,
    ) -> None:
        """Initialize FourierFeatures layer.

        Args:
            in_features: Input dimension I.
            out_features: Output dimension M (half of final output due to sin/cos).
            scale: Scaling factor for frequencies.
        """
        super().__init__()
        # Random frequency matrix (not trainable)
        self.register_buffer("B", torch.randn(in_features, out_features) * scale)
        self.out_dim = out_features * 2  # sin and cos

    def forward(self, x: Tensor) -> Tensor:
        """Apply Fourier feature encoding.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Tensor of shape (..., out_features * 2).
        """
        # x @ B: (..., out_features)
        projected = x @ self.B
        # Concatenate sin and cos for each frequency
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class SharedMLP(nn.Module):
    """Shared MLP backbone for functional sharing.

    Parameterizes the shared function f in functional sharing.
    Uses Fourier features followed by MLP layers with nonlinearity.

    From paper: φ^(γ) : R^I → R^M := φ_L ∘ ... ∘ φ_1 ∘ FF

    Attributes:
        fourier: FourierFeatures input encoding.
        layers: Sequential MLP layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        activation: nn.Module = nn.SiLU(),
        fourier_scale: float = 1.0,
    ) -> None:
        """Initialize SharedMLP.

        Args:
            input_dim: Dimension of input (e.g., latent variable dimension).
            hidden_dim: Dimension of hidden layers M.
            num_layers: Number of hidden layers L.
            activation: Activation function ψ.
            fourier_scale: Scale for Fourier features.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Fourier feature encoding
        self.fourier = FourierFeatures(input_dim, hidden_dim, scale=fourier_scale)
        fourier_out = self.fourier.out_dim

        # MLP layers
        layers: List[nn.Module] = []

        # First layer: Fourier output → hidden
        layers.append(nn.Linear(fourier_out, hidden_dim))
        layers.append(activation)

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through shared MLP.

        Args:
            x: Input tensor of shape (..., input_dim).

        Returns:
            Hidden representation of shape (..., hidden_dim).
        """
        # Apply Fourier features
        h = self.fourier(x)
        # Apply MLP layers
        return self.layers(h)


class MultiHeadedMLP(nn.Module):
    """Multi-headed MLP for C-sharing (composite sharing).

    Shares a SharedMLP backbone across multiple functions, with separate
    output heads for each function. This enables efficient C-sharing where
    fi = hi ∘ f, sharing inner function f.

    From paper (neural C-sharing):
    fi : R^M → R := softplus(h^(i) · φ^(γ) + b^(i))

    Attributes:
        shared: SharedMLP backbone.
        heads: List of linear heads for each function.
    """

    def __init__(
        self,
        shared_mlp: SharedMLP,
        num_heads: int,
        output_activation: Optional[nn.Module] = None,
    ) -> None:
        """Initialize MultiHeadedMLP.

        Args:
            shared_mlp: Shared MLP backbone.
            num_heads: Number of output heads N.
            output_activation: Activation for outputs (default: softplus for positivity).
        """
        super().__init__()

        self.shared = shared_mlp
        self.num_heads = num_heads

        # Create heads: each is (h^(i), b^(i)) pair
        hidden_dim = shared_mlp.hidden_dim
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_heads)])

        self.output_activation = output_activation or nn.Softplus()

    def forward(self, x: Tensor, head_idx: Optional[int] = None) -> Tensor:
        """Forward pass through multi-headed MLP.

        Args:
            x: Input tensor of shape (..., input_dim).
            head_idx: Optional specific head index. If None, returns all heads.

        Returns:
            If head_idx is specified: Output of shape (..., 1).
            Otherwise: Output of shape (..., num_heads).
        """
        # Shared backbone
        h = self.shared(x)  # (..., hidden_dim)

        if head_idx is not None:
            # Single head output
            out = self.heads[head_idx](h)
            return self.output_activation(out)
        else:
            # All heads
            outputs = torch.cat([head(h) for head in self.heads], dim=-1)  # (..., num_heads)
            return self.output_activation(outputs)


class FunctionGroup(nn.Module):
    """Container for grouping PIC units with functional sharing.

    Groups integral/input units that share the same MLP for efficient
    materialization.

    Attributes:
        sharing_type: Type of sharing ("f" for F-sharing, "c" for C-sharing).
        units: List of units in this group.
        mlp: Shared MLP for this group.
    """

    def __init__(
        self,
        sharing_type: str = "c",
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        """Initialize FunctionGroup.

        Args:
            sharing_type: "f" for F-sharing (all same), "c" for C-sharing (multi-headed).
            input_dim: Input dimension for MLP.
            hidden_dim: Hidden dimension for MLP.
            num_layers: Number of layers in MLP.
        """
        super().__init__()
        if sharing_type not in {"c", "f"}:
            raise ValueError("sharing_type must be 'c' (C-sharing) or 'f' (F-sharing).")

        self.sharing_type = sharing_type
        self.units: list = []

        # Create shared MLP
        self.mlp = SharedMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Will be populated as units are added
        self._multi_headed: Optional[MultiHeadedMLP] = None
        self._f_head: Optional[nn.Linear] = None

    def add_unit(self, unit) -> int:
        """Add a unit to this group.

        Args:
            unit: PIC unit (Integral or input unit).

        Returns:
            Index of the unit in this group (for C-sharing head selection).
        """
        idx = len(self.units)
        self.units.append(unit)
        return idx

    def finalize(self) -> None:
        """Finalize the group after all units are added.

        Creates the multi-headed MLP for C-sharing.
        """
        if len(self.units) == 0:
            return

        if self.sharing_type == "c":
            self._multi_headed = MultiHeadedMLP(
                shared_mlp=self.mlp,
                num_heads=len(self.units),
            )
        else:
            self._f_head = nn.Linear(self.mlp.hidden_dim, 1)

    def evaluate_batched(self, z: Tensor, y: Tensor) -> Tensor:
        """Evaluate all functions in the group in a single shared-backbone pass.

        This implements the C-sharing/F-sharing semantics from Sec. 3.3 of the paper:
        - C-sharing: different heads over a shared backbone
        - F-sharing: a single head shared across units

        Args:
            z: Tensor with last dimension matching the z-input dimensionality.
            y: Tensor with last dimension matching the y-input dimensionality.
               `z` and `y` must be broadcastable to the same leading shape.

        Returns:
            If C-sharing: Tensor of shape (num_units, *leading_shape).
            If F-sharing: Tensor of shape (1, *leading_shape).
        """
        if self.sharing_type == "c" and self._multi_headed is None:
            self.finalize()
        if self.sharing_type == "f" and self._f_head is None:
            self.finalize()

        # Broadcast z and y to a common leading shape (excluding last dim).
        leading_shape = torch.broadcast_shapes(z.shape[:-1], y.shape[:-1])
        z_b = z.expand(*leading_shape, z.shape[-1])
        y_b = y.expand(*leading_shape, y.shape[-1])
        xy = torch.cat([z_b, y_b], dim=-1)

        flat = xy.reshape(-1, xy.shape[-1])
        shared_h = self.mlp(flat)  # (N, hidden_dim)

        if self.sharing_type == "c":
            assert self._multi_headed is not None
            # Reuse shared backbone representation (avoid recomputing MLP per head).
            # MultiHeadedMLP expects the original x; we bypass it and apply heads directly.
            outputs = torch.cat([head(shared_h) for head in self._multi_headed.heads], dim=-1)
            outputs = self._multi_headed.output_activation(outputs)  # (N, num_units)
            outputs = outputs.transpose(0, 1)  # (num_units, N)
        else:
            assert self._f_head is not None
            out = F.softplus(self._f_head(shared_h)).squeeze(-1)  # (N,)
            outputs = out.unsqueeze(0)  # (1, N)

        return outputs.reshape(outputs.shape[0], *leading_shape)

    def get_function(self, unit_idx: int = 0) -> Callable[[Tensor, Tensor], Tensor]:
        """Get a callable function for a specific unit/head.

        The returned callable preserves broadcast shapes: for broadcastable `z` and `y`,
        it returns a tensor with the broadcasted leading shape.

        Args:
            unit_idx: Index of the unit in this group (only used for C-sharing).

        Returns:
            Callable mapping `(z, y)` to a positive tensor.
        """

        def _fn(z: Tensor, y: Tensor) -> Tensor:
            outputs = self.evaluate_batched(z, y)
            head = 0 if self.sharing_type == "f" else unit_idx
            return outputs[head]

        return _fn
