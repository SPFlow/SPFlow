"""Conv-PC-based APC joint encoder over data and latent variables.

This encoder constructs a Conv-PC over a flattened joint variable block ``[X, Z]``
and injects the latent branch at a configurable depth in the Conv-PC hierarchy.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterError, ShapeError
from spflow.modules.conv.prod_conv import ProdConv
from spflow.modules.conv.sum_conv import SumConv
from spflow.modules.leaves import Normal
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.products.elementwise_product import ElementwiseProduct
from spflow.modules.sums import Sum
from spflow.modules.sums.repetition_mixing_layer import RepetitionMixingLayer
from spflow.zoo.apc.encoders.base import LatentStats
from spflow.zoo.conv.conv_pc import compute_non_overlapping_kernel_and_padding

LeafFactory = Callable[[list[int], int, int], LeafModule]


def _default_normal_leaf(scope_indices: list[int], out_channels: int, num_repetitions: int) -> LeafModule:
    """Create a normal leaf over a scope block."""
    return Normal(
        scope=scope_indices,
        out_channels=out_channels,
        num_repetitions=num_repetitions,
    )


class ConvPcJointEncoder(nn.Module):
    """Joint APC encoder using a Conv-PC backbone with latent fusion."""

    def __init__(
        self,
        *,
        input_height: int,
        input_width: int,
        input_channels: int = 1,
        latent_dim: int,
        channels: int = 16,
        depth: int = 2,
        kernel_size: int = 2,
        num_repetitions: int = 1,
        use_sum_conv: bool = False,
        latent_depth: int = 0,
        latent_channels: int | None = None,
        x_leaf_channels: int | None = None,
        x_leaf_factory: LeafFactory | None = None,
        z_leaf_factory: LeafFactory | None = None,
        posterior_stat_samples: int = 4,
        posterior_var_floor: float = 1e-6,
    ) -> None:
        """Initialize a Conv-PC APC encoder.

        Args:
            input_height: Input image height.
            input_width: Input image width.
            input_channels: Input image channels.
            latent_dim: Number of latent dimensions in ``Z``.
            channels: Internal Conv-PC channel count.
            depth: Conv-PC depth.
            kernel_size: Product/sum kernel size.
            num_repetitions: Number of repetitions/channels in the circuit.
            use_sum_conv: Whether to use ``SumConv`` layers instead of dense ``Sum``.
            latent_depth: Sum-stage depth where the latent branch is fused.
            latent_channels: Latent leaf channels (defaults to ``channels``).
            x_leaf_channels: Data leaf channels (defaults to ``channels``).
            x_leaf_factory: Factory to create the ``X`` leaf module.
            z_leaf_factory: Factory to create the ``Z`` leaf module.
            posterior_stat_samples: Number of posterior samples used to estimate moments.
            posterior_var_floor: Numerical floor for latent variance estimates.
        """
        super().__init__()

        if input_height <= 0 or input_width <= 0:
            raise InvalidParameterError(
                f"input_height and input_width must be >= 1, got ({input_height}, {input_width})."
            )
        if input_channels <= 0:
            raise InvalidParameterError(f"input_channels must be >= 1, got {input_channels}.")
        if latent_dim <= 0:
            raise InvalidParameterError(f"latent_dim must be >= 1, got {latent_dim}.")
        if channels <= 0:
            raise InvalidParameterError(f"channels must be >= 1, got {channels}.")
        if depth <= 0:
            raise InvalidParameterError(f"depth must be >= 1, got {depth}.")
        if kernel_size <= 0:
            raise InvalidParameterError(f"kernel_size must be >= 1, got {kernel_size}.")
        if num_repetitions <= 0:
            raise InvalidParameterError(f"num_repetitions must be >= 1, got {num_repetitions}.")
        if latent_depth < 0 or latent_depth >= depth:
            raise InvalidParameterError(f"latent_depth must be in [0, {depth - 1}], got {latent_depth}.")
        if posterior_stat_samples <= 0:
            raise InvalidParameterError(f"posterior_stat_samples must be >= 1, got {posterior_stat_samples}.")
        if posterior_var_floor <= 0.0:
            raise InvalidParameterError(f"posterior_var_floor must be > 0, got {posterior_var_floor}.")

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.num_x_features = input_height * input_width * input_channels
        self.latent_dim = latent_dim
        self.posterior_stat_samples = posterior_stat_samples
        self.posterior_var_floor = posterior_var_floor

        self._x_cols = list(range(self.num_x_features))
        self._z_cols = list(range(self.num_x_features, self.num_x_features + latent_dim))

        x_leaf_channels = channels if x_leaf_channels is None else x_leaf_channels
        latent_channels = channels if latent_channels is None else latent_channels
        if x_leaf_channels <= 0 or latent_channels <= 0:
            raise InvalidParameterError(
                f"x_leaf_channels and latent_channels must be >= 1, got ({x_leaf_channels}, {latent_channels})."
            )

        x_leaf_factory = x_leaf_factory or _default_normal_leaf
        z_leaf_factory = z_leaf_factory or _default_normal_leaf

        x_leaf = x_leaf_factory(self._x_cols, x_leaf_channels, num_repetitions)
        self._validate_leaf_scope(leaf=x_leaf, expected_scope=self._x_cols, role="x")

        z_leaf = z_leaf_factory(self._z_cols, latent_channels, num_repetitions)
        self._validate_leaf_scope(leaf=z_leaf, expected_scope=self._z_cols, role="z")

        self.pc = self._build_joint_convpc(
            x_leaf=x_leaf,
            z_leaf=z_leaf,
            channels=channels,
            depth=depth,
            kernel_size=kernel_size,
            num_repetitions=num_repetitions,
            use_sum_conv=use_sum_conv,
            latent_depth=latent_depth,
        )

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

    def _build_joint_convpc(
        self,
        *,
        x_leaf: LeafModule,
        z_leaf: LeafModule,
        channels: int,
        depth: int,
        kernel_size: int,
        num_repetitions: int,
        use_sum_conv: bool,
        latent_depth: int,
    ) -> nn.Module:
        """Build a joint Conv-PC chain and inject latent branch at configured depth.

        The latent branch is fused through :class:`ElementwiseProduct`. This requires
        matching feature cardinality at the injection node, so ``latent_dim`` must
        equal the target node feature count.
        """
        layer_specs: list[tuple[str, dict[str, int]]] = []
        layer_specs.append(("sum_root", {"out_channels": 1}))

        h, w = 1, 1
        for _ in reversed(range(depth)):
            layer_specs.append(("prod", {"kernel_size": kernel_size}))
            h, w = h * kernel_size, w * kernel_size
            layer_specs.append(
                (
                    "sum",
                    {
                        "out_channels": channels,
                        "kernel_size": kernel_size,
                    },
                )
            )

        (kh, kw), (ph, pw) = compute_non_overlapping_kernel_and_padding(
            H_data=self.input_height,
            W_data=self.input_width,
            H_target=h,
            W_target=w,
        )
        layer_specs.append(
            (
                "prod_bottom",
                {"kernel_size_h": kh, "kernel_size_w": kw, "padding_h": ph, "padding_w": pw},
            )
        )

        layer_specs = list(reversed(layer_specs))

        current: nn.Module = x_leaf
        latent_injected = False
        sum_stage = -1

        for layer_type, params in layer_specs:
            if layer_type == "prod_bottom":
                current = ProdConv(
                    inputs=current,
                    kernel_size_h=params["kernel_size_h"],
                    kernel_size_w=params["kernel_size_w"],
                    padding_h=params["padding_h"],
                    padding_w=params["padding_w"],
                )
            elif layer_type == "sum":
                if use_sum_conv:
                    current = SumConv(
                        inputs=current,
                        out_channels=params["out_channels"],
                        kernel_size=params["kernel_size"],
                        num_repetitions=num_repetitions,
                    )
                else:
                    current = Sum(
                        inputs=current,
                        out_channels=params["out_channels"],
                        num_repetitions=num_repetitions,
                    )

                sum_stage += 1
                if sum_stage == latent_depth:
                    # Latent evidence is represented as one latent variable per target feature.
                    target_features = current.out_shape.features
                    if target_features != self.latent_dim:
                        raise InvalidParameterError(
                            "latent_dim must match the feature count at latent injection depth. "
                            f"Expected {target_features}, got {self.latent_dim}."
                        )

                    latent_stream: nn.Module = z_leaf
                    if z_leaf.out_shape.channels != current.out_shape.channels:
                        # Align channels before multiplicative fusion.
                        latent_stream = Sum(
                            inputs=latent_stream,
                            out_channels=current.out_shape.channels,
                            num_repetitions=num_repetitions,
                        )

                    current = ElementwiseProduct(inputs=[current, latent_stream])
                    latent_injected = True

            elif layer_type == "prod":
                current = ProdConv(
                    inputs=current,
                    kernel_size_h=params["kernel_size"],
                    kernel_size_w=params["kernel_size"],
                )
            elif layer_type == "sum_root":
                current = Sum(
                    inputs=current,
                    out_channels=params["out_channels"],
                    num_repetitions=num_repetitions,
                )
            else:
                raise RuntimeError(f"Unexpected layer type '{layer_type}'.")

        if not latent_injected:
            raise RuntimeError("Latent branch was not injected. Check latent_depth configuration.")

        if num_repetitions > 1:
            current = RepetitionMixingLayer(
                inputs=current,
                out_channels=1,
                num_repetitions=num_repetitions,
            )

        return current

    def _flatten_x(self, x: Tensor) -> Tensor:
        """Flatten 2D/4D input ``x`` to ``(B, num_x_features)`` and validate shape."""
        if x.dim() == 2:
            x_flat = x
        elif x.dim() == 4:
            if (
                x.shape[1] != self.input_channels
                or x.shape[2] != self.input_height
                or x.shape[3] != self.input_width
            ):
                raise ShapeError(
                    "x image shape mismatch. "
                    f"Expected (B, {self.input_channels}, {self.input_height}, {self.input_width}), "
                    f"got {tuple(x.shape)}."
                )
            x_flat = x.reshape(x.shape[0], -1)
        else:
            raise ShapeError(f"x must be rank-2 or rank-4, got shape {tuple(x.shape)}.")

        if x_flat.shape[1] != self.num_x_features:
            raise ShapeError(
                f"Expected x to have {self.num_x_features} flattened features, got {x_flat.shape[1]}."
            )
        return x_flat

    def _reshape_x_like(self, x_flat: Tensor, x_like: Tensor | None) -> Tensor:
        """Reshape flattened ``x`` back to image shape when needed."""
        if x_like is not None and x_like.dim() == 2:
            return x_flat
        return x_flat.view(-1, self.input_channels, self.input_height, self.input_width)

    def _flatten_z(self, z: Tensor) -> Tensor:
        """Flatten ``z`` to ``(B, latent_dim)`` and validate dimensionality."""
        if z.dim() < 2:
            raise ShapeError(f"z must have at least 2 dimensions, got shape {tuple(z.shape)}.")
        z_flat = z.reshape(z.shape[0], -1)
        if z_flat.shape[1] != self.latent_dim:
            raise ShapeError(f"Expected z to have latent_dim={self.latent_dim}, got {z_flat.shape[1]}.")
        return z_flat

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
                device = self.pc.device

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

    def _posterior_sample(self, x_flat: Tensor, *, mpe: bool, tau: float) -> Tensor:
        """Sample ``z ~ p(Z|X=x)`` with runtime fallback to non-differentiable sampling."""
        evidence = self._build_evidence(x_flat=x_flat, z_flat=None)
        if mpe:
            joint = self.pc.sample(data=evidence, is_mpe=True)
        else:
            try:
                joint = self.pc.rsample(data=evidence, is_mpe=False, tau=tau, hard=True)
            except (RuntimeError, ValueError):
                # Some fused-product sampling paths do not currently support soft routing.
                joint = self.pc.sample(data=evidence, is_mpe=False)
        return joint[:, self._z_cols]

    def encode(
        self,
        x: Tensor,
        *,
        mpe: bool = False,
        tau: float = 1.0,
        return_latent_stats: bool = False,
    ) -> Tensor | tuple[LatentStats, Tensor]:
        """Encode observations into latent samples.

        Args:
            x: Observation tensor (flattened or image-shaped).
            mpe: Whether to use deterministic MPE routing.
            tau: Temperature for differentiable sampling.
            return_latent_stats: When ``True``, return ``(LatentStats, z)``.

        Returns:
            Either ``z`` or ``(LatentStats, z)``.
        """
        x_flat = self._flatten_x(x)
        z = self._posterior_sample(x_flat, mpe=mpe, tau=tau)
        if return_latent_stats:
            return self.latent_stats(x, tau=tau), z
        return z

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

        if mpe:
            joint = self.pc.sample(data=evidence, is_mpe=True)
        else:
            try:
                joint = self.pc.rsample(data=evidence, is_mpe=False, tau=tau, hard=True)
            except (RuntimeError, ValueError):
                joint = self.pc.sample(data=evidence, is_mpe=False)

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
        ll = self.pc.log_likelihood(evidence)
        return self._flatten_ll(ll)

    def log_likelihood_x(self, x: Tensor) -> Tensor:
        """Compute per-sample marginal log-likelihood ``log p(x)``."""
        x_flat = self._flatten_x(x)
        evidence = self._build_evidence(x_flat=x_flat, z_flat=None)
        ll = self.pc.log_likelihood(evidence)
        return self._flatten_ll(ll)

    def sample_prior_z(self, num_samples: int, *, tau: float = 1.0) -> Tensor:
        """Sample latent variables from the model prior over ``Z``."""
        if num_samples <= 0:
            raise InvalidParameterError(f"num_samples must be >= 1, got {num_samples}.")
        evidence = self._build_evidence(
            x_flat=None, z_flat=None, num_samples=num_samples, device=self.pc.device
        )
        try:
            joint = self.pc.rsample(
                data=evidence,
                is_mpe=False,
                tau=tau,
                hard=True,
            )
        except (RuntimeError, ValueError):
            joint = self.pc.sample(data=evidence, is_mpe=False)
        return joint[:, self._z_cols]

    def latent_stats(self, x: Tensor, *, tau: float = 1.0) -> LatentStats:
        """Estimate posterior moments for ``p(Z|X=x)`` via Monte Carlo samples."""
        x_flat = self._flatten_x(x)
        samples = [
            self._posterior_sample(x_flat, mpe=False, tau=tau) for _ in range(self.posterior_stat_samples)
        ]
        stacked = torch.stack(samples, dim=0)  # (K, B, latent_dim)
        mu = stacked.mean(dim=0)
        var = stacked.var(dim=0, unbiased=False).clamp_min(self.posterior_var_floor)
        logvar = var.log()
        return LatentStats(mu=mu, logvar=logvar)
