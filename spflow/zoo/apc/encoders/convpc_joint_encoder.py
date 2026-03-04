"""Conv-PC-based APC joint encoder over data and latent variables.

This encoder builds a joint Conv-PC over ``[X, Z]`` and supports two modes:
- ``reference``: mirrors the reference APC Conv-PC topology and latent fusion.
- ``legacy``: preserves the previous SPFlow latent-injection behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from einops import reduce, repeat
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterError, ShapeError
from spflow.meta.data import Scope
from spflow.modules.conv.prod_conv import ProdConv
from spflow.modules.conv.sum_conv import SumConv
from spflow.modules.leaves import Normal
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.products.elementwise_product import ElementwiseProduct
from spflow.modules.sums import Sum
from spflow.modules.sums.repetition_mixing_layer import RepetitionMixingLayer
from spflow.utils.cache import Cache, cached
from spflow.utils.sampling_context import SamplingContext
from spflow.zoo.apc.encoders.joint_pc_base import JointPcEncoderBase
from spflow.zoo.conv.conv_pc import compute_non_overlapping_kernel_and_padding

LeafFactory = Callable[[list[int], int, int], LeafModule]


def _default_normal_leaf(scope_indices: list[int], out_channels: int, num_repetitions: int) -> LeafModule:
    """Create a Normal leaf with reference-style ``(mu, logvar)`` init.

    The reference APC initializes Normal leaves via ``mu ~ N(0,1)`` and
    ``logvar ~ N(0,1)``. SPFlow's generic Normal leaf uses ``scale`` as its
    constructor argument, so we convert by ``scale = exp(0.5 * logvar)``.
    """
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


class _PairwiseLatentProduct(Module):
    """Pairwise latent feature reduction in log-space.

    This matches reference behavior where each reduction layer maps
    ``(f0, f1, f2, f3, ...) -> (f0+f1, f2+f3, ...)``.
    """

    def __init__(self, inputs: Module) -> None:
        super().__init__()
        if inputs.out_shape.features % 2 != 0:
            raise InvalidParameterError(
                "Latent product reduction requires an even feature count, "
                f"got {inputs.out_shape.features}."
            )

        self.inputs = inputs
        self.scope = inputs.scope
        self.in_shape = inputs.out_shape
        self.out_shape = ModuleShape(
            features=inputs.out_shape.features // 2,
            channels=inputs.out_shape.channels,
            repetitions=inputs.out_shape.repetitions,
        )

        child_f2s = inputs.feature_to_scope
        out_f2s = np.empty((self.out_shape.features, self.out_shape.repetitions), dtype=object)
        for r in range(self.out_shape.repetitions):
            for f in range(self.out_shape.features):
                left = child_f2s[2 * f, r]
                right = child_f2s[2 * f + 1, r]
                out_f2s[f, r] = Scope.join_all([left, right])
        self._feature_to_scope = out_f2s

    @property
    def feature_to_scope(self) -> np.ndarray:
        return self._feature_to_scope

    @cached
    def log_likelihood(self, data: Tensor, cache: Cache | None = None) -> Tensor:
        ll = self.inputs.log_likelihood(data, cache=cache)
        b, f, c, r = ll.shape
        if f != self.in_shape.features:
            raise ShapeError(
                f"Unexpected feature size in latent product layer: expected {self.in_shape.features}, got {f}."
            )
        return reduce(ll, "b (f pair) c r -> b f c r", "sum", f=self.out_shape.features, pair=2)

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
    ) -> Tensor:
        sampling_ctx.validate_sampling_context(
            num_samples=data.shape[0],
            num_features=self.out_shape.features,
            num_channels=self.out_shape.channels,
            num_repetitions=self.out_shape.repetitions,
            allowed_feature_widths=(1, self.out_shape.features, self.in_shape.features),
        )

        ctx_features = sampling_ctx.channel_index.shape[1]
        if ctx_features == self.in_shape.features:
            return self.inputs._sample(data=data, cache=cache, sampling_ctx=sampling_ctx)
        if ctx_features == 1:
            num_parent_features = self.out_shape.features
            if sampling_ctx.is_differentiable:
                parent_idx = repeat(sampling_ctx.channel_index, "b 1 c -> b f c", f=num_parent_features)
            else:
                parent_idx = repeat(sampling_ctx.channel_index, "b 1 -> b f", f=num_parent_features)
            parent_mask = repeat(sampling_ctx.mask, "b 1 -> b f", f=num_parent_features)
        elif ctx_features == self.out_shape.features:
            parent_idx = sampling_ctx.channel_index
            parent_mask = sampling_ctx.mask
        else:
            raise ShapeError(
                "SamplingContext feature dimension mismatch in latent product layer: "
                f"got {ctx_features}, expected 1, {self.out_shape.features}, or {self.in_shape.features}."
            )

        if sampling_ctx.is_differentiable:
            child_idx = repeat(parent_idx, "b f c -> b (f two) c", two=2)
        else:
            child_idx = repeat(parent_idx, "b f -> b (f two)", two=2)
        child_mask = repeat(parent_mask, "b f -> b (f two)", two=2)
        sampling_ctx.channel_index = child_idx
        sampling_ctx.mask = child_mask
        return self.inputs._sample(data=data, cache=cache, sampling_ctx=sampling_ctx)

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Module | None:
        del marg_rvs, prune, cache
        raise NotImplementedError("Marginalization is not implemented for _PairwiseLatentProduct.")


class _LatentFeaturePacking(Module):
    """Pack latent features to injection width with optional permutation."""

    def __init__(
        self,
        inputs: Module,
        target_features: int,
        perm: Tensor | None,
        perm_inv: Tensor | None,
    ) -> None:
        super().__init__()
        if target_features < inputs.out_shape.features:
            raise InvalidParameterError(
                "target_features must be >= input features for latent packing, "
                f"got target={target_features}, input={inputs.out_shape.features}."
            )
        if (perm is None) ^ (perm_inv is None):
            raise InvalidParameterError("perm and perm_inv must either both be set or both be None.")
        if perm is not None and perm.shape[0] != target_features:
            raise InvalidParameterError(
                f"perm length must equal target_features={target_features}, got {perm.shape[0]}."
            )

        self.inputs = inputs
        self.scope = inputs.scope
        self.in_shape = inputs.out_shape
        self.out_shape = ModuleShape(
            features=target_features,
            channels=inputs.out_shape.channels,
            repetitions=inputs.out_shape.repetitions,
        )
        self.target_features = target_features
        self.perm = perm
        self.perm_inv = perm_inv

        base_f2s = np.empty((target_features, self.out_shape.repetitions), dtype=object)
        for r in range(self.out_shape.repetitions):
            for f in range(target_features):
                base_f2s[f, r] = Scope([])
        base_f2s[: self.in_shape.features, :] = inputs.feature_to_scope
        if self.perm is not None:
            base_f2s = base_f2s[self.perm.detach().cpu().numpy(), :]
        self._feature_to_scope = base_f2s

    @property
    def feature_to_scope(self) -> np.ndarray:
        return self._feature_to_scope

    @cached
    def log_likelihood(self, data: Tensor, cache: Cache | None = None) -> Tensor:
        ll = self.inputs.log_likelihood(data, cache=cache)
        b, f, c, r = ll.shape
        if f != self.in_shape.features:
            raise ShapeError(
                f"Unexpected feature size in latent packing layer: expected {self.in_shape.features}, got {f}."
            )

        packed = ll.new_zeros((b, self.target_features, c, r))
        packed[:, :f] = ll
        if self.perm is not None:
            packed = packed[:, self.perm.to(device=packed.device), :, :]
        return packed

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
    ) -> Tensor:
        sampling_ctx.validate_sampling_context(
            num_samples=data.shape[0],
            num_features=self.out_shape.features,
            num_channels=self.out_shape.channels,
            num_repetitions=self.out_shape.repetitions,
            allowed_feature_widths=(1, self.target_features, self.in_shape.features),
        )

        ctx_features = sampling_ctx.channel_index.shape[1]
        if ctx_features == self.in_shape.features:
            return self.inputs._sample(data=data, cache=cache, sampling_ctx=sampling_ctx)
        if ctx_features == 1:
            num_target_features = self.target_features
            if sampling_ctx.is_differentiable:
                parent_idx = repeat(sampling_ctx.channel_index, "b 1 c -> b f c", f=num_target_features)
            else:
                parent_idx = repeat(sampling_ctx.channel_index, "b 1 -> b f", f=num_target_features)
            parent_mask = repeat(sampling_ctx.mask, "b 1 -> b f", f=num_target_features)
        elif ctx_features == self.target_features:
            parent_idx = sampling_ctx.channel_index
            parent_mask = sampling_ctx.mask
        else:
            raise ShapeError(
                "SamplingContext feature dimension mismatch in latent packing layer: "
                f"got {ctx_features}, expected 1, {self.target_features}, or {self.in_shape.features}."
            )

        if self.perm_inv is not None:
            perm_inv = self.perm_inv.to(device=parent_idx.device)
            if sampling_ctx.is_differentiable:
                parent_idx = parent_idx[:, perm_inv, :]
            else:
                parent_idx = parent_idx[:, perm_inv]
            parent_mask = parent_mask[:, perm_inv]

        if sampling_ctx.is_differentiable:
            child_idx = parent_idx[:, : self.in_shape.features, :]
        else:
            child_idx = parent_idx[:, : self.in_shape.features]
        child_mask = parent_mask[:, : self.in_shape.features]
        sampling_ctx.channel_index = child_idx
        sampling_ctx.mask = child_mask
        return self.inputs._sample(data=data, cache=cache, sampling_ctx=sampling_ctx)

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Module | None:
        del marg_rvs, prune, cache
        raise NotImplementedError("Marginalization is not implemented for _LatentFeaturePacking.")


class ConvPcJointEncoder(JointPcEncoderBase):
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
        architecture: Literal["reference", "legacy"] = "reference",
        perm_latents: bool = False,
        latent_channels: int | None = None,
        x_leaf_channels: int | None = None,
        x_leaf_factory: LeafFactory | None = None,
        z_leaf_factory: LeafFactory | None = None,
    ) -> None:
        """Initialize a Conv-PC APC encoder."""
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
        if architecture not in {"reference", "legacy"}:
            raise InvalidParameterError(
                f"architecture must be one of {{'reference', 'legacy'}}, got '{architecture}'."
            )
        if architecture == "reference":
            if depth < 2:
                raise InvalidParameterError(f"reference architecture requires depth >= 2, got depth={depth}.")
            if latent_depth < 0 or latent_depth >= depth:
                raise InvalidParameterError(
                    f"latent_depth must be in [0, {depth - 1}] for reference architecture, got {latent_depth}."
                )
        else:
            if latent_depth < 0 or latent_depth >= depth:
                raise InvalidParameterError(
                    f"latent_depth must be in [0, {depth - 1}] for legacy architecture, got {latent_depth}."
                )
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.num_x_features = input_height * input_width * input_channels
        self.latent_dim = latent_dim
        self.architecture = architecture
        self.perm_latents = perm_latents

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
        self._z_leaf = z_leaf

        self.layers = nn.ModuleList()
        self.latent_sum_layer: Module | None = None
        self.latent_prod_layers = nn.ModuleList()
        self.register_buffer("latent_perm", None)
        self.register_buffer("latent_perm_inv", None)
        self._latent_target_features = latent_dim

        if architecture == "reference":
            self.pc = self._build_joint_convpc_reference(
                x_leaf=x_leaf,
                z_leaf=z_leaf,
                channels=channels,
                depth=depth,
                kernel_size=kernel_size,
                num_repetitions=num_repetitions,
                use_sum_conv=use_sum_conv,
                latent_depth=latent_depth,
                latent_channels=latent_channels,
                perm_latents=perm_latents,
            )
        else:
            self.pc = self._build_joint_convpc_legacy(
                x_leaf=x_leaf,
                z_leaf=z_leaf,
                channels=channels,
                depth=depth,
                kernel_size=kernel_size,
                num_repetitions=num_repetitions,
                use_sum_conv=use_sum_conv,
                latent_depth=latent_depth,
            )

    def _build_joint_convpc_legacy(
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
    ) -> Module:
        """Build legacy Conv-PC architecture with strict latent feature matching."""
        layer_specs: list[tuple[str, dict[str, int]]] = [("sum_root", {"out_channels": 1})]

        h, w = 1, 1
        for _ in reversed(range(depth)):
            layer_specs.append(("prod", {"kernel_size": kernel_size}))
            h, w = h * kernel_size, w * kernel_size
            layer_specs.append(("sum", {"out_channels": channels, "kernel_size": kernel_size}))

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

        current: Module = x_leaf
        layers_built: list[Module] = []
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
                    target_features = current.out_shape.features
                    if target_features != self.latent_dim:
                        raise InvalidParameterError(
                            "latent_dim must match the feature count at latent injection depth "
                            f"in legacy architecture. Expected {target_features}, got {self.latent_dim}."
                        )

                    latent_stream: Module = z_leaf
                    if z_leaf.out_shape.channels != current.out_shape.channels:
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

            layers_built.append(current)

        if not latent_injected:
            raise RuntimeError("Latent branch was not injected. Check latent_depth configuration.")

        if num_repetitions > 1:
            current = RepetitionMixingLayer(inputs=current, out_channels=1, num_repetitions=num_repetitions)
            layers_built.append(current)

        self.layers = nn.ModuleList(layers_built)
        self.latent_prod_layers = nn.ModuleList()
        self.latent_sum_layer = None
        return current

    def _build_reference_latent_stream(
        self,
        *,
        z_leaf: LeafModule,
        target_features: int,
        channels: int,
        num_repetitions: int,
        perm_latents: bool,
    ) -> Module:
        """Build reference latent branch: leaf -> product reductions -> packing -> sum."""
        latent_stream: Module = z_leaf

        latent_prod_layers: list[Module] = []
        reduced_features = self.latent_dim
        while reduced_features > target_features:
            if reduced_features % 2 != 0:
                raise InvalidParameterError(
                    "reference latent product reduction requires even latent width at each step. "
                    f"Got latent width {reduced_features} with target {target_features}."
                )
            prod_layer = _PairwiseLatentProduct(inputs=latent_stream)
            latent_prod_layers.append(prod_layer)
            latent_stream = prod_layer
            reduced_features = prod_layer.out_shape.features

        self.latent_prod_layers = nn.ModuleList(latent_prod_layers)

        needs_packing = reduced_features < target_features or perm_latents
        if needs_packing:
            if perm_latents:
                self.latent_perm = torch.randperm(target_features)
                self.latent_perm_inv = torch.argsort(self.latent_perm)
            else:
                self.latent_perm = None
                self.latent_perm_inv = None

            latent_stream = _LatentFeaturePacking(
                inputs=latent_stream,
                target_features=target_features,
                perm=self.latent_perm,
                perm_inv=self.latent_perm_inv,
            )
        else:
            self.latent_perm = None
            self.latent_perm_inv = None

        self._latent_target_features = target_features
        self.latent_sum_layer = Sum(
            inputs=latent_stream,
            out_channels=channels,
            num_repetitions=num_repetitions,
        )
        return self.latent_sum_layer

    def _build_joint_convpc_reference(
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
        latent_channels: int,
        perm_latents: bool,
    ) -> Module:
        """Build reference-parity Conv-PC topology.

        Reference semantics use ``depth - 1`` intermediate sum-product stages.
        """
        del latent_channels
        layer_specs: list[tuple[str, dict[str, int]]] = [("sum_root", {"out_channels": 1})]

        h, w = 1, 1
        for _ in reversed(range(0, depth - 1)):
            layer_specs.append(("prod", {"kernel_size": kernel_size}))
            h, w = h * kernel_size, w * kernel_size
            layer_specs.append(("sum", {"out_channels": channels, "kernel_size": kernel_size}))

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

        current: Module = x_leaf
        layers_built: list[Module] = []
        latent_injected = False
        prod_stage = -1
        latent_stream: Module | None = None

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

            if layer_type in {"prod_bottom", "prod"}:
                prod_stage += 1
                if prod_stage == latent_depth:
                    if latent_stream is None:
                        latent_stream = self._build_reference_latent_stream(
                            z_leaf=z_leaf,
                            target_features=current.out_shape.features,
                            channels=channels,
                            num_repetitions=num_repetitions,
                            perm_latents=perm_latents,
                        )
                    if current.out_shape.features != latent_stream.out_shape.features:
                        raise ShapeError(
                            "Latent fusion feature mismatch in reference Conv-PC: "
                            f"pc features={current.out_shape.features}, latent features={latent_stream.out_shape.features}."
                        )
                    if current.out_shape.channels != latent_stream.out_shape.channels:
                        raise ShapeError(
                            "Latent fusion channel mismatch in reference Conv-PC: "
                            f"pc channels={current.out_shape.channels}, latent channels={latent_stream.out_shape.channels}."
                        )
                    if current.out_shape.repetitions != latent_stream.out_shape.repetitions:
                        raise ShapeError(
                            "Latent fusion repetition mismatch in reference Conv-PC: "
                            f"pc reps={current.out_shape.repetitions}, latent reps={latent_stream.out_shape.repetitions}."
                        )
                    # Additive fusion in log-space is equivalent to elementwise product in probability space.
                    current = ElementwiseProduct(inputs=[current, latent_stream])
                    latent_injected = True

            layers_built.append(current)

        if not latent_injected:
            raise RuntimeError("Latent branch was not injected. Check latent_depth configuration.")

        if num_repetitions > 1:
            current = RepetitionMixingLayer(inputs=current, out_channels=1, num_repetitions=num_repetitions)
            layers_built.append(current)

        self.layers = nn.ModuleList(layers_built)
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
