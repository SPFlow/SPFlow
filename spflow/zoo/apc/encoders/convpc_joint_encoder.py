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
from spflow.utils.cache import Cache
from spflow.utils.diff_sampling import DiffSampleMethod, select_with_soft_or_hard
from spflow.utils.diff_sampling_context import DifferentiableSamplingContext
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context
from spflow.zoo.apc.debug_trace import trace_sampling_context, trace_tensor
from spflow.zoo.apc.encoders.base import LatentStats
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

    def log_likelihood(self, data: Tensor, cache: Cache | None = None) -> Tensor:
        if cache is None:
            cache = Cache()
        ll = self.inputs.log_likelihood(data, cache=cache)
        b, f, c, r = ll.shape
        if f != self.in_shape.features:
            raise ShapeError(
                f"Unexpected feature size in latent product layer: expected {self.in_shape.features}, got {f}."
            )
        return ll.view(b, self.out_shape.features, 2, c, r).sum(dim=2)

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        if cache is None:
            cache = Cache()
        data = self._prepare_sample_data(num_samples, data)
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0], data.device)

        ctx_features = sampling_ctx.channel_index.shape[1]
        if ctx_features == self.in_shape.features:
            return self.inputs.sample(data=data, is_mpe=is_mpe, cache=cache, sampling_ctx=sampling_ctx)
        if ctx_features == 1:
            parent_idx = sampling_ctx.channel_index.expand(-1, self.out_shape.features)
            parent_mask = sampling_ctx.mask.expand(-1, self.out_shape.features)
        elif ctx_features == self.out_shape.features:
            parent_idx = sampling_ctx.channel_index
            parent_mask = sampling_ctx.mask
        else:
            raise ShapeError(
                "SamplingContext feature dimension mismatch in latent product layer: "
                f"got {ctx_features}, expected 1, {self.out_shape.features}, or {self.in_shape.features}."
            )

        child_idx = parent_idx.repeat_interleave(2, dim=1)
        child_mask = parent_mask.repeat_interleave(2, dim=1)
        sampling_ctx.update(child_idx, child_mask)
        return self.inputs.sample(data=data, is_mpe=is_mpe, cache=cache, sampling_ctx=sampling_ctx)

    def rsample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
        method: str = "simple",
        tau: float = 1.0,
        hard: bool = True,
    ) -> Tensor:
        if cache is None:
            cache = Cache()
        data = self._prepare_sample_data(num_samples, data)
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0], data.device)

        ctx_features = sampling_ctx.channel_index.shape[1]
        if ctx_features == self.in_shape.features:
            return self.inputs.rsample(
                data=data,
                is_mpe=is_mpe,
                cache=cache,
                sampling_ctx=sampling_ctx,
                method=method,
                tau=tau,
                hard=hard,
            )
        if ctx_features == 1:
            parent_idx = sampling_ctx.channel_index.expand(-1, self.out_shape.features)
            parent_mask = sampling_ctx.mask.expand(-1, self.out_shape.features)
        elif ctx_features == self.out_shape.features:
            parent_idx = sampling_ctx.channel_index
            parent_mask = sampling_ctx.mask
        else:
            raise ShapeError(
                "SamplingContext feature dimension mismatch in latent product layer: "
                f"got {ctx_features}, expected 1, {self.out_shape.features}, or {self.in_shape.features}."
            )

        child_idx = parent_idx.repeat_interleave(2, dim=1)
        child_mask = parent_mask.repeat_interleave(2, dim=1)
        sampling_ctx.update(child_idx, child_mask)
        return self.inputs.rsample(
            data=data,
            is_mpe=is_mpe,
            cache=cache,
            sampling_ctx=sampling_ctx,
            method=method,
            tau=tau,
            hard=hard,
        )

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

    def log_likelihood(self, data: Tensor, cache: Cache | None = None) -> Tensor:
        if cache is None:
            cache = Cache()
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

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        if cache is None:
            cache = Cache()
        data = self._prepare_sample_data(num_samples, data)
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0], data.device)

        ctx_features = sampling_ctx.channel_index.shape[1]
        if ctx_features == self.in_shape.features:
            return self.inputs.sample(data=data, is_mpe=is_mpe, cache=cache, sampling_ctx=sampling_ctx)
        if ctx_features == 1:
            parent_idx = sampling_ctx.channel_index.expand(-1, self.target_features)
            parent_mask = sampling_ctx.mask.expand(-1, self.target_features)
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
            parent_idx = parent_idx[:, perm_inv]
            parent_mask = parent_mask[:, perm_inv]

        child_idx = parent_idx[:, : self.in_shape.features]
        child_mask = parent_mask[:, : self.in_shape.features]
        sampling_ctx.update(child_idx, child_mask)
        return self.inputs.sample(data=data, is_mpe=is_mpe, cache=cache, sampling_ctx=sampling_ctx)

    def rsample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
        method: str = "simple",
        tau: float = 1.0,
        hard: bool = True,
    ) -> Tensor:
        if cache is None:
            cache = Cache()
        data = self._prepare_sample_data(num_samples, data)
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0], data.device)

        ctx_features = sampling_ctx.channel_index.shape[1]
        if ctx_features == self.in_shape.features:
            return self.inputs.rsample(
                data=data,
                is_mpe=is_mpe,
                cache=cache,
                sampling_ctx=sampling_ctx,
                method=method,
                tau=tau,
                hard=hard,
            )
        if ctx_features == 1:
            parent_idx = sampling_ctx.channel_index.expand(-1, self.target_features)
            parent_mask = sampling_ctx.mask.expand(-1, self.target_features)
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
            parent_idx = parent_idx[:, perm_inv]
            parent_mask = parent_mask[:, perm_inv]

        child_idx = parent_idx[:, : self.in_shape.features]
        child_mask = parent_mask[:, : self.in_shape.features]
        sampling_ctx.update(child_idx, child_mask)
        return self.inputs.rsample(
            data=data,
            is_mpe=is_mpe,
            cache=cache,
            sampling_ctx=sampling_ctx,
            method=method,
            tau=tau,
            hard=hard,
        )

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Module | None:
        del marg_rvs, prune, cache
        raise NotImplementedError("Marginalization is not implemented for _LatentFeaturePacking.")


class _LatentSelectionCapture(Module):
    """Identity wrapper that records leaf-level latent routing indices."""

    def __init__(
        self,
        inputs: Module,
        capture_fn: Callable[
            [Tensor, Tensor | None, Tensor | None, Tensor | None, bool],
            None,
        ],
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.scope = inputs.scope
        self.in_shape = inputs.out_shape
        self.out_shape = inputs.out_shape
        self._capture_fn = capture_fn

    @property
    def feature_to_scope(self) -> np.ndarray:
        return self.inputs.feature_to_scope

    def _capture(self, sampling_ctx: SamplingContext, data_num_features: int) -> None:
        ctx_features = sampling_ctx.channel_index.shape[1]
        target_features = self.in_shape.features

        if ctx_features == 1:
            channel_idx = sampling_ctx.channel_index.expand(-1, target_features)
        elif ctx_features == target_features:
            channel_idx = sampling_ctx.channel_index
        elif ctx_features == data_num_features:
            scope_cols = list(self.scope.query)
            channel_idx = sampling_ctx.channel_index[:, scope_cols]
        else:
            channel_idx = sampling_ctx.channel_index[:, :1].expand(-1, target_features)

        rep_idx = sampling_ctx.repetition_idx
        if rep_idx is None:
            repetition_idx = None
        elif rep_idx.dim() == 1:
            repetition_idx = rep_idx.unsqueeze(1).expand(-1, target_features)
        elif rep_idx.shape[1] == 1:
            repetition_idx = rep_idx.expand(-1, target_features)
        elif rep_idx.shape[1] == target_features:
            repetition_idx = rep_idx
        elif rep_idx.shape[1] == data_num_features:
            scope_cols = list(self.scope.query)
            repetition_idx = rep_idx[:, scope_cols]
        else:
            repetition_idx = rep_idx[:, :1].expand(-1, target_features)

        channel_select = getattr(sampling_ctx, "channel_select", None)
        if channel_select is not None and channel_select.dim() == 3:
            if channel_select.shape[1] == 1:
                channel_select = channel_select.expand(-1, target_features, -1)
            elif channel_select.shape[1] == target_features:
                pass
            elif channel_select.shape[1] == data_num_features:
                scope_cols = list(self.scope.query)
                channel_select = channel_select[:, scope_cols, :]
            else:
                channel_select = channel_select[:, :1, :].expand(-1, target_features, -1)
        else:
            channel_select = None

        repetition_select = getattr(sampling_ctx, "repetition_select", None)
        if repetition_select is not None and repetition_select.dim() == 3:
            if repetition_select.shape[1] == 1:
                repetition_select = repetition_select.expand(-1, target_features, -1)
            elif repetition_select.shape[1] == target_features:
                pass
            elif repetition_select.shape[1] == data_num_features:
                scope_cols = list(self.scope.query)
                repetition_select = repetition_select[:, scope_cols, :]
            else:
                repetition_select = repetition_select[:, :1, :].expand(-1, target_features, -1)
        else:
            repetition_select = None

        keep_selectors_attached = isinstance(sampling_ctx, DifferentiableSamplingContext)

        self._capture_fn(
            channel_idx.detach().clone().to(dtype=torch.long),
            None if repetition_idx is None else repetition_idx.detach().clone().to(dtype=torch.long),
            None if channel_select is None else channel_select.clone(),
            None if repetition_select is None else repetition_select.clone(),
            keep_selectors_attached,
        )

    def log_likelihood(self, data: Tensor, cache: Cache | None = None) -> Tensor:
        if cache is None:
            cache = Cache()
        return self.inputs.log_likelihood(data, cache=cache)

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        if cache is None:
            cache = Cache()
        data = self._prepare_sample_data(num_samples, data)
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0], data.device)
        self._capture(sampling_ctx, data.shape[1])
        return self.inputs.sample(data=data, is_mpe=is_mpe, cache=cache, sampling_ctx=sampling_ctx)

    def rsample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
        method: str = "simple",
        tau: float = 1.0,
        hard: bool = True,
    ) -> Tensor:
        if cache is None:
            cache = Cache()
        data = self._prepare_sample_data(num_samples, data)
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0], data.device)
        self._capture(sampling_ctx, data.shape[1])
        return self.inputs.rsample(
            data=data,
            is_mpe=is_mpe,
            cache=cache,
            sampling_ctx=sampling_ctx,
            method=method,
            tau=tau,
            hard=hard,
        )

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Module | None:
        del marg_rvs, prune, cache
        raise NotImplementedError("Marginalization is not implemented for _LatentSelectionCapture.")


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
        architecture: Literal["reference", "legacy"] = "reference",
        perm_latents: bool = False,
        latent_channels: int | None = None,
        x_leaf_channels: int | None = None,
        x_leaf_factory: LeafFactory | None = None,
        z_leaf_factory: LeafFactory | None = None,
        posterior_stat_samples: int = 4,
        posterior_var_floor: float = 1e-6,
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
            if latent_depth < 0 or latent_depth >= (depth - 1):
                raise InvalidParameterError(
                    f"latent_depth must be in [0, {depth - 2}] for reference architecture, got {latent_depth}."
                )
        else:
            if latent_depth < 0 or latent_depth >= depth:
                raise InvalidParameterError(
                    f"latent_depth must be in [0, {depth - 1}] for legacy architecture, got {latent_depth}."
                )
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
        self.latent_perm: Tensor | None = None
        self.latent_perm_inv: Tensor | None = None
        self._latent_target_features = latent_dim

        self._last_latent_leaf_channel_index: Tensor | None = None
        self._last_latent_leaf_repetition_index: Tensor | None = None
        self._last_latent_leaf_channel_select: Tensor | None = None
        self._last_latent_leaf_repetition_select: Tensor | None = None

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

    def _record_latent_leaf_selection(
        self,
        channel_index: Tensor,
        repetition_idx: Tensor | None,
        channel_select: Tensor | None,
        repetition_select: Tensor | None,
        keep_selectors_attached: bool,
    ) -> None:
        """Capture leaf-level latent routing signals from the sampling path."""
        self._last_latent_leaf_channel_index = channel_index
        self._last_latent_leaf_repetition_index = repetition_idx
        if channel_select is None:
            self._last_latent_leaf_channel_select = None
        elif keep_selectors_attached:
            self._last_latent_leaf_channel_select = channel_select
        else:
            self._last_latent_leaf_channel_select = channel_select.detach().clone()

        if repetition_select is None:
            self._last_latent_leaf_repetition_select = None
        elif keep_selectors_attached:
            self._last_latent_leaf_repetition_select = repetition_select
        else:
            self._last_latent_leaf_repetition_select = repetition_select.detach().clone()

    def _reset_latent_leaf_selection(self) -> None:
        self._last_latent_leaf_channel_index = None
        self._last_latent_leaf_repetition_index = None
        self._last_latent_leaf_channel_select = None
        self._last_latent_leaf_repetition_select = None

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
        """Build reference latent branch: capture -> product reductions -> packing -> sum."""
        latent_stream: Module = _LatentSelectionCapture(
            inputs=z_leaf,
            capture_fn=self._record_latent_leaf_selection,
        )

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
        latent_target_features: int | None = None
        for top_idx in reversed(range(0, depth - 1)):
            layer_specs.append(("prod", {"kernel_size": kernel_size}))
            h, w = h * kernel_size, w * kernel_size
            layer_specs.append(("sum", {"out_channels": channels, "kernel_size": kernel_size}))
            if top_idx == latent_depth:
                latent_target_features = h * w

        if latent_target_features is None:
            raise RuntimeError("Failed to infer latent injection feature width for reference architecture.")

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

        latent_stream = self._build_reference_latent_stream(
            z_leaf=z_leaf,
            target_features=latent_target_features,
            channels=channels,
            num_repetitions=num_repetitions,
            perm_latents=perm_latents,
        )

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
                    # Additive fusion in log-space is equivalent to elementwise product in probability space.
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

    def _posterior_sample(
        self, x_flat: Tensor, *, mpe: bool, tau: float, return_sampling_ctx: bool = False
    ) -> Tensor | tuple[Tensor, SamplingContext]:
        """Sample ``z ~ p(Z|X=x)`` and optionally return sampling context."""
        self._reset_latent_leaf_selection()
        trace_tensor("convpc.posterior.x_flat", x_flat)
        evidence = self._build_evidence(x_flat=x_flat, z_flat=None)
        trace_tensor("convpc.posterior.evidence", evidence)
        if mpe:
            sampling_ctx: SamplingContext = SamplingContext(num_samples=x_flat.shape[0], device=x_flat.device)
        else:
            # Keep one shared differentiable context so routing selectors propagate back
            # to APC latent-stat extraction and tracing.
            sampling_ctx = DifferentiableSamplingContext(
                num_samples=x_flat.shape[0],
                device=x_flat.device,
                method=DiffSampleMethod.SIMPLE,
                tau=tau,
                hard=True,
                skip_rsample_noise=return_sampling_ctx and isinstance(self._z_leaf, Normal),
            )
        trace_sampling_context("convpc.posterior.ctx_init", sampling_ctx)
        if mpe:
            joint = self.pc.sample(data=evidence, is_mpe=True, sampling_ctx=sampling_ctx)
        else:
            joint = self.pc.rsample(
                data=evidence, is_mpe=False, tau=tau, hard=True, sampling_ctx=sampling_ctx
            )
        trace_tensor("convpc.posterior.joint", joint)
        trace_sampling_context("convpc.posterior.ctx_out", sampling_ctx)
        z = joint[:, self._z_cols]
        trace_tensor("convpc.posterior.z", z)
        if return_sampling_ctx:
            return z, sampling_ctx
        return z

    def _resolve_latent_channel_indices(
        self,
        *,
        sampling_ctx: SamplingContext,
        batch_size: int,
        loc: Tensor,
    ) -> Tensor:
        """Resolve leaf-level latent component indices for posterior stats."""
        if self._last_latent_leaf_channel_index is not None:
            channel_idx = self._last_latent_leaf_channel_index.to(device=loc.device, dtype=torch.long)
            if channel_idx.shape[0] != batch_size:
                raise ShapeError(
                    "Captured latent leaf channel index batch mismatch: "
                    f"expected {batch_size}, got {channel_idx.shape[0]}."
                )
            if channel_idx.shape[1] == 1:
                channel_idx = channel_idx.expand(-1, self.latent_dim)
            elif channel_idx.shape[1] >= self.latent_dim:
                channel_idx = channel_idx[:, : self.latent_dim]
            else:
                raise ShapeError(
                    "Captured latent leaf channel index feature mismatch: "
                    f"expected at least {self.latent_dim}, got {channel_idx.shape[1]}."
                )
            return channel_idx.clamp(min=0, max=loc.shape[1] - 1)

        channels = sampling_ctx.channel_index
        ctx_features = channels.shape[1]
        if ctx_features == 1:
            channel_idx = channels.expand(-1, self.latent_dim)
        elif ctx_features == self.latent_dim:
            channel_idx = channels
        elif ctx_features == (self.num_x_features + self.latent_dim):
            channel_idx = channels[:, self._z_cols]
        else:
            raise InvalidParameterError(
                f"Unexpected sampling context feature width {ctx_features} for latent stats extraction."
            )
        return channel_idx.to(device=loc.device, dtype=torch.long).clamp(min=0, max=loc.shape[1] - 1)

    def _resolve_latent_repetition_indices(
        self,
        *,
        sampling_ctx: SamplingContext,
        batch_size: int,
        loc: Tensor,
    ) -> Tensor:
        """Resolve leaf-level latent repetition indices for posterior stats."""
        rep = self._last_latent_leaf_repetition_index
        if rep is not None:
            repetition_idx = rep.to(device=loc.device, dtype=torch.long)
            if repetition_idx.shape[0] != batch_size:
                raise ShapeError(
                    "Captured latent repetition index batch mismatch: "
                    f"expected {batch_size}, got {repetition_idx.shape[0]}."
                )
            if repetition_idx.shape[1] == 1:
                repetition_idx = repetition_idx.expand(-1, self.latent_dim)
            elif repetition_idx.shape[1] >= self.latent_dim:
                repetition_idx = repetition_idx[:, : self.latent_dim]
            else:
                raise ShapeError(
                    "Captured latent repetition index feature mismatch: "
                    f"expected at least {self.latent_dim}, got {repetition_idx.shape[1]}."
                )
            return repetition_idx.clamp(min=0, max=loc.shape[2] - 1)

        if sampling_ctx.repetition_idx is None:
            return torch.zeros((batch_size, self.latent_dim), dtype=torch.long, device=loc.device)

        rep = sampling_ctx.repetition_idx
        if rep.dim() == 1:
            repetition_idx = rep.unsqueeze(1).expand(-1, self.latent_dim)
        elif rep.shape[1] == 1:
            repetition_idx = rep.expand(-1, self.latent_dim)
        elif rep.shape[1] == self.latent_dim:
            repetition_idx = rep
        elif rep.shape[1] == (self.num_x_features + self.latent_dim):
            repetition_idx = rep[:, self._z_cols]
        else:
            repetition_idx = rep[:, :1].expand(-1, self.latent_dim)
        return repetition_idx.to(device=loc.device, dtype=torch.long).clamp(min=0, max=loc.shape[2] - 1)

    def _resolve_latent_channel_selector(
        self,
        *,
        sampling_ctx: SamplingContext,
        batch_size: int,
        loc: Tensor,
    ) -> Tensor | None:
        """Resolve soft/hard channel selectors for latent leaf extraction."""
        selector = self._last_latent_leaf_channel_select
        if selector is None:
            selector = getattr(sampling_ctx, "channel_select", None)
        if selector is None or selector.dim() != 3:
            return None
        if selector.shape[0] != batch_size:
            raise ShapeError(
                "Latent channel selector batch mismatch: " f"expected {batch_size}, got {selector.shape[0]}."
            )
        if selector.shape[1] == 1:
            selector = selector.expand(-1, self.latent_dim, -1)
        elif selector.shape[1] == self.latent_dim:
            pass
        elif selector.shape[1] == (self.num_x_features + self.latent_dim):
            selector = selector[:, self._z_cols, :]
        elif selector.shape[1] >= self.latent_dim:
            selector = selector[:, : self.latent_dim, :]
        else:
            raise ShapeError(
                "Latent channel selector feature mismatch: "
                f"expected at least {self.latent_dim}, got {selector.shape[1]}."
            )
        if selector.shape[2] != loc.shape[1]:
            return None
        return selector.to(device=loc.device, dtype=loc.dtype)

    def _resolve_latent_repetition_selector(
        self,
        *,
        sampling_ctx: SamplingContext,
        batch_size: int,
        loc: Tensor,
    ) -> Tensor | None:
        """Resolve soft/hard repetition selectors for latent leaf extraction."""
        selector = self._last_latent_leaf_repetition_select
        if selector is None:
            selector = getattr(sampling_ctx, "repetition_select", None)
        if selector is None or selector.dim() != 3:
            return None
        if selector.shape[0] != batch_size:
            raise ShapeError(
                "Latent repetition selector batch mismatch: "
                f"expected {batch_size}, got {selector.shape[0]}."
            )
        if selector.shape[1] == 1:
            selector = selector.expand(-1, self.latent_dim, -1)
        elif selector.shape[1] == self.latent_dim:
            pass
        elif selector.shape[1] == (self.num_x_features + self.latent_dim):
            selector = selector[:, self._z_cols, :]
        elif selector.shape[1] >= self.latent_dim:
            selector = selector[:, : self.latent_dim, :]
        else:
            raise ShapeError(
                "Latent repetition selector feature mismatch: "
                f"expected at least {self.latent_dim}, got {selector.shape[1]}."
            )
        if selector.shape[2] != loc.shape[2]:
            return None
        return selector.to(device=loc.device, dtype=loc.dtype)

    def _latent_stats_from_leaf_params(self, sampling_ctx: SamplingContext, batch_size: int) -> LatentStats:
        """Extract posterior ``mu/logvar`` from selected latent leaf parameters."""
        if not isinstance(self._z_leaf, Normal):
            raise InvalidParameterError("Latent stats from leaf params require a Normal latent leaf.")

        loc = self._z_leaf.loc
        scale = self._z_leaf.scale.clamp_min(self.posterior_var_floor**0.5)
        if loc.dim() != 3 or scale.dim() != 3:
            raise InvalidParameterError("Unexpected latent leaf parameter shape for Normal leaf.")

        loc_selected = loc.unsqueeze(0).expand(batch_size, -1, -1, -1)
        scale_selected = scale.unsqueeze(0).expand(batch_size, -1, -1, -1)

        repetition_selector = self._resolve_latent_repetition_selector(
            sampling_ctx=sampling_ctx,
            batch_size=batch_size,
            loc=loc,
        )
        if repetition_selector is not None:
            repetition_selector = repetition_selector.unsqueeze(2)  # (B, latent_dim, 1, repetitions)
            loc_selected = select_with_soft_or_hard(
                loc_selected,
                selector=repetition_selector,
                dim=3,
            )
            scale_selected = select_with_soft_or_hard(
                scale_selected,
                selector=repetition_selector,
                dim=3,
            )
        else:
            repetition_idx = self._resolve_latent_repetition_indices(
                sampling_ctx=sampling_ctx,
                batch_size=batch_size,
                loc=loc,
            )
            repetition_gather = (
                repetition_idx.unsqueeze(2).unsqueeze(3).expand(-1, -1, loc_selected.shape[2], 1)
            )
            loc_selected = loc_selected.gather(dim=3, index=repetition_gather).squeeze(3)
            scale_selected = scale_selected.gather(dim=3, index=repetition_gather).squeeze(3)

        channel_selector = self._resolve_latent_channel_selector(
            sampling_ctx=sampling_ctx,
            batch_size=batch_size,
            loc=loc,
        )
        if channel_selector is not None:
            mu = select_with_soft_or_hard(
                loc_selected,
                selector=channel_selector,
                dim=2,
            )
            sel_scale = select_with_soft_or_hard(
                scale_selected,
                selector=channel_selector,
                dim=2,
            )
        else:
            channel_idx = self._resolve_latent_channel_indices(
                sampling_ctx=sampling_ctx,
                batch_size=batch_size,
                loc=loc,
            )
            channel_gather = channel_idx.unsqueeze(2)
            mu = loc_selected.gather(dim=2, index=channel_gather).squeeze(2)
            sel_scale = scale_selected.gather(dim=2, index=channel_gather).squeeze(2)

        logvar = (sel_scale.pow(2)).clamp_min(self.posterior_var_floor).log()
        return LatentStats(mu=mu, logvar=logvar)

    def _latent_stats_mc_fallback(
        self,
        *,
        x_flat: Tensor,
        first_sample: Tensor,
        mpe: bool,
        tau: float,
    ) -> LatentStats:
        """Fallback posterior stats estimation for non-Normal latent leaves."""
        samples = [first_sample]
        for _ in range(self.posterior_stat_samples - 1):
            sample_i = self._posterior_sample(x_flat, mpe=mpe, tau=tau, return_sampling_ctx=False)
            if not isinstance(sample_i, torch.Tensor):
                raise RuntimeError("Unexpected posterior sample type in MC stats fallback.")
            samples.append(sample_i)

        z_stack = torch.stack(samples, dim=0)
        mu = z_stack.mean(dim=0)
        var = z_stack.var(dim=0, unbiased=False).clamp_min(self.posterior_var_floor)
        return LatentStats(mu=mu, logvar=var.log())

    def encode(
        self,
        x: Tensor,
        *,
        mpe: bool = False,
        tau: float = 1.0,
        return_latent_stats: bool = False,
    ) -> Tensor | tuple[LatentStats, Tensor]:
        """Encode observations into latent samples."""
        trace_tensor("convpc.encode.x_in", x)
        x_flat = self._flatten_x(x)
        z_out = self._posterior_sample(x_flat, mpe=mpe, tau=tau, return_sampling_ctx=return_latent_stats)

        if not return_latent_stats:
            if isinstance(z_out, Tensor):
                trace_tensor("convpc.encode.z_out", z_out)
            return z_out

        z_sample, sampling_ctx = z_out
        if isinstance(self._z_leaf, Normal):
            stats = self._latent_stats_from_leaf_params(
                sampling_ctx=sampling_ctx,
                batch_size=z_sample.shape[0],
            )
            # Match reference contract: reparameterize from selected leaf parameters.
            z = tau * torch.randn_like(stats.mu) * torch.exp(0.5 * stats.logvar) + stats.mu
            trace_tensor("convpc.encode.stats.mu", stats.mu)
            trace_tensor("convpc.encode.stats.logvar", stats.logvar)
            trace_tensor("convpc.encode.z_out", z)
            return stats, z

        # Explicit fallback only for non-Normal latent leaves.
        stats = self._latent_stats_mc_fallback(x_flat=x_flat, first_sample=z_sample, mpe=mpe, tau=tau)
        trace_tensor("convpc.encode.stats.mu", stats.mu)
        trace_tensor("convpc.encode.stats.logvar", stats.logvar)
        trace_tensor("convpc.encode.z_out", z_sample)
        return stats, z_sample

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
        trace_tensor("convpc.decode.z_in", z)
        trace_tensor("convpc.decode.x_evidence_in", x)
        z_flat = self._flatten_z(z)
        x_flat = None if x is None else self._flatten_x(x)
        evidence = self._build_evidence(x_flat=x_flat, z_flat=z_flat)
        trace_tensor("convpc.decode.evidence", evidence)

        if mpe:
            joint = self.pc.sample(data=evidence, is_mpe=True)
        else:
            joint = self.pc.rsample(data=evidence, is_mpe=False, tau=tau, hard=True)
        trace_tensor("convpc.decode.joint", joint)

        x_rec_flat = joint[:, self._x_cols]
        if fill_evidence and x_flat is not None:
            finite_mask = torch.isfinite(x_flat)
            x_rec_flat = torch.where(finite_mask, x_flat.to(x_rec_flat.dtype), x_rec_flat)
        trace_tensor("convpc.decode.x_rec_flat", x_rec_flat)
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
        trace_tensor("convpc.prior.evidence", evidence)
        joint = self.pc.rsample(data=evidence, is_mpe=False, tau=tau, hard=True)
        trace_tensor("convpc.prior.joint", joint)
        z = joint[:, self._z_cols]
        trace_tensor("convpc.prior.z", z)
        return z

    def latent_stats(self, x: Tensor, *, tau: float = 1.0) -> LatentStats:
        """Return latent posterior stats for ``x``."""
        x_flat = self._flatten_x(x)
        z, sampling_ctx = self._posterior_sample(x_flat, mpe=False, tau=tau, return_sampling_ctx=True)
        if isinstance(self._z_leaf, Normal):
            return self._latent_stats_from_leaf_params(sampling_ctx=sampling_ctx, batch_size=z.shape[0])
        return self._latent_stats_mc_fallback(x_flat=x_flat, first_sample=z, mpe=False, tau=tau)
