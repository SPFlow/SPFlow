"""Folded tensorized QPC implementation (PIC → folded QPC).

This module implements a folded/tensorized QPC evaluation path inspired by the
authors' reference implementation in `reference-repos/ten-pics`.

Key features:
- Supports n-ary partitions (e.g., quad trees) via `fold_mask` padding.
- Parameterizes sum-product layers via ten-pics-style `InnerNet` with:
  - `perm_dim` tensor permutation after evaluation
  - `norm_dim` quadrature-weighted normalization
  - `n_chunks` chunked meshgrid evaluation to reduce peak memory
- Supports leaf types: Normal, Bernoulli, Categorical.

Note: This class is a SPFlow `Module` and therefore exposes `log_likelihood`.
Sampling/marginalization are not implemented yet for the folded representation.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from spflow.exceptions import InvalidParameterError, ShapeError, StructureError
from spflow.meta.region_graph import Region, RegionGraph
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.zoo.pic.pipeline import QuadratureRule
from spflow.zoo.pic.tensorized.utils import eval_collapsed_cp, eval_mixing, eval_tucker
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext


@dataclasses.dataclass(frozen=True)
class TensorizedQPCConfig:
    """Configuration for folded tensorized QPC materialization."""

    leaf_type: Literal["normal", "bernoulli", "categorical"]
    num_categories: int | None = None

    # Network settings (match ten-pics defaults).
    net_dim: int = 64
    bias: bool = False
    input_sharing: Literal["none", "f", "c"] = "none"
    inner_sharing: Literal["none", "f", "c"] = "none"
    ff_dim: int | None = None
    sigma: float = 1.0
    learn_ff: bool = False

    # Evaluation settings.
    n_chunks: int = 1

    # Circuit settings.
    num_classes: int = 1
    layer_cls: Literal["auto", "tucker", "cp"] = "auto"


@dataclasses.dataclass(frozen=True)
class IntegralGroupArgs:
    """Per-layer group metadata for InnerNet parameterization (ten-pics parity)."""

    num_dim: int
    num_funcs: int
    perm_dim: tuple[int, ...]
    norm_dim: tuple[int, ...]


class FourierLayer(nn.Module):
    """Random Fourier features layer returning channel-first features.

    This mirrors `reference-repos/ten-pics/pic.py:FourierLayer`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: float = 1.0,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        if out_features % 2 != 0:
            raise InvalidParameterError("FourierLayer out_features must be even.")
        coeff = torch.normal(0.0, sigma, (in_features, out_features // 2))
        if learnable:
            self.coeff = nn.Parameter(coeff)
        else:
            self.register_buffer("coeff", coeff)

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, z: Tensor) -> Tensor:
        # Accept unbatched (L, D) or batched (..., L, D).
        z_proj = 2 * torch.pi * z @ self.coeff  # (..., L, out/2)
        feats = torch.cat([z_proj.cos(), z_proj.sin()], dim=-1)  # (..., L, out)
        return feats.transpose(-2, -1)  # (..., out, L)


class InputNet(nn.Module):
    """Parameterizes leaf distributions f_u(X_u, Z_u=z) over 1D latent z."""

    def __init__(
        self,
        *,
        num_vars: int,
        num_param: int,
        net_dim: int = 64,
        bias: bool = False,
        sharing: Literal["none", "f", "c"] = "none",
        ff_dim: int | None = None,
        sigma: float = 1.0,
        learn_ff: bool = False,
    ) -> None:
        super().__init__()
        if sharing not in {"none", "f", "c"}:
            raise InvalidParameterError("sharing must be one of {'none','f','c'}.")
        self.num_vars = num_vars
        self.num_param = num_param
        self.sharing = sharing

        ff_dim_eff = net_dim if ff_dim is None else ff_dim

        # Grouping mirrors ten-pics. We keep num_channels==1 for now.
        inner_conv_groups = 1 if sharing in {"f", "c"} else num_vars
        last_conv_groups = 1 if sharing == "f" else num_vars

        self.net = nn.Sequential(
            FourierLayer(1, ff_dim_eff, sigma=sigma, learnable=learn_ff),
            nn.Conv1d(
                ff_dim_eff * inner_conv_groups,
                net_dim * inner_conv_groups,
                1,
                groups=inner_conv_groups,
                bias=bias,
            ),
            nn.Tanh(),
            nn.Conv1d(
                net_dim * last_conv_groups,
                num_param * last_conv_groups,
                1,
                groups=last_conv_groups,
                bias=bias,
            ),
        )

        # Initialize all heads to be equal when using composite sharing.
        if sharing == "c":
            self.net[-1].weight.data = self.net[-1].weight.data[:num_param].repeat(num_vars, 1, 1)
            if self.net[-1].bias is not None:
                self.net[-1].bias.data = self.net[-1].bias.data[:num_param].repeat(num_vars)

    def forward(self, z_quad: Tensor, *, n_chunks: int = 1) -> Tensor:
        if z_quad.ndim != 1:
            raise ShapeError("InputNet expects z_quad to be 1D.")
        if n_chunks <= 0:
            raise InvalidParameterError("n_chunks must be positive.")

        # ten-pics toggles conv groups at runtime to replicate shared features.
        self.net[1].groups = 1
        self.net[-1].groups = 1 if self.sharing in {"f", "c"} else self.num_vars

        z = z_quad.view(-1, 1)  # (K, 1)
        out = torch.cat([self.net(chunk) for chunk in z.chunk(n_chunks, dim=0)], dim=-1)  # (P*V, K)

        if self.sharing == "f":
            out = out.unsqueeze(0)  # (1, P, K)

        # Shape: (V, K, P)
        out = out.view(-1, self.num_param, z_quad.numel()).transpose(1, 2).contiguous()
        if out.shape[0] == 1 and self.num_vars > 1 and self.sharing == "f":
            out = out.expand(self.num_vars, -1, -1).contiguous()
        if out.shape[0] != self.num_vars:
            raise ShapeError(f"Expected {self.num_vars} vars (or 1 with F-sharing), got {out.shape[0]}.")
        return out


class InnerNet(nn.Module):
    """Parameterizes folded sum-product layer weights with quadrature normalization."""

    def __init__(
        self,
        *,
        group_args: IntegralGroupArgs,
        net_dim: int = 64,
        bias: bool = False,
        sharing: Literal["none", "f", "c"] = "none",
        ff_dim: int | None = None,
        sigma: float = 1.0,
        learn_ff: bool = False,
    ) -> None:
        super().__init__()
        if sharing not in {"none", "f", "c"}:
            raise InvalidParameterError("sharing must be one of {'none','f','c'}.")

        self.num_dim = group_args.num_dim
        self.num_funcs = group_args.num_funcs
        self.sharing = sharing

        perm_dim = (0,) + (
            tuple(range(1, self.num_dim + 1)) if len(group_args.perm_dim) == 0 else group_args.perm_dim
        )
        if perm_dim[0] != 0 or set(perm_dim) != set(range(self.num_dim + 1)):
            raise InvalidParameterError("Invalid perm_dim for InnerNet.")
        self.perm_dim = perm_dim

        if 0 in group_args.norm_dim or not set(group_args.norm_dim).issubset(set(perm_dim)):
            raise InvalidParameterError("Invalid norm_dim for InnerNet.")
        self.norm_dim = group_args.norm_dim

        self.eps = float(np.sqrt(torch.finfo(torch.get_default_dtype()).tiny))

        ff_dim_eff = net_dim if ff_dim is None else ff_dim
        inner_conv_groups = 1 if sharing in {"c", "f"} else self.num_funcs
        last_conv_groups = 1 if sharing == "f" else self.num_funcs

        self.net = nn.Sequential(
            FourierLayer(self.num_dim, ff_dim_eff, sigma=sigma, learnable=learn_ff),
            nn.Conv1d(
                inner_conv_groups * ff_dim_eff,
                inner_conv_groups * net_dim,
                1,
                groups=inner_conv_groups,
                bias=bias,
            ),
            nn.Tanh(),
            nn.Conv1d(
                inner_conv_groups * net_dim,
                inner_conv_groups * net_dim,
                1,
                groups=inner_conv_groups,
                bias=bias,
            ),
            nn.Tanh(),
            nn.Conv1d(last_conv_groups * net_dim, last_conv_groups, 1, groups=last_conv_groups, bias=bias),
            nn.Softplus(beta=1.0),
        )

        if sharing == "c":
            # Match ten-pics: initialize all heads equal for composite sharing.
            self.net[-2].weight.data = self.net[-2].weight.data[:1].repeat(self.num_funcs, 1, 1)
            if self.net[-2].bias is not None:
                self.net[-2].bias.data = self.net[-2].bias.data[:1].repeat(self.num_funcs)

    def forward(self, z_quad: Tensor, w_quad: Tensor, *, n_chunks: int = 1) -> Tensor:
        if z_quad.ndim != 1 or w_quad.ndim != 1 or z_quad.numel() != w_quad.numel():
            raise ShapeError("InnerNet expects z_quad and w_quad to be 1D and same length.")
        if n_chunks <= 0:
            raise InvalidParameterError("n_chunks must be positive.")

        nip = int(z_quad.numel())
        self.net[1].groups = 1
        self.net[-2].groups = 1 if self.sharing in {"c", "f"} else self.num_funcs

        z_meshgrid = torch.stack(torch.meshgrid([z_quad] * self.num_dim, indexing="ij")).flatten(1).t()

        logits = torch.cat([self.net(chunk) for chunk in z_meshgrid.chunk(n_chunks, dim=0)], dim=-1)
        logits = logits + self.eps

        # Expand does something when sharing == 'f'.
        logits = logits.expand(self.num_funcs, -1)
        logits = logits.view(-1, *([nip] * self.num_dim)).permute(self.perm_dim)

        w_shape = [nip if i in self.norm_dim else 1 for i in range(self.num_dim + 1)]
        w_meshgrid = (
            torch.stack(torch.meshgrid([w_quad] * len(self.norm_dim), indexing="ij")).prod(0).view(w_shape)
        )

        denom = (logits * w_meshgrid).sum(self.norm_dim, keepdim=True)
        param = (logits / denom) * w_meshgrid

        return param


@dataclasses.dataclass(frozen=True)
class _BookEntry:
    should_pad: bool
    in_layer_ids: list[int]
    fold_indices: Tensor  # (F, H), dtype long


@dataclasses.dataclass(frozen=True)
class _PartitionLayer:
    kind: Literal["tucker", "cp"]
    num_folds: int
    arity: int
    num_input_units: int
    num_output_units: int
    fold_mask: Tensor | None
    group_args: IntegralGroupArgs


@dataclasses.dataclass(frozen=True)
class _MixingLayer:
    num_folds: int
    arity: int
    num_units: int
    fold_mask: Tensor | None


class TensorizedQPC(Module):
    """Folded tensorized QPC as a SPFlow `Module`."""

    def __init__(
        self,
        *,
        rg: RegionGraph,
        quadrature_rule: QuadratureRule,
        config: TensorizedQPCConfig,
    ) -> None:
        super().__init__()

        if quadrature_rule.points.ndim != 1 or quadrature_rule.weights.ndim != 1:
            raise ShapeError("QuadratureRule points and weights must be 1D.")
        if quadrature_rule.points.shape[0] != quadrature_rule.weights.shape[0]:
            raise ShapeError("QuadratureRule points and weights must have the same length.")
        if config.n_chunks <= 0:
            raise InvalidParameterError("config.n_chunks must be positive.")
        if config.leaf_type == "categorical" and (
            config.num_categories is None or config.num_categories <= 1
        ):
            raise InvalidParameterError("num_categories must be provided (>1) for categorical leaves.")
        if config.num_classes <= 0:
            raise InvalidParameterError("num_classes must be positive.")

        self.rg = rg
        self.quadrature_rule = quadrature_rule
        self.config = config

        # Model scope: root scope.
        self.scope = rg.root.scope

        # Leaf order and validation.
        leaf_regions = [r for r in rg.post_order() if not r.children]
        if any(len(r.scope.query) != 1 for r in leaf_regions):
            raise StructureError("TensorizedQPC requires univariate leaf regions.")

        # Deterministic order: sort by RV index.
        leaf_regions = sorted(leaf_regions, key=lambda r: int(r.scope.query[0]))
        leaf_vars = [int(r.scope.query[0]) for r in leaf_regions]
        if len(set(leaf_vars)) != len(leaf_vars):
            raise StructureError("TensorizedQPC requires unique leaf variables.")

        self._leaf_regions = leaf_regions
        self._leaf_vars = torch.tensor(leaf_vars, dtype=torch.long)

        self.num_vars = int(len(leaf_vars))
        self.num_units = int(quadrature_rule.points.shape[0])

        # Build folded bookkeeping and layer descriptors.
        self.bookkeeping, self.partition_layers, self.mixing_layers, self.eval_plan = _build_folded_layers(
            rg=rg,
            leaf_regions=leaf_regions,
            num_units=self.num_units,
            num_classes=config.num_classes,
            layer_cls=config.layer_cls,
        )

        # Create parameter nets.
        input_num_param = {"bernoulli": 1, "categorical": int(config.num_categories or 0), "normal": 2}[
            config.leaf_type
        ]
        self.input_net = InputNet(
            num_vars=self.num_vars,
            num_param=input_num_param,
            net_dim=config.net_dim,
            bias=config.bias,
            sharing=config.input_sharing,
            ff_dim=config.ff_dim,
            sigma=config.sigma,
            learn_ff=config.learn_ff,
        )

        self.inner_nets = nn.ModuleList(
            [
                InnerNet(
                    group_args=pl.group_args,
                    net_dim=config.net_dim,
                    bias=config.bias,
                    sharing=config.inner_sharing,
                    ff_dim=config.ff_dim,
                    sigma=config.sigma,
                    learn_ff=config.learn_ff,
                )
                for pl in self.partition_layers
            ]
        )

        # Mixing weights: logits over arity, per fold and per unit K.
        self._mixing_logits = nn.ParameterList(
            [nn.Parameter(torch.zeros(ml.num_folds, ml.arity, ml.num_units)) for ml in self.mixing_layers]
        )

        # Shape placeholders.
        self.in_shape = ModuleShape(features=len(self.scope.query), channels=1, repetitions=1)
        self.out_shape = ModuleShape(features=1, channels=1, repetitions=1)

    @classmethod
    def from_region_graph(
        cls,
        rg: RegionGraph,
        *,
        quadrature_rule: QuadratureRule,
        config: TensorizedQPCConfig,
    ) -> "TensorizedQPC":
        return cls(rg=rg, quadrature_rule=quadrature_rule, config=config)

    @property
    def feature_to_scope(self) -> np.ndarray:
        # Root outputs scalar over all variables.
        return np.array([self.scope])

    def _leaf_log_likelihood(self, data: Tensor, leaf_param: Tensor) -> Tensor:
        """Compute per-leaf log-likelihood for each quadrature unit.

        Args:
            data: Input data, shape (B, D_total).
            leaf_param: Parameters from InputNet, shape (V, K, P).

        Returns:
            Log-likelihoods of shape (B, V, K).
        """
        B = int(data.shape[0])
        V, K, _ = leaf_param.shape
        if V != self.num_vars or K != self.num_units:
            raise ShapeError("leaf_param shape mismatch.")

        x = data[:, self._leaf_vars.to(device=data.device)].to(dtype=leaf_param.dtype)  # (B, V)

        if self.config.leaf_type == "normal":
            loc = leaf_param[..., 0]  # (V, K)
            scale = F.softplus(leaf_param[..., 1]) + 1e-6  # (V, K)
            # Broadcast to (B,V,K)
            x_b = x.unsqueeze(-1).expand(B, V, K)
            loc_b = loc.unsqueeze(0).expand(B, V, K)
            scale_b = scale.unsqueeze(0).expand(B, V, K)
            dist = torch.distributions.Normal(loc=loc_b, scale=scale_b)
            return dist.log_prob(x_b)

        if self.config.leaf_type == "bernoulli":
            logits = leaf_param[..., 0]  # (V, K)
            x_b = x.unsqueeze(-1).expand(B, V, K)
            logits_b = logits.unsqueeze(0).expand(B, V, K)
            return -F.binary_cross_entropy_with_logits(logits_b, x_b, reduction="none")

        if self.config.leaf_type == "categorical":
            C = int(self.config.num_categories or 0)
            if leaf_param.shape[-1] != C:
                raise ShapeError("Categorical leaf_param last dim must be num_categories.")
            logits = leaf_param  # (V,K,C)
            logp = F.log_softmax(logits, dim=-1)  # (V,K,C)
            x_idx = x.to(dtype=torch.long).clamp(min=0, max=C - 1)  # (B,V)
            # Gather: expand logp to (B,V,K,C), gather on last dim.
            logp_b = logp.unsqueeze(0).expand(B, V, K, C)
            idx = x_idx.unsqueeze(-1).unsqueeze(-1).expand(B, V, K, 1)
            return torch.gather(logp_b, dim=-1, index=idx).squeeze(-1)

        raise StructureError(f"Unsupported leaf_type: {self.config.leaf_type}.")

    def log_likelihood(self, data: Tensor, cache: Cache | None = None) -> Tensor:
        if cache is None:
            cache = Cache()

        z_quad = self.quadrature_rule.points.to(device=data.device, dtype=data.dtype)
        w_quad = self.quadrature_rule.weights.to(device=data.device, dtype=data.dtype)

        # Leaf parameterization and leaf log-likelihoods.
        leaf_param = self.input_net(z_quad, n_chunks=self.config.n_chunks)  # (V,K,P)
        leaf_ll = self._leaf_log_likelihood(data, leaf_param)  # (B,V,K)

        # Initial folded outputs for leaf regions: (F0, K, B)
        x0 = leaf_ll.permute(1, 2, 0).contiguous()
        layer_outputs: list[Tensor] = [x0]
        fold_counts: list[int] = [x0.shape[0]]

        # Precompute all partition params (ten-pics style) in order.
        partition_params: list[Tensor] = []
        for net, pl in zip(self.inner_nets, self.partition_layers):
            param = net(z_quad, w_quad, n_chunks=self.config.n_chunks)
            if pl.kind == "tucker":
                if pl.num_output_units == 1 and param.ndim != 3:
                    raise ShapeError("Expected Tucker(root) param with 3 dims (F,K,K).")
                if pl.num_output_units != 1 and param.ndim != 4:
                    raise ShapeError("Expected Tucker param with 4 dims (F,K,K,K).")
                W = param.view(pl.num_folds, self.num_units, self.num_units, pl.num_output_units)
                partition_params.append(W)
            else:
                # CP: param comes as (F*H, K, O) or (F*H, K)
                W = param.view(pl.num_folds, pl.arity, self.num_units, pl.num_output_units)
                partition_params.append(W)

        for book, (kind, idx) in zip(self.bookkeeping, self.eval_plan):
            inputs_layers = [layer_outputs[i] for i in book.in_layer_ids]
            if book.should_pad:
                pad = torch.zeros(()).to(inputs_layers[0]).expand_as(inputs_layers[0][0:1])
                inputs_layers.append(pad)

            cat = torch.cat(inputs_layers, dim=0)
            inputs = cat[book.fold_indices]  # (F, H, K, B)

            if kind == "partition":
                pl = self.partition_layers[idx]
                params = partition_params[idx]
                out = _eval_partition_layer(inputs, layer=pl, params=params)
            else:
                ml = self.mixing_layers[idx]
                logits = self._mixing_logits[idx]
                weights = _masked_softmax(logits, ml.fold_mask, dim=1)
                out = _eval_mixing_layer(inputs, layer=ml, weights=weights)

            layer_outputs.append(out)
            fold_counts.append(int(out.shape[0]))

        # Final output should be (F=1, C=num_classes, B); convert to SPFlow convention.
        out = layer_outputs[-1]  # (1, C, B)
        if out.shape[0] != 1:
            raise ShapeError("Expected final folded output to have F=1.")
        out = out.squeeze(0).transpose(0, 1).contiguous()  # (B, C)
        if out.ndim != 2 or out.shape[1] != self.config.num_classes:
            raise ShapeError("Unexpected final output shape.")

        # SPFlow modules use shape (B, features=1, channels=1, repetitions=1) for scalar roots.
        if self.config.num_classes != 1:
            raise NotImplementedError("TensorizedQPC currently expects num_classes=1 for SPFlow root.")
        return out.view(out.shape[0], 1, 1, 1)

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        raise NotImplementedError("Sampling is not implemented for TensorizedQPC yet.")

    def marginalize(
        self, marg_rvs: list[int], prune: bool = True, cache: Cache | None = None
    ) -> Module | None:
        raise NotImplementedError("Marginalization is not implemented for TensorizedQPC yet.")


def _masked_softmax(logits: Tensor, fold_mask: Tensor | None, *, dim: int) -> Tensor:
    if fold_mask is None:
        return torch.softmax(logits, dim=dim)
    # fold_mask: (F,H) -> broadcast to logits shape (F,H,K)
    mask = fold_mask.to(dtype=torch.bool, device=logits.device).unsqueeze(-1)
    neg_inf = torch.full_like(logits, float("-inf"))
    masked_logits = torch.where(mask, logits, neg_inf)
    return torch.softmax(masked_logits, dim=dim)


def _eval_partition_layer(inputs: Tensor, *, layer: _PartitionLayer, params: Tensor) -> Tensor:
    """Evaluate a folded partition (sum-product) layer in log-space.

    Args:
        inputs: Tensor of shape (F, H, K, B).
        layer: Layer descriptor.
        params: Tucker: (F, K, K, O), CP: (F, H, K, O).

    Returns:
        Tensor of shape (F, O, B).
    """
    if layer.kind == "tucker":
        if layer.arity != 2:
            raise StructureError("Tucker partition layer requires arity=2.")
        return eval_tucker(inputs[:, 0], inputs[:, 1], params)

    # Collapsed CP.
    return eval_collapsed_cp(inputs, params, layer.fold_mask)


def _eval_mixing_layer(inputs: Tensor, *, layer: _MixingLayer, weights: Tensor) -> Tensor:
    """Evaluate a folded mixing sum layer in log-space.

    Args:
        inputs: Tensor of shape (F, H, K, B).
        layer: Mixing layer descriptor.
        weights: Tensor of shape (F, H, K) (already masked/normalized).

    Returns:
        Tensor of shape (F, K, B).
    """
    return eval_mixing(inputs, weights, layer.fold_mask)


def _build_folded_layers(
    *,
    rg: RegionGraph,
    leaf_regions: Sequence[Region],
    num_units: int,
    num_classes: int,
    layer_cls: Literal["auto", "tucker", "cp"],
) -> tuple[
    list[_BookEntry],
    list[_PartitionLayer],
    list[_MixingLayer],
    list[tuple[Literal["partition", "mixing"], int]],
]:
    """Build folded bookkeeping + layer descriptors from a RegionGraph."""

    # Compute region heights (leaves at 0).
    height: dict[Region, int] = {}
    for region in rg.post_order():
        if not region.children:
            height[region] = 0
        else:
            height[region] = 1 + max(height[ch] for part in region.children for ch in part)

    max_h = max(height.values()) if height else 0

    # Map leaf regions to folds (layer_id=0).
    region_id_fold: dict[Region, tuple[int, int]] = {r: (i, 0) for i, r in enumerate(leaf_regions)}
    fold_counts: list[int] = [len(leaf_regions)]

    bookkeeping: list[_BookEntry] = []
    partition_layers: list[_PartitionLayer] = []
    mixing_layers: list[_MixingLayer] = []
    eval_plan: list[tuple[Literal["partition", "mixing"], int]] = []

    layer_output_idx = 0  # 0 is leaf outputs

    def _fold_count(layer_id: int) -> int:
        return fold_counts[layer_id]

    for h in range(1, max_h + 1):
        lregions = [r for r, rh in height.items() if rh == h]
        if not lregions:
            continue

        # Collect all partitions feeding these regions.
        lpartitions: list[tuple[Region, tuple[Region, ...]]] = []
        for region in lregions:
            for part in region.children:
                lpartitions.append((region, part))

        # Partition layer fold indices (map child region -> (fold_idx, layer_id)).
        input_layer_ids: list[int] = []
        for _, part in lpartitions:
            for child in part:
                _, lid = region_id_fold[child]
                input_layer_ids.append(lid)
        unique_layer_ids = sorted(set(input_layer_ids))

        base = {}
        offset = 0
        for lid in unique_layer_ids:
            base[lid] = offset
            offset += _fold_count(lid)
        total_in_folds = offset

        max_arity = max(len(part) for _, part in lpartitions) if lpartitions else 0
        should_pad = any(len(part) < max_arity for _, part in lpartitions)

        fold_indices_rows: list[list[int]] = []
        for _, part in lpartitions:
            row: list[int] = []
            for child in part:
                fidx, lid = region_id_fold[child]
                row.append(base[lid] + fidx)
            if len(row) < max_arity:
                row.extend([total_in_folds] * (max_arity - len(row)))
            fold_indices_rows.append(row)

        fold_indices = torch.tensor(fold_indices_rows, dtype=torch.long)
        bookkeeping.append(
            _BookEntry(should_pad=should_pad, in_layer_ids=unique_layer_ids, fold_indices=fold_indices)
        )

        fold_mask = (fold_indices < total_in_folds) if should_pad else None

        # Decide partition layer kind.
        if layer_cls == "tucker" and max_arity == 2 and fold_mask is None:
            kind: Literal["tucker", "cp"] = "tucker"
        elif layer_cls == "cp":
            kind = "cp"
        else:
            # auto: use Tucker only for strict binary/no-mask, else CP.
            kind = "tucker" if (max_arity == 2 and fold_mask is None) else "cp"

        num_folds = len(lpartitions)
        num_out = num_units if h < max_h else num_classes

        group_args = _layer_to_group_args(
            kind=kind, num_folds=num_folds, arity=max_arity, num_units=num_units, num_out=num_out
        )
        partition_layers.append(
            _PartitionLayer(
                kind=kind,
                num_folds=num_folds,
                arity=max_arity,
                num_input_units=num_units,
                num_output_units=num_out,
                fold_mask=fold_mask,
                group_args=group_args,
            )
        )
        eval_plan.append(("partition", len(partition_layers) - 1))

        # Update fold_counts and region_id_fold for regions that do not need mixing.
        layer_output_idx += 1
        fold_counts.append(num_folds)

        region_mixing_indices: dict[Region, list[int]] = {}
        for i, (out_region, _) in enumerate(lpartitions):
            if len(out_region.children) == 1:
                region_id_fold[out_region] = (i, layer_output_idx)
            else:
                region_mixing_indices.setdefault(out_region, []).append(i)

        # Optional mixing layer folds for regions with >1 partition.
        non_unary_regions = [r for r in lregions if len(r.children) > 1]
        if non_unary_regions:
            max_num_inputs = max(len(r.children) for r in non_unary_regions)
            should_pad_mix = any(len(r.children) < max_num_inputs for r in non_unary_regions)

            mix_rows: list[list[int]] = []
            for region in non_unary_regions:
                idxs = list(region_mixing_indices.get(region, []))
                if len(idxs) != len(region.children):
                    raise StructureError("Inconsistent region partition indexing.")
                if len(idxs) < max_num_inputs:
                    idxs.extend([num_folds] * (max_num_inputs - len(idxs)))
                mix_rows.append(idxs)

            mix_indices = torch.tensor(mix_rows, dtype=torch.long)
            bookkeeping.append(
                _BookEntry(
                    should_pad=should_pad_mix, in_layer_ids=[layer_output_idx], fold_indices=mix_indices
                )
            )

            mix_mask = (mix_indices < num_folds) if should_pad_mix else None
            mixing_layers.append(
                _MixingLayer(
                    num_folds=len(non_unary_regions),
                    arity=max_num_inputs,
                    num_units=num_out,
                    fold_mask=mix_mask,
                )
            )
            eval_plan.append(("mixing", len(mixing_layers) - 1))

            layer_output_idx += 1
            fold_counts.append(len(non_unary_regions))

            for i, region in enumerate(non_unary_regions):
                region_id_fold[region] = (i, layer_output_idx)

    return bookkeeping, partition_layers, mixing_layers, eval_plan


def _layer_to_group_args(
    *,
    kind: Literal["tucker", "cp"],
    num_folds: int,
    arity: int,
    num_units: int,
    num_out: int,
) -> IntegralGroupArgs:
    """Match ten-pics `pc2integral_group_args` logic for (num_dim, perm_dim, norm_dim)."""
    if kind == "cp":
        num_dim = 1 if num_out == 1 else 2
        num_funcs = num_folds * arity
        if num_dim == 1:
            return IntegralGroupArgs(num_dim=num_dim, num_funcs=num_funcs, perm_dim=(1,), norm_dim=(1,))
        return IntegralGroupArgs(num_dim=num_dim, num_funcs=num_funcs, perm_dim=(2, 1), norm_dim=(1,))

    # Tucker.
    num_dim = 2 if num_out == 1 else 3
    num_funcs = num_folds
    if num_dim == 2:
        return IntegralGroupArgs(num_dim=num_dim, num_funcs=num_funcs, perm_dim=(1, 2), norm_dim=(1, 2))
    return IntegralGroupArgs(num_dim=num_dim, num_funcs=num_funcs, perm_dim=(3, 2, 1), norm_dim=(1, 2))
