"""Equivalence tests for tensorized vs expanded (unfolded) QPC evaluation.

These tests build a `TensorizedQPC` via `pic2qpc(..., mode="tensorized")`, then
construct an *unfolded* SPFlow module graph with the *same* learned parameters
(leaf params, partition params, mixing weights). The two evaluations must match.

This specifically targets correctness of:
- fold masking / padding semantics
- n-ary partitions
- mixing layers (multi-partition regions)
- leaf families (normal/bernoulli/categorical)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from spflow.meta.data.scope import Scope
from spflow.meta.region_graph import Region, RegionGraph
from spflow.modules.leaves.bernoulli import Bernoulli
from spflow.modules.leaves.categorical import Categorical
from spflow.modules.leaves.normal import Normal
from spflow.modules.module import Module
from spflow.modules.ops.cat import Cat
from spflow.modules.products.elementwise_product import ElementwiseProduct
from spflow.modules.products.outer_product import OuterProduct
from spflow.exp.pic import QuadratureRule, pic2qpc, rg2pic
from spflow.exp.pic.tensorized.qpc import TensorizedQPC, TensorizedQPCConfig, _masked_softmax
from spflow.exp.pic.weighted_sum import WeightedSum


class _DummyLeaf(Module):
    """Symbolic RG leaf used only to build a PIC shell for tensorized mode."""

    def __init__(self, x_scope: Scope, z_scope: Scope) -> None:
        super().__init__()
        self.scope = x_scope
        self.latent_scope = z_scope
        self.in_shape = None
        self.out_shape = None

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([self.scope])

    def log_likelihood(self, data: Tensor, cache=None) -> Tensor:  # pragma: no cover
        raise NotImplementedError

    def sample(
        self, num_samples=None, data=None, is_mpe=False, cache=None, sampling_ctx=None
    ) -> Tensor:  # pragma: no cover
        raise NotImplementedError

    def marginalize(self, marg_rvs, prune: bool = True, cache=None):  # pragma: no cover
        raise NotImplementedError


def _leaf_modules_from_params(
    *,
    leaf_type: str,
    num_categories: int | None,
    leaf_regions: list[Region],
    leaf_param: Tensor,  # (V,K,P) for normal/bernoulli, (V,K,C) for categorical
) -> list[Module]:
    V = len(leaf_regions)
    if leaf_param.shape[0] != V:
        raise ValueError("leaf_param first dim must match number of variables.")

    K = int(leaf_param.shape[1])

    out: list[Module] = []
    for v, region in enumerate(leaf_regions):
        scope = region.scope
        if leaf_type == "normal":
            loc = leaf_param[v, :, 0].view(1, K, 1)
            scale = (F.softplus(leaf_param[v, :, 1]) + 1e-6).view(1, K, 1)
            out.append(Normal(scope=scope, out_channels=K, num_repetitions=1, loc=loc, scale=scale))
        elif leaf_type == "bernoulli":
            logits = leaf_param[v, :, 0].view(1, K, 1)
            out.append(Bernoulli(scope=scope, out_channels=K, num_repetitions=1, logits=logits))
        elif leaf_type == "categorical":
            C = int(num_categories or 0)
            logits = leaf_param[v].view(1, K, 1, C)
            out.append(Categorical(scope=scope, out_channels=K, num_repetitions=1, K=C, logits=logits))
        else:
            raise ValueError(f"Unknown leaf_type: {leaf_type}")
    return out


def _partition_module(
    *,
    kind: str,
    params_fold: Tensor,
    children: list[Module],
) -> Module:
    # All children must have disjoint scopes.
    if kind == "tucker":
        assert len(children) == 2
        left, right = children
        op = OuterProduct(inputs=[left, right])
        K = left.out_shape.channels
        O = int(params_fold.shape[-1])
        W = params_fold.reshape(K * K, O)
        weights = W.view(1, K * K, O, 1)
        return WeightedSum(inputs=op, weights=weights)

    if kind == "cp":
        H = int(params_fold.shape[0])
        h_len = len(children)
        assert h_len <= H
        weighted_children: list[Module] = []
        for h in range(h_len):
            child = children[h]
            W = params_fold[h]  # (K,O)
            weights = W.view(1, W.shape[0], W.shape[1], 1)
            weighted_children.append(WeightedSum(inputs=child, weights=weights))
        # Product across children in log-space = sum of log-likelihoods.
        if len(weighted_children) == 2:
            return ElementwiseProduct(inputs=weighted_children)
        cur = weighted_children[0]
        for nxt in weighted_children[1:]:
            cur = ElementwiseProduct(inputs=[cur, nxt])
        return cur

    raise ValueError(f"Unknown partition kind: {kind}")


def _mixing_module(
    *,
    parts: list[Module],
    weights: Tensor,  # (N,O)
) -> Module:
    if len(parts) == 1:
        return parts[0]
    out_ch = int(parts[0].out_shape.channels)
    if any(p.out_shape.channels != out_ch for p in parts):
        raise ValueError("All partition outputs must have same channel count for mixing.")
    if weights.shape != (len(parts), out_ch):
        raise ValueError("weights must have shape (num_parts, out_channels).")

    cat = Cat(inputs=parts, dim=2)
    N = len(parts)
    in_ch = N * out_ch

    W = torch.zeros((in_ch, out_ch), dtype=weights.dtype, device=weights.device)
    for i in range(N):
        w_i = weights[i]  # (out_ch,)
        W[i * out_ch : (i + 1) * out_ch, :] = torch.diag(w_i)

    return WeightedSum(inputs=cat, weights=W.view(1, in_ch, out_ch, 1))


def _expanded_from_tensorized(qpc: TensorizedQPC) -> Module:
    rg = qpc.rg
    rule = qpc.quadrature_rule

    z_quad = rule.points
    w_quad = rule.weights

    # Leaf parameters and leaf modules.
    leaf_param = qpc.input_net(z_quad, n_chunks=qpc.config.n_chunks)  # (V,K,P) or (V,K,C)
    leaf_modules = _leaf_modules_from_params(
        leaf_type=qpc.config.leaf_type,
        num_categories=qpc.config.num_categories,
        leaf_regions=qpc._leaf_regions,  # pylint: disable=protected-access
        leaf_param=leaf_param,
    )

    region_out: dict[Region, Module] = {r: m for r, m in zip(qpc._leaf_regions, leaf_modules)}  # pylint: disable=protected-access

    # Height dict insertion order should match TensorizedQPC builder.
    height: dict[Region, int] = {}
    for region in rg.post_order():
        if not region.children:
            height[region] = 0
        else:
            height[region] = 1 + max(height[ch] for part in region.children for ch in part)
    max_h = max(height.values()) if height else 0

    partition_layer_idx = 0
    mixing_layer_idx = 0

    for h in range(1, max_h + 1):
        lregions = [r for r, rh in height.items() if rh == h]
        if not lregions:
            continue

        lpartitions: list[tuple[Region, tuple[Region, ...]]] = []
        for region in lregions:
            for part in region.children:
                lpartitions.append((region, part))

        pl = qpc.partition_layers[partition_layer_idx]
        net = qpc.inner_nets[partition_layer_idx]
        raw = net(z_quad, w_quad, n_chunks=qpc.config.n_chunks)
        if pl.kind == "tucker":
            params = raw.view(pl.num_folds, qpc.num_units, qpc.num_units, pl.num_output_units)
        else:
            params = raw.view(pl.num_folds, pl.arity, qpc.num_units, pl.num_output_units)

        partition_outs: dict[Region, list[Module]] = {r: [] for r in lregions}
        for fold_idx, (region, part) in enumerate(lpartitions):
            children = [region_out[ch] for ch in part]
            part_mod = _partition_module(kind=pl.kind, params_fold=params[fold_idx], children=children)
            partition_outs[region].append(part_mod)

        partition_layer_idx += 1

        non_unary_regions = [r for r in lregions if len(r.children) > 1]
        if non_unary_regions:
            ml = qpc.mixing_layers[mixing_layer_idx]
            logits = qpc._mixing_logits[mixing_layer_idx]  # pylint: disable=protected-access
            mix_w = _masked_softmax(logits, ml.fold_mask, dim=1)  # (F, H, O)

            for ridx, region in enumerate(non_unary_regions):
                parts = partition_outs[region]
                w = mix_w[ridx, : len(parts), :]  # (N,O)
                region_out[region] = _mixing_module(parts=parts, weights=w)
            mixing_layer_idx += 1

        for region in lregions:
            if region in non_unary_regions:
                continue
            # Unary region: just pass through the only partition output.
            region_out[region] = partition_outs[region][0]

    return region_out[rg.root]


def _build_rgs() -> list[RegionGraph]:
    # RG1: pure binary tree (should trigger Tucker in auto mode).
    r0 = Region(Scope([0]))
    r1 = Region(Scope([1]))
    root01 = Region(Scope([0, 1]))
    root01.add_partition((r0, r1))

    # RG2: pure 4-ary partition (forces CP in auto mode).
    a0 = Region(Scope([0]))
    a1 = Region(Scope([1]))
    a2 = Region(Scope([2]))
    a3 = Region(Scope([3]))
    root4 = Region(Scope([0, 1, 2, 3]))
    root4.add_partition((a0, a1, a2, a3))

    # RG3: mixing at root with varying arity (forces fold_mask + mixing).
    b0 = Region(Scope([0]))
    b1 = Region(Scope([1]))
    b2 = Region(Scope([2]))
    b3 = Region(Scope([3]))
    b01 = Region(Scope([0, 1]))
    b01.add_partition((b0, b1))
    root_mix = Region(Scope([0, 1, 2, 3]))
    root_mix.add_partition((b0, b1, b2, b3))  # arity 4
    root_mix.add_partition((b01, b2, b3))  # arity 3

    return [RegionGraph(root01), RegionGraph(root4), RegionGraph(root_mix)]


@pytest.mark.parametrize("K", [3, 4])
@pytest.mark.parametrize("leaf_type", ["normal", "bernoulli", "categorical"])
def test_tensorized_matches_unfolded_expanded_graph(K: int, leaf_type: str):
    torch.manual_seed(0)

    rule = QuadratureRule(points=torch.linspace(-1, 1, K), weights=torch.ones(K) * (2.0 / K))

    num_categories = 4 if leaf_type == "categorical" else None
    cfg = TensorizedQPCConfig(
        leaf_type=leaf_type, num_categories=num_categories, layer_cls="auto", n_chunks=2
    )

    for rg in _build_rgs():
        # Build a PIC shell to exercise the public `pic2qpc` API.
        pic = rg2pic(rg, leaf_factory=lambda x, z: _DummyLeaf(x, z))
        tqpc = pic2qpc(pic, rule, mode="tensorized", tensorized_config=cfg)
        assert isinstance(tqpc, TensorizedQPC)

        expanded = _expanded_from_tensorized(tqpc)

        D = int(max(rg.root.scope.query) + 1) if len(rg.root.scope.query) else 0
        B = 8

        if leaf_type == "normal":
            data = torch.randn(B, D)
        elif leaf_type == "bernoulli":
            data = torch.randint(0, 2, (B, D)).float()
        else:
            assert num_categories is not None
            data = torch.randint(0, num_categories, (B, D)).float()

        ll_t = tqpc.log_likelihood(data).squeeze()
        ll_e = expanded.log_likelihood(data).squeeze()

        assert torch.allclose(ll_t, ll_e, atol=1e-5, rtol=1e-5)
