"""Tests for folded tensorized QPC materialization."""

import numpy as np
import torch
from torch import Tensor

from spflow.meta.data.scope import Scope
from spflow.meta.region_graph import Region, RegionGraph
from spflow.modules.module import Module
from spflow.zoo.pic import QuadratureRule, pic2qpc
from spflow.zoo.pic.tensorized.qpc import InnerNet, IntegralGroupArgs, TensorizedQPC, TensorizedQPCConfig


class DummyPIC(Module):
    """Minimal symbolic PIC wrapper for tensorized materialization tests."""

    def __init__(self, rg: RegionGraph) -> None:
        super().__init__()
        self.scope = rg.root.scope
        setattr(self, "_region_graph", rg)

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([self.scope])

    def log_likelihood(self, data: Tensor, cache=None) -> Tensor:  # pragma: no cover
        raise NotImplementedError

    def sample(
        self, num_samples=None, data=None, is_mpe=False, cache=None, sampling_ctx=None
    ) -> Tensor:  # pragma: no cover
        raise NotImplementedError

    def _sample(self, data: Tensor, sampling_ctx, cache, is_mpe: bool = False) -> Tensor:  # pragma: no cover
        raise NotImplementedError

    def marginalize(self, marg_rvs, prune: bool = True, cache=None):  # pragma: no cover
        raise NotImplementedError


def test_pic2qpc_tensorized_runs_on_nary_partition():
    # Root region with a single 4-ary partition.
    leaves = [Region(Scope([i])) for i in range(4)]
    root = Region(Scope([0, 1, 2, 3]))
    root.add_partition(tuple(leaves))
    rg = RegionGraph(root)

    pic = DummyPIC(rg)

    K = 3
    rule = QuadratureRule(points=torch.linspace(-1, 1, K), weights=torch.ones(K) * (2.0 / K))
    cfg = TensorizedQPCConfig(leaf_type="normal", layer_cls="cp", n_chunks=2)

    qpc = pic2qpc(pic, rule, mode="tensorized", tensorized_config=cfg)
    assert isinstance(qpc, Module)

    data = torch.randn(5, 4)
    ll = qpc.log_likelihood(data)
    assert ll.shape == (5, 1, 1, 1)
    assert torch.isfinite(ll).all()


def test_tensorized_qpc_builds_fold_mask_when_arities_vary():
    # Build an RG where the root has two partitions with different arity, forcing padding.
    r0 = Region(Scope([0]))
    r1 = Region(Scope([1]))
    r2 = Region(Scope([2]))
    r3 = Region(Scope([3]))

    r01 = Region(Scope([0, 1]))
    r01.add_partition((r0, r1))  # binary

    root = Region(Scope([0, 1, 2, 3]))
    root.add_partition((r0, r1, r2, r3))  # arity 4
    root.add_partition((r01, r2, r3))  # arity 3

    rg = RegionGraph(root)

    K = 3
    rule = QuadratureRule(points=torch.linspace(-1, 1, K), weights=torch.ones(K) * (2.0 / K))
    cfg = TensorizedQPCConfig(leaf_type="normal", layer_cls="cp", n_chunks=1)

    qpc = TensorizedQPC.from_region_graph(rg, quadrature_rule=rule, config=cfg)
    assert any(pl.fold_mask is not None for pl in qpc.partition_layers)

    data = torch.randn(2, 4)
    ll = qpc.log_likelihood(data)
    assert ll.shape == (2, 1, 1, 1)
    assert torch.isfinite(ll).all()


def test_innernet_normalizes_over_norm_dim():
    K = 5
    z = torch.linspace(-1, 1, K)
    w = torch.ones(K) * (2.0 / K)

    args = IntegralGroupArgs(num_dim=2, num_funcs=3, perm_dim=(1, 2), norm_dim=(1, 2))
    net = InnerNet(group_args=args, net_dim=8, sharing="f")

    param = net(z, w, n_chunks=2)
    # param has shape (num_funcs, K, K) (after permute including func dim).
    assert param.shape[0] == 3
    # Sum over norm dims should be 1 for each function.
    summed = param.sum(dim=(1, 2))
    assert torch.allclose(summed, torch.ones_like(summed), atol=1e-4)
