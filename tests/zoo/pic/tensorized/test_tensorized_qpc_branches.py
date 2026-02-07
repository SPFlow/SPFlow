"""Additional branch-coverage tests for tensorized QPC internals."""

import pytest
import torch
from torch import nn

from spflow.exceptions import InvalidParameterError, ShapeError, StructureError
from spflow.meta.data.scope import Scope
from spflow.meta.region_graph import Region, RegionGraph
from spflow.zoo.pic import QuadratureRule
from spflow.zoo.pic.tensorized.qpc import (
    FourierLayer,
    InnerNet,
    InputNet,
    IntegralGroupArgs,
    TensorizedQPC,
    TensorizedQPCConfig,
    _PartitionLayer,
    _eval_partition_layer,
    _layer_to_group_args,
    _masked_softmax,
)


def _rule(k: int = 3) -> QuadratureRule:
    return QuadratureRule(points=torch.linspace(-1, 1, k), weights=torch.ones(k) * (2.0 / k))


def _rg_binary() -> RegionGraph:
    r0 = Region(Scope([0]))
    r1 = Region(Scope([1]))
    root = Region(Scope([0, 1]))
    root.add_partition((r0, r1))
    return RegionGraph(root)


def _rg_single_leaf() -> RegionGraph:
    return RegionGraph(Region(Scope([0])))


def test_fourier_layer_validates_even_output_features():
    with pytest.raises(InvalidParameterError):
        FourierLayer(in_features=1, out_features=3)


def test_fourier_layer_learnable_coeff_is_parameter():
    layer = FourierLayer(in_features=2, out_features=4, learnable=True)
    assert isinstance(layer.coeff, nn.Parameter)


def test_inputnet_validation_and_f_sharing_expand():
    with pytest.raises(InvalidParameterError):
        InputNet(num_vars=2, num_param=2, sharing="x")  # type: ignore[arg-type]

    net = InputNet(num_vars=3, num_param=2, sharing="f", net_dim=8, ff_dim=8)

    with pytest.raises(ShapeError):
        net(torch.ones(2, 2))

    with pytest.raises(InvalidParameterError):
        net(torch.linspace(-1, 1, 4), n_chunks=0)

    out = net(torch.linspace(-1, 1, 4), n_chunks=2)
    assert out.shape == (3, 4, 2)


def test_inputnet_composite_sharing_bias_and_shape_guard(monkeypatch: pytest.MonkeyPatch):
    net = InputNet(num_vars=3, num_param=2, sharing="c", net_dim=8, ff_dim=8, bias=True)
    assert net.net[-1].bias is not None
    assert net.net[-1].bias.shape[0] == 6

    class _BadSequential(nn.Sequential):
        def __init__(self, num_param: int) -> None:
            super().__init__(nn.Identity(), nn.Conv1d(1, 1, 1), nn.Conv1d(1, 1, 1))
            self.num_param = num_param

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return torch.zeros((self.num_param, z.shape[0]), dtype=z.dtype, device=z.device)

    broken = InputNet(num_vars=2, num_param=2, sharing="none", net_dim=8, ff_dim=8)
    monkeypatch.setattr(broken, "net", _BadSequential(num_param=2))
    with pytest.raises(ShapeError):
        broken(torch.linspace(-1, 1, 3))


def test_innernet_validation_paths():
    with pytest.raises(InvalidParameterError):
        InnerNet(group_args=IntegralGroupArgs(2, 1, (1, 2), (1, 2)), sharing="bad")  # type: ignore[arg-type]

    with pytest.raises(InvalidParameterError):
        InnerNet(group_args=IntegralGroupArgs(2, 1, (1, 1), (1, 2)))

    with pytest.raises(InvalidParameterError):
        InnerNet(group_args=IntegralGroupArgs(2, 1, (1, 2), (0,)))

    net = InnerNet(group_args=IntegralGroupArgs(2, 2, (1, 2), (1, 2)), sharing="c", net_dim=8, ff_dim=8)
    z = torch.linspace(-1, 1, 3)
    w = torch.ones(3) * (2.0 / 3.0)
    with pytest.raises(ShapeError):
        net(z.view(1, -1), w)
    with pytest.raises(InvalidParameterError):
        net(z, w, n_chunks=0)
    out = net(z, w, n_chunks=2)
    assert out.shape == (2, 3, 3)


def test_innernet_composite_sharing_bias_repetition():
    net = InnerNet(
        group_args=IntegralGroupArgs(2, 2, (1, 2), (1, 2)),
        sharing="c",
        net_dim=8,
        ff_dim=8,
        bias=True,
    )
    assert net.net[-2].bias is not None
    assert net.net[-2].bias.shape[0] == 2


def test_masked_softmax_with_and_without_mask():
    logits = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
    no_mask = _masked_softmax(logits, None, dim=1)
    assert torch.allclose(no_mask.sum(dim=1), torch.ones_like(no_mask.sum(dim=1)))

    mask = torch.tensor([[True, False]])
    with_mask = _masked_softmax(logits, mask, dim=1)
    assert torch.allclose(with_mask[:, 0, :], torch.ones_like(with_mask[:, 0, :]))
    assert torch.allclose(with_mask[:, 1, :], torch.zeros_like(with_mask[:, 1, :]))


def test_eval_partition_layer_tucker_rejects_non_binary_arity():
    layer = _PartitionLayer(
        kind="tucker",
        num_folds=1,
        arity=3,
        num_input_units=2,
        num_output_units=2,
        fold_mask=None,
        group_args=IntegralGroupArgs(3, 1, (3, 2, 1), (1, 2)),
    )
    inputs = torch.zeros(1, 3, 2, 1)
    params = torch.zeros(1, 2, 2, 2)
    with pytest.raises(StructureError):
        _eval_partition_layer(inputs, layer=layer, params=params)


@pytest.mark.parametrize(
    ("kind", "num_out", "expected"),
    [
        ("cp", 1, IntegralGroupArgs(1, 6, (1,), (1,))),
        ("cp", 4, IntegralGroupArgs(2, 6, (2, 1), (1,))),
        ("tucker", 1, IntegralGroupArgs(2, 3, (1, 2), (1, 2))),
        ("tucker", 5, IntegralGroupArgs(3, 3, (3, 2, 1), (1, 2))),
    ],
)
def test_layer_to_group_args_variants(kind: str, num_out: int, expected: IntegralGroupArgs):
    got = _layer_to_group_args(kind=kind, num_folds=3, arity=2, num_units=4, num_out=num_out)
    assert got == expected


def test_tensorized_qpc_constructor_validation_errors():
    rg = _rg_binary()
    cfg = TensorizedQPCConfig(leaf_type="normal")

    bad_points = QuadratureRule(points=torch.ones(2, 2), weights=torch.ones(2))
    with pytest.raises(ShapeError):
        TensorizedQPC.from_region_graph(rg, quadrature_rule=bad_points, config=cfg)

    bad_len = QuadratureRule(points=torch.ones(3), weights=torch.ones(2))
    with pytest.raises(ShapeError):
        TensorizedQPC.from_region_graph(rg, quadrature_rule=bad_len, config=cfg)

    with pytest.raises(InvalidParameterError):
        TensorizedQPC.from_region_graph(
            rg, quadrature_rule=_rule(), config=TensorizedQPCConfig(leaf_type="normal", n_chunks=0)
        )

    with pytest.raises(InvalidParameterError):
        TensorizedQPC.from_region_graph(
            rg, quadrature_rule=_rule(), config=TensorizedQPCConfig(leaf_type="categorical", num_categories=1)
        )

    with pytest.raises(InvalidParameterError):
        TensorizedQPC.from_region_graph(
            rg, quadrature_rule=_rule(), config=TensorizedQPCConfig(leaf_type="normal", num_classes=0)
        )


def test_tensorized_qpc_rejects_non_univariate_leaf_regions():
    a = Region(Scope([0, 1]))
    b = Region(Scope([2]))
    root_bad_leaf = Region(Scope([0, 1, 2]))
    root_bad_leaf.add_partition((a, b))
    with pytest.raises(StructureError):
        TensorizedQPC.from_region_graph(
            RegionGraph(root_bad_leaf),
            quadrature_rule=_rule(),
            config=TensorizedQPCConfig(leaf_type="normal"),
        )


def test_tensorized_qpc_feature_to_scope_property():
    qpc = TensorizedQPC.from_region_graph(
        _rg_single_leaf(), quadrature_rule=_rule(), config=TensorizedQPCConfig(leaf_type="normal")
    )
    scopes = qpc.feature_to_scope
    assert scopes.shape == (1,)
    assert scopes[0] == qpc.scope


def test_leaf_log_likelihood_shape_check_and_categorical_param_check():
    qpc_cat = TensorizedQPC.from_region_graph(
        _rg_single_leaf(),
        quadrature_rule=_rule(),
        config=TensorizedQPCConfig(leaf_type="categorical", num_categories=3),
    )

    data = torch.tensor([[1.0], [2.0]])
    with pytest.raises(ShapeError):
        qpc_cat._leaf_log_likelihood(
            data, torch.zeros(2, qpc_cat.num_units, 3)
        )  # pylint: disable=protected-access

    with pytest.raises(ShapeError):
        qpc_cat._leaf_log_likelihood(  # pylint: disable=protected-access
            data, torch.zeros(qpc_cat.num_vars, qpc_cat.num_units, 2)
        )


def test_leaf_log_likelihood_rejects_unsupported_leaf_type():
    qpc = TensorizedQPC.from_region_graph(
        _rg_single_leaf(), quadrature_rule=_rule(), config=TensorizedQPCConfig(leaf_type="normal")
    )
    object.__setattr__(qpc.config, "leaf_type", "weird")
    with pytest.raises(StructureError):
        qpc._leaf_log_likelihood(torch.zeros((2, 1)), torch.zeros((qpc.num_vars, qpc.num_units, 2)))


def test_log_likelihood_tucker_param_dimension_guards():
    class _BadInnerNet(nn.Module):
        def __init__(self, out: torch.Tensor) -> None:
            super().__init__()
            self._out = out

        def forward(self, z_quad: torch.Tensor, w_quad: torch.Tensor, *, n_chunks: int = 1) -> torch.Tensor:
            return self._out.to(device=z_quad.device, dtype=z_quad.dtype)

    qpc_root = TensorizedQPC.from_region_graph(
        _rg_binary(),
        quadrature_rule=_rule(),
        config=TensorizedQPCConfig(leaf_type="normal", layer_cls="tucker"),
    )
    qpc_root.inner_nets = nn.ModuleList([_BadInnerNet(torch.zeros(1, 3, 3, 1))])  # ndim=4, root expects 3
    with pytest.raises(ShapeError):
        qpc_root.log_likelihood(torch.randn(2, 2))

    r0 = Region(Scope([0]))
    r1 = Region(Scope([1]))
    r2 = Region(Scope([2]))
    r3 = Region(Scope([3]))
    r01 = Region(Scope([0, 1]))
    r23 = Region(Scope([2, 3]))
    r01.add_partition((r0, r1))
    r23.add_partition((r2, r3))
    root = Region(Scope([0, 1, 2, 3]))
    root.add_partition((r01, r23))
    qpc_mid = TensorizedQPC.from_region_graph(
        RegionGraph(root),
        quadrature_rule=_rule(),
        config=TensorizedQPCConfig(leaf_type="normal", layer_cls="tucker"),
    )
    qpc_mid.inner_nets = nn.ModuleList(
        [
            _BadInnerNet(torch.zeros(2, 3, 3)),  # ndim=3, non-root tucker expects 4
            _BadInnerNet(torch.zeros(1, 3, 3)),
        ]
    )
    with pytest.raises(ShapeError):
        qpc_mid.log_likelihood(torch.randn(2, 4))


def test_log_likelihood_final_output_shape_guards():
    qpc = TensorizedQPC.from_region_graph(
        _rg_binary(), quadrature_rule=_rule(), config=TensorizedQPCConfig(leaf_type="normal")
    )
    qpc.bookkeeping = []
    qpc.eval_plan = []
    with pytest.raises(ShapeError):
        qpc.log_likelihood(torch.randn(2, 2))

    qpc_single = TensorizedQPC.from_region_graph(
        _rg_single_leaf(), quadrature_rule=_rule(), config=TensorizedQPCConfig(leaf_type="normal")
    )
    with pytest.raises(ShapeError):
        qpc_single.log_likelihood(torch.randn(2, 1))


def test_layer_cls_tucker_forces_tucker_when_binary_unpadded():
    qpc = TensorizedQPC.from_region_graph(
        _rg_binary(),
        quadrature_rule=_rule(),
        config=TensorizedQPCConfig(leaf_type="normal", layer_cls="tucker"),
    )
    assert qpc.partition_layers
    assert all(pl.kind == "tucker" for pl in qpc.partition_layers)


def test_log_likelihood_num_classes_not_supported_for_spflow_root():
    qpc = TensorizedQPC.from_region_graph(
        _rg_single_leaf(),
        quadrature_rule=_rule(k=2),
        config=TensorizedQPCConfig(leaf_type="normal", num_classes=2),
    )
    with pytest.raises(NotImplementedError):
        qpc.log_likelihood(torch.randn(3, 1))


def test_sample_and_marginalize_not_implemented():
    qpc = TensorizedQPC.from_region_graph(
        _rg_binary(), quadrature_rule=_rule(), config=TensorizedQPCConfig(leaf_type="normal")
    )
    with pytest.raises(NotImplementedError):
        qpc.sample(num_samples=2)
    with pytest.raises(NotImplementedError):
        qpc.marginalize([0])
