"""Tests for PIC pipeline functions (rg2pic, pic2qpc)."""

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterError, ShapeError, StructureError
from spflow.meta.data.scope import Scope
from spflow.meta.region_graph import Region, RegionGraph
from spflow.modules.module import Module
from spflow.modules.leaves.normal import Normal
from spflow.modules.products.outer_product import OuterProduct
from spflow.modules.products.elementwise_product import ElementwiseProduct
from spflow.zoo.pic import (
    Integral,
    MergeStrategy,
    PICProduct,
    PICSum,
    QuadratureRule,
    WeightedSum,
    pic2qpc,
    rg2pic,
)
from spflow.zoo.pic.functional_sharing import FunctionGroup
from spflow.zoo.pic.pipeline import _maybe_attach_function_group, _merge_units
from spflow.zoo.pic.tensorized.qpc import TensorizedQPCConfig


class ConstantOneFunction(nn.Module):
    """Return 1 for any broadcasted z/y inputs."""

    def forward(self, z: Tensor, y: Tensor) -> Tensor:
        # z, y are expected to have the same leading shape, e.g. (out_ch, in_ch, dim)
        if z.shape[:-1] != y.shape[:-1]:
            raise ValueError("z and y leading shapes must match.")
        return torch.ones(z.shape[:-1], device=z.device, dtype=z.dtype)


class NormalPICInput(Module):
    """PIC input unit that materializes to Normal(X | Z=z_k) with mean=z_k, scale=1."""

    def __init__(self, x_scope: Scope, z_scope: Scope) -> None:
        super().__init__()
        self.scope = x_scope
        self.latent_scope = z_scope
        self.in_shape = None  # not used for symbolic PIC nodes
        self.out_shape = None  # not used for symbolic PIC nodes

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([self.scope])

    def materialize(self, quadrature_rule: QuadratureRule) -> Module:
        points = quadrature_rule.points
        K = int(points.shape[0])
        if len(self.latent_scope.query) == 0:
            loc = torch.zeros((1, 1, 1), dtype=points.dtype, device=points.device)
            scale = torch.ones((1, 1, 1), dtype=points.dtype, device=points.device)
            return Normal(scope=self.scope, out_channels=1, loc=loc, scale=scale)
        if len(self.latent_scope.query) != 1:
            raise NotImplementedError("This test input supports only |Z_u| ∈ {0,1}.")

        loc = points.view(1, K, 1).to(dtype=points.dtype, device=points.device)
        scale = torch.ones((1, K, 1), dtype=points.dtype, device=points.device)
        return Normal(scope=self.scope, out_channels=K, num_repetitions=1, loc=loc, scale=scale)

    def log_likelihood(self, data: Tensor, cache=None) -> Tensor:  # pragma: no cover
        raise NotImplementedError("NormalPICInput is symbolic; materialize to QPC with pic2qpc().")

    def sample(
        self, num_samples=None, data=None, is_mpe=False, cache=None
    ) -> Tensor:  # pragma: no cover
        raise NotImplementedError

    def _sample(self, data: Tensor, sampling_ctx, cache, is_mpe: bool = False) -> Tensor:  # pragma: no cover
        raise NotImplementedError

    def marginalize(self, marg_rvs, prune: bool = True, cache=None):  # pragma: no cover
        raise NotImplementedError


class ConstantPICInput(Module):
    """PIC input unit that materializes to a constant log-likelihood vector over Z channels."""

    def __init__(self, x_scope: Scope, z_scope: Scope, values: Tensor) -> None:
        super().__init__()
        self.scope = x_scope
        self.latent_scope = z_scope
        self._values = values
        self.in_shape = None
        self.out_shape = None

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([self.scope])

    def materialize(self, quadrature_rule: QuadratureRule) -> Module:
        K = int(quadrature_rule.points.shape[0])
        if self._values.shape != (K,):
            raise ValueError("values must match quadrature K.")
        return ConstantQPCModule(scope=self.scope, log_values=self._values)

    def log_likelihood(self, data: Tensor, cache=None) -> Tensor:  # pragma: no cover
        raise NotImplementedError("ConstantPICInput is symbolic; materialize to QPC with pic2qpc().")

    def sample(
        self, num_samples=None, data=None, is_mpe=False, cache=None
    ) -> Tensor:  # pragma: no cover
        raise NotImplementedError

    def _sample(self, data: Tensor, sampling_ctx, cache, is_mpe: bool = False) -> Tensor:  # pragma: no cover
        raise NotImplementedError

    def marginalize(self, marg_rvs, prune: bool = True, cache=None):  # pragma: no cover
        raise NotImplementedError


class ConstantQPCModule(Module):
    """QPC module that returns fixed log-likelihood values over channels."""

    def __init__(self, scope: Scope, log_values: Tensor) -> None:
        super().__init__()
        self.scope = scope
        self.log_values = log_values
        from spflow.modules.module_shape import ModuleShape

        self.in_shape = ModuleShape(features=1, channels=1, repetitions=1)
        self.out_shape = ModuleShape(features=1, channels=int(log_values.shape[0]), repetitions=1)

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([self.scope])

    def log_likelihood(self, data: Tensor, cache=None) -> Tensor:
        B = data.shape[0]
        ll = self.log_values.to(device=data.device, dtype=data.dtype)
        return ll.view(1, 1, -1, 1).expand(B, 1, -1, 1)

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None) -> Tensor:
        raise NotImplementedError

    def _sample(self, data: Tensor, sampling_ctx, cache, is_mpe: bool = False) -> Tensor:
        raise NotImplementedError

    def marginalize(self, marg_rvs, prune: bool = True, cache=None):
        if any(rv in self.scope.query for rv in marg_rvs):
            return None
        return self


def test_rg2pic_single_region_root_leaf_has_empty_latent():
    r0 = Region(Scope([0]))
    rg = RegionGraph(r0)

    pic = rg2pic(rg, leaf_factory=lambda x, z: NormalPICInput(x, z))
    assert isinstance(pic, NormalPICInput)
    assert pic.latent_scope.empty()

    rule = QuadratureRule(points=torch.linspace(-1, 1, 5), weights=torch.ones(5) / 5)
    qpc = pic2qpc(pic, rule)
    assert isinstance(qpc, Normal)
    assert qpc.out_shape.channels == 1


def test_tucker_root_materializes_to_scalar_and_matches_explicit_quadrature():
    # RG: {0,1} -> {0} | {1}
    r0 = Region(Scope([0]))
    r1 = Region(Scope([1]))
    root = Region(Scope([0, 1]))
    root.add_partition((r0, r1))
    rg = RegionGraph(root)

    pic = rg2pic(
        rg,
        merge_strategy=MergeStrategy.AUTO,
        leaf_factory=lambda x, z: NormalPICInput(x, z),
        function_factory=lambda z_dim, y_dim: ConstantOneFunction(),
    )

    assert isinstance(pic, Integral)
    assert pic.latent_scope.empty()
    assert isinstance(pic.inputs, PICProduct)

    K = 5
    points = torch.linspace(-1, 1, K)
    weights = torch.ones(K) / K
    rule = QuadratureRule(points=points, weights=weights)

    qpc = pic2qpc(pic, rule)
    assert isinstance(qpc, WeightedSum)
    assert qpc.out_shape.channels == 1
    assert isinstance(qpc.inputs, OuterProduct)

    # Numerical correctness: compare to explicit tensor-product quadrature.
    data = torch.tensor([[0.25, -0.5], [1.0, 0.0]])
    ll = qpc.log_likelihood(data).squeeze()  # (B,)

    dist = torch.distributions.Normal(loc=points, scale=torch.ones_like(points))
    ll0 = dist.log_prob(data[:, 0].unsqueeze(-1))  # (B, K)
    ll1 = dist.log_prob(data[:, 1].unsqueeze(-1))  # (B, K)

    log_w = torch.log(weights)
    # (B, K, K): log(w_i w_j) + ll0_i + ll1_j
    scores = log_w.view(1, K, 1) + log_w.view(1, 1, K) + ll0.view(-1, K, 1) + ll1.view(-1, 1, K)
    expected = torch.logsumexp(scores.reshape(data.shape[0], -1), dim=-1)
    assert torch.allclose(ll, expected, atol=1e-5)


def test_pic_sum_eq4_no_cross_channel_mixing():
    # Two children, each outputs K channels. Sum should mix per-channel independently (Eq. 4).
    K = 3
    points = torch.linspace(-1, 1, K)
    rule = QuadratureRule(points=points, weights=torch.ones(K) / K)

    z = Scope([10])
    c1 = ConstantPICInput(Scope([0]), z, values=torch.log(torch.tensor([1.0, 2.0, 3.0])))
    c2 = ConstantPICInput(Scope([0]), z, values=torch.log(torch.tensor([10.0, 20.0, 30.0])))

    pic_sum = PICSum(inputs=[c1, c2], weights=torch.tensor([0.25, 0.75]), latent_scope=z)
    qpc = pic2qpc(pic_sum, rule)
    assert isinstance(qpc, WeightedSum)

    data = torch.zeros((2, 1))
    ll = qpc.log_likelihood(data)[0, 0, :, 0]  # (K,)

    expected = torch.log(0.25 * torch.tensor([1.0, 2.0, 3.0]) + 0.75 * torch.tensor([10.0, 20.0, 30.0]))
    assert torch.allclose(ll, expected, atol=1e-6)


def test_product_materialization_kronecker_vs_hadamard():
    K = 4
    rule = QuadratureRule(points=torch.linspace(-1, 1, K), weights=torch.ones(K) / K)

    left_a = ConstantPICInput(Scope([0]), Scope([2]), values=torch.zeros(K))
    right_a = ConstantPICInput(Scope([1]), Scope([3]), values=torch.zeros(K))
    pic_prod_a = PICProduct(left_a, right_a)
    qpc_a = pic2qpc(pic_prod_a, rule)
    assert isinstance(qpc_a, OuterProduct)

    left_b = ConstantPICInput(Scope([0]), Scope([2]), values=torch.zeros(K))
    right_b = ConstantPICInput(Scope([1]), Scope([2]), values=torch.zeros(K))
    pic_prod_b = PICProduct(left_b, right_b)
    qpc_b = pic2qpc(pic_prod_b, rule)
    assert isinstance(qpc_b, ElementwiseProduct)


class _NoLatentLeaf(Module):
    def __init__(self, scope: Scope) -> None:
        super().__init__()
        self.scope = scope
        self.in_shape = None
        self.out_shape = None

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([self.scope])

    def log_likelihood(self, data: Tensor, cache=None) -> Tensor:
        raise NotImplementedError

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None) -> Tensor:
        raise NotImplementedError

    def _sample(self, data: Tensor, sampling_ctx, cache, is_mpe: bool = False) -> Tensor:
        raise NotImplementedError

    def marginalize(self, marg_rvs, prune: bool = True, cache=None):
        raise NotImplementedError


class _BareModule(Module):
    def __init__(self) -> None:
        super().__init__()
        self.scope = Scope([0])
        from spflow.modules.module_shape import ModuleShape

        self.in_shape = ModuleShape(features=1, channels=1, repetitions=1)
        self.out_shape = ModuleShape(features=1, channels=1, repetitions=1)

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([Scope([0])])

    def log_likelihood(self, data: Tensor, cache=None) -> Tensor:
        return data.new_zeros((data.shape[0], 1, 1, 1))

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None) -> Tensor:
        raise NotImplementedError

    def _sample(self, data: Tensor, sampling_ctx, cache, is_mpe: bool = False) -> Tensor:
        raise NotImplementedError

    def marginalize(self, marg_rvs, prune: bool = True, cache=None):
        raise NotImplementedError


class _VariableChannelPICInput(Module):
    def __init__(self, scope: Scope, latent_scope: Scope, num_channels: int) -> None:
        super().__init__()
        self.scope = scope
        self.latent_scope = latent_scope
        self._num_channels = num_channels
        self.in_shape = None
        self.out_shape = None

    @property
    def feature_to_scope(self) -> np.ndarray:
        return np.array([self.scope])

    def materialize(self, quadrature_rule: QuadratureRule) -> Module:
        log_values = torch.log(torch.arange(1, self._num_channels + 1, dtype=quadrature_rule.points.dtype))
        return ConstantQPCModule(scope=self.scope, log_values=log_values)

    def log_likelihood(self, data: Tensor, cache=None) -> Tensor:
        raise NotImplementedError

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None) -> Tensor:
        raise NotImplementedError

    def _sample(self, data: Tensor, sampling_ctx, cache, is_mpe: bool = False) -> Tensor:
        raise NotImplementedError

    def marginalize(self, marg_rvs, prune: bool = True, cache=None):
        raise NotImplementedError


class _SelfMaterializingLeaf(ConstantQPCModule):
    def __init__(self, scope: Scope, latent_scope: Scope, log_values: Tensor) -> None:
        super().__init__(scope=scope, log_values=log_values)
        self.latent_scope = latent_scope

    def materialize(self, quadrature_rule: QuadratureRule) -> Module:
        return self


def test_picsum_constructor_validations():
    z = Scope([7])
    c1 = ConstantPICInput(Scope([0]), z, values=torch.log(torch.tensor([1.0, 2.0])))
    c2 = ConstantPICInput(Scope([0]), z, values=torch.log(torch.tensor([3.0, 4.0])))

    pic_sum = PICSum(inputs=[c1, c2], weights=torch.tensor([0.4, 0.6]), latent_scope=z)
    assert pic_sum.in_shape.features == 1
    assert pic_sum.out_shape.features == 1
    assert [s.query for s in pic_sum.feature_to_scope.tolist()] == [(0,)]

    with pytest.raises(InvalidParameterError):
        PICSum(inputs=[], weights=torch.tensor([]), latent_scope=z)

    with pytest.raises(ShapeError):
        PICSum(inputs=[c1, c2], weights=torch.ones((2, 1)), latent_scope=z)

    with pytest.raises(InvalidParameterError):
        PICSum(inputs=[c1, c2], weights=torch.tensor([0.0, 1.0]), latent_scope=z)

    with pytest.raises(InvalidParameterError):
        PICSum(inputs=[c1, c2], weights=torch.tensor([0.2, 0.2]), latent_scope=z)

    c3 = ConstantPICInput(Scope([1]), z, values=torch.log(torch.tensor([5.0, 6.0])))
    with pytest.raises(StructureError):
        PICSum(inputs=[c1, c3], weights=torch.tensor([0.5, 0.5]), latent_scope=z)


def test_picproduct_constructor_validations():
    left = _NoLatentLeaf(Scope([0]))
    right = _NoLatentLeaf(Scope([1]))
    prod = PICProduct(left, right)
    assert prod.latent_scope.empty()
    assert [s.query for s in prod.feature_to_scope.tolist()] == [(0,), (1,)]

    overlap_left = _NoLatentLeaf(Scope([0]))
    overlap_right = _NoLatentLeaf(Scope([0]))
    with pytest.raises(StructureError):
        PICProduct(overlap_left, overlap_right)


def test_rg2pic_assigns_leaf_latent_and_attaches_metadata():
    root = Region(Scope([0]))
    rg = RegionGraph(root)
    pic = rg2pic(rg, leaf_factory=lambda x, z: _NoLatentLeaf(x))
    assert getattr(pic, "latent_scope") == Scope([])
    assert getattr(pic, "_region_graph") is rg


def test_rg2pic_rejects_invalid_leaf_factory_and_partition_shapes():
    root = Region(Scope([0]))
    rg = RegionGraph(root)
    with pytest.raises(InvalidParameterError):
        rg2pic(rg, leaf_factory=None)  # type: ignore[arg-type]

    bad_leaf = Region(Scope([0, 1]))
    bad_root = Region(Scope([0, 1]))
    bad_root.children.append((bad_leaf,))
    bad_rg = RegionGraph(bad_root)
    with pytest.raises(StructureError):
        rg2pic(bad_rg, leaf_factory=lambda x, z: NormalPICInput(x, z))


def test_rg2pic_leaf_factory_must_preserve_assigned_latent_scope():
    left = Region(Scope([0]))
    right = Region(Scope([1]))
    root = Region(Scope([0, 1]))
    root.add_partition((left, right))
    rg = RegionGraph(root)

    def bad_leaf_factory(x_scope: Scope, z_scope: Scope) -> Module:
        return NormalPICInput(x_scope, Scope([]))

    with pytest.raises(StructureError):
        rg2pic(rg, leaf_factory=bad_leaf_factory)


def test_rg2pic_multiple_partitions_creates_picsum():
    l0 = Region(Scope([0]))
    l1 = Region(Scope([1]))
    root = Region(Scope([0, 1]))
    root.add_partition((l0, l1))
    root.add_partition((l0, l1))
    rg = RegionGraph(root)

    pic = rg2pic(
        rg,
        merge_strategy=MergeStrategy.TUCKER,
        leaf_factory=lambda x, z: NormalPICInput(x, z),
        function_factory=lambda z_dim, y_dim: ConstantOneFunction(),
        sum_weights_factory=lambda n: torch.tensor([0.25, 0.75]),
    )
    assert isinstance(pic, PICSum)
    assert torch.allclose(pic.weights, torch.tensor([0.25, 0.75]))


def test_rg2pic_raises_when_region_partitions_produce_different_latent_scopes():
    l0 = Region(Scope([0]))
    l1 = Region(Scope([1]))
    l2 = Region(Scope([2]))
    mid = Region(Scope([0, 1]))
    mid.add_partition((l0, l1))
    mid.add_partition((l0, l1))
    root = Region(Scope([0, 1, 2]))
    root.add_partition((mid, l2))
    rg = RegionGraph(root)

    with pytest.raises(StructureError):
        rg2pic(
            rg,
            merge_strategy=MergeStrategy.TUCKER,
            leaf_factory=lambda x, z: NormalPICInput(x, z),
            function_factory=lambda z_dim, y_dim: ConstantOneFunction(),
        )


def test_merge_units_cp_tucker_and_invalid_strategy():
    u1 = ConstantQPCModule(scope=Scope([0]), log_values=torch.log(torch.tensor([1.0, 2.0])))
    u2 = ConstantQPCModule(scope=Scope([1]), log_values=torch.log(torch.tensor([3.0, 4.0])))
    u1.latent_scope = Scope([10])  # type: ignore[attr-defined]
    u2.latent_scope = Scope([11])  # type: ignore[attr-defined]

    with pytest.raises(StructureError):
        _merge_units(
            u1=u1,
            u2=u2,
            rho=False,
            merge_strategy=MergeStrategy.CP,
            function_factory=None,
            alloc_latent=lambda: 99,
            depth=1,
            integral_group_factory=None,
            integral_groups={},
        )

    u3 = ConstantQPCModule(scope=Scope([0]), log_values=torch.log(torch.tensor([1.0, 2.0])))
    u4 = ConstantQPCModule(scope=Scope([1]), log_values=torch.log(torch.tensor([3.0, 4.0])))
    u3.latent_scope = Scope([])  # type: ignore[attr-defined]
    u4.latent_scope = Scope([])  # type: ignore[attr-defined]
    with pytest.raises(StructureError):
        _merge_units(
            u1=u3,
            u2=u4,
            rho=False,
            merge_strategy=MergeStrategy.CP,
            function_factory=None,
            alloc_latent=lambda: 99,
            depth=1,
            integral_group_factory=None,
            integral_groups={},
        )

    u5 = ConstantQPCModule(scope=Scope([0]), log_values=torch.log(torch.tensor([1.0, 2.0])))
    u6 = ConstantQPCModule(scope=Scope([1]), log_values=torch.log(torch.tensor([3.0, 4.0])))
    u5.latent_scope = Scope([10])  # type: ignore[attr-defined]
    u6.latent_scope = Scope([10])  # type: ignore[attr-defined]
    cp_out = _merge_units(
        u1=u5,
        u2=u6,
        rho=False,
        merge_strategy=MergeStrategy.CP,
        function_factory=lambda z_dim, y_dim: ConstantOneFunction(),
        alloc_latent=lambda: 42,
        depth=1,
        integral_group_factory=None,
        integral_groups={},
    )
    assert isinstance(cp_out, PICProduct)
    assert isinstance(cp_out.left, Integral)
    assert isinstance(cp_out.right, Integral)
    assert cp_out.left.latent_scope == Scope([42])

    tucker_out = _merge_units(
        u1=u1,
        u2=u2,
        rho=True,
        merge_strategy=MergeStrategy.TUCKER,
        function_factory=lambda z_dim, y_dim: ConstantOneFunction(),
        alloc_latent=lambda: 100,
        depth=1,
        integral_group_factory=None,
        integral_groups={},
    )
    assert isinstance(tucker_out, Integral)
    assert tucker_out.latent_scope.empty()

    with pytest.raises(InvalidParameterError):
        _merge_units(
            u1=u1,
            u2=u2,
            rho=False,
            merge_strategy="bad",  # type: ignore[arg-type]
            function_factory=None,
            alloc_latent=lambda: 100,
            depth=1,
            integral_group_factory=None,
            integral_groups={},
        )


def test_maybe_attach_function_group_reuses_group_by_key():
    input_mod = ConstantQPCModule(scope=Scope([0]), log_values=torch.log(torch.tensor([1.0, 2.0])))
    integral = Integral(
        input_module=input_mod,
        latent_scope=Scope([5]),
        integrated_latent_scope=Scope([3]),
        function=ConstantOneFunction(),
    )
    other = Integral(
        input_module=input_mod,
        latent_scope=Scope([6]),
        integrated_latent_scope=Scope([3]),
        function=ConstantOneFunction(),
    )
    groups = {}

    _maybe_attach_function_group(
        integral=integral,
        depth=2,
        z_dim=1,
        y_dim=1,
        integral_group_factory=lambda depth, z_dim, y_dim: FunctionGroup(
            sharing_type="c", input_dim=z_dim + y_dim, hidden_dim=8
        ),
        integral_groups=groups,
    )
    _maybe_attach_function_group(
        integral=other,
        depth=2,
        z_dim=1,
        y_dim=1,
        integral_group_factory=lambda depth, z_dim, y_dim: FunctionGroup(
            sharing_type="c", input_dim=z_dim + y_dim, hidden_dim=8
        ),
        integral_groups=groups,
    )

    assert isinstance(integral.function, FunctionGroup)
    assert integral.function is other.function
    assert integral.function_head_idx == 0
    assert other.function_head_idx == 1


def test_maybe_attach_function_group_skips_zero_dim_latents():
    input_mod = ConstantQPCModule(scope=Scope([0]), log_values=torch.log(torch.tensor([1.0, 2.0])))
    integral = Integral(
        input_module=input_mod,
        latent_scope=Scope([]),
        integrated_latent_scope=Scope([3]),
        function=ConstantOneFunction(),
    )
    groups = {}

    _maybe_attach_function_group(
        integral=integral,
        depth=2,
        z_dim=0,
        y_dim=1,
        integral_group_factory=lambda depth, z_dim, y_dim: (_ for _ in ()).throw(
            RuntimeError("must not be called")
        ),
        integral_groups=groups,
    )
    assert groups == {}
    assert isinstance(integral.function, ConstantOneFunction)


def test_pic2qpc_mode_and_quadrature_validation_errors():
    pic = NormalPICInput(Scope([0]), Scope([]))
    rule = QuadratureRule(points=torch.linspace(-1, 1, 2), weights=torch.tensor([0.5, 0.5]))

    with pytest.raises(InvalidParameterError):
        pic2qpc(pic, rule, mode="invalid")

    with pytest.raises(InvalidParameterError):
        pic2qpc(pic, rule, mode="tensorized")

    with pytest.raises(StructureError):
        pic2qpc(pic, rule, mode="tensorized", tensorized_config=TensorizedQPCConfig(leaf_type="normal"))

    setattr(pic, "_region_graph", object())
    with pytest.raises(StructureError):
        pic2qpc(pic, rule, mode="tensorized", tensorized_config=TensorizedQPCConfig(leaf_type="normal"))

    bad_rule = QuadratureRule(points=torch.ones(2, 1), weights=torch.ones(2))
    with pytest.raises(ShapeError):
        pic2qpc(pic, bad_rule)

    bad_rule = QuadratureRule(points=torch.ones(2), weights=torch.ones(3))
    with pytest.raises(ShapeError):
        pic2qpc(pic, bad_rule)

    bad_rule = QuadratureRule(points=torch.ones(2), weights=torch.tensor([0.6, -0.1]))
    with pytest.raises(InvalidParameterError):
        pic2qpc(pic, bad_rule)


def test_pic2qpc_raises_for_sum_channel_mismatch_and_integral_errors():
    rule = QuadratureRule(points=torch.linspace(-1, 1, 2), weights=torch.tensor([0.5, 0.5]))

    z = Scope([10])
    c1 = _VariableChannelPICInput(scope=Scope([0]), latent_scope=z, num_channels=2)
    c2 = _VariableChannelPICInput(scope=Scope([0]), latent_scope=z, num_channels=3)
    bad_sum = PICSum(inputs=[c1, c2], weights=torch.tensor([0.5, 0.5]), latent_scope=z)
    with pytest.raises(ShapeError):
        pic2qpc(bad_sum, rule)

    child = _SelfMaterializingLeaf(
        scope=Scope([0]), latent_scope=Scope([9]), log_values=torch.log(torch.tensor([1.0, 2.0]))
    )
    bad_integral = Integral(
        input_module=child,
        latent_scope=Scope([]),
        integrated_latent_scope=Scope([9]),
        function=None,
    )
    with pytest.raises(StructureError):
        pic2qpc(bad_integral, rule)

    mismatch_integral = Integral(
        input_module=child,
        latent_scope=Scope([]),
        integrated_latent_scope=Scope([9, 10]),
        function=ConstantOneFunction(),
    )
    with pytest.raises(ShapeError):
        pic2qpc(mismatch_integral, rule)


def test_pic2qpc_function_group_materialization_path():
    rule = QuadratureRule(points=torch.linspace(-1, 1, 2), weights=torch.tensor([0.5, 0.5]))
    group = FunctionGroup(sharing_type="c", input_dim=2, hidden_dim=8)

    left = _SelfMaterializingLeaf(
        scope=Scope([0]), latent_scope=Scope([1]), log_values=torch.log(torch.tensor([1.0, 2.0]))
    )
    right = _SelfMaterializingLeaf(
        scope=Scope([0]), latent_scope=Scope([1]), log_values=torch.log(torch.tensor([3.0, 4.0]))
    )
    extra = _SelfMaterializingLeaf(
        scope=Scope([0]), latent_scope=Scope([1]), log_values=torch.log(torch.tensor([5.0, 6.0]))
    )

    i1 = Integral(
        input_module=left, latent_scope=Scope([2]), integrated_latent_scope=Scope([1]), function=group
    )
    i2 = Integral(
        input_module=right, latent_scope=Scope([2]), integrated_latent_scope=Scope([1]), function=group
    )
    i3 = Integral(
        input_module=extra, latent_scope=Scope([2]), integrated_latent_scope=Scope([1]), function=group
    )

    i1.function_head_idx = group.add_unit(i1)
    i2.function_head_idx = group.add_unit(i2)
    group.add_unit(i3)  # head_idx intentionally left None to exercise skip branch

    pic_sum = PICSum(inputs=[i1, i2], weights=torch.tensor([0.5, 0.5]), latent_scope=Scope([2]))
    qpc = pic2qpc(pic_sum, rule)
    assert isinstance(qpc, WeightedSum)


def test_pic2qpc_integral_with_empty_integrated_scope_uses_scalar_kron_weights():
    rule = QuadratureRule(points=torch.linspace(-1, 1, 3), weights=torch.ones(3) / 3)
    child = _SelfMaterializingLeaf(scope=Scope([0]), latent_scope=Scope([]), log_values=torch.tensor([0.0]))
    integral = Integral(
        input_module=child,
        latent_scope=Scope([2]),
        integrated_latent_scope=Scope([]),
        function=ConstantOneFunction(),
    )
    qpc = pic2qpc(integral, rule)
    assert isinstance(qpc, WeightedSum)
    assert qpc.out_shape.channels == 3


def test_pic2qpc_function_group_empty_units_and_zero_dim_bucket_paths():
    rule = QuadratureRule(points=torch.linspace(-1, 1, 2), weights=torch.tensor([0.5, 0.5]))
    empty_group = FunctionGroup(sharing_type="c", input_dim=2, hidden_dim=8)
    empty_group.add_unit(object())  # non-Integral placeholder: keeps evaluate_batched valid
    child = _SelfMaterializingLeaf(
        scope=Scope([0]), latent_scope=Scope([1]), log_values=torch.log(torch.tensor([1.0, 2.0]))
    )
    i_empty_group = Integral(
        input_module=child,
        latent_scope=Scope([2]),
        integrated_latent_scope=Scope([1]),
        function=empty_group,
    )
    qpc = pic2qpc(i_empty_group, rule)
    assert isinstance(qpc, WeightedSum)

    zero_dim_group = FunctionGroup(sharing_type="c", input_dim=1, hidden_dim=8)
    root_like_integral = Integral(
        input_module=child,
        latent_scope=Scope([]),
        integrated_latent_scope=Scope([1]),
        function=zero_dim_group,
    )
    root_like_integral.function_head_idx = zero_dim_group.add_unit(root_like_integral)
    qpc_root = pic2qpc(root_like_integral, rule)
    assert isinstance(qpc_root, WeightedSum)


def test_pic2qpc_function_group_revisit_skips_already_materialized_units():
    rule = QuadratureRule(points=torch.linspace(-1, 1, 2), weights=torch.tensor([0.5, 0.5]))
    group = FunctionGroup(sharing_type="c", input_dim=2, hidden_dim=8)

    left = _SelfMaterializingLeaf(
        scope=Scope([0]), latent_scope=Scope([1]), log_values=torch.log(torch.tensor([1.0, 2.0]))
    )
    right = _SelfMaterializingLeaf(
        scope=Scope([0]), latent_scope=Scope([1]), log_values=torch.log(torch.tensor([3.0, 4.0]))
    )
    i1 = Integral(
        input_module=left, latent_scope=Scope([2]), integrated_latent_scope=Scope([1]), function=group
    )
    i2 = Integral(
        input_module=right, latent_scope=Scope([2]), integrated_latent_scope=Scope([1]), function=group
    )
    i1.function_head_idx = group.add_unit(i1)
    group.add_unit(i2)  # leave i2.function_head_idx unset; fallback path should still materialize i2

    pic_sum = PICSum(inputs=[i1, i2], weights=torch.tensor([0.5, 0.5]), latent_scope=Scope([2]))
    qpc = pic2qpc(pic_sum, rule)
    assert isinstance(qpc, WeightedSum)


def test_pic2qpc_function_group_raises_on_grouped_child_channel_mismatch():
    rule = QuadratureRule(points=torch.linspace(-1, 1, 2), weights=torch.tensor([0.5, 0.5]))
    group = FunctionGroup(sharing_type="c", input_dim=3, hidden_dim=8)
    child = _SelfMaterializingLeaf(
        scope=Scope([0]), latent_scope=Scope([9]), log_values=torch.log(torch.tensor([1.0, 2.0]))
    )
    bad = Integral(
        input_module=child,
        latent_scope=Scope([2]),
        integrated_latent_scope=Scope([9, 10]),
        function=group,
    )
    bad.function_head_idx = group.add_unit(bad)

    with pytest.raises(ShapeError):
        pic2qpc(bad, rule)


def test_pic2qpc_unsupported_node_type_raises():
    rule = QuadratureRule(points=torch.linspace(-1, 1, 2), weights=torch.tensor([0.5, 0.5]))
    with pytest.raises(StructureError):
        pic2qpc(_BareModule(), rule)
