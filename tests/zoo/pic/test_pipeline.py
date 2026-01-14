"""Tests for PIC pipeline functions (rg2pic, pic2qpc)."""

import numpy as np
import torch
from torch import Tensor, nn

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
        self, num_samples=None, data=None, is_mpe=False, cache=None, sampling_ctx=None
    ) -> Tensor:  # pragma: no cover
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
        self, num_samples=None, data=None, is_mpe=False, cache=None, sampling_ctx=None
    ) -> Tensor:  # pragma: no cover
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

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None, sampling_ctx=None) -> Tensor:
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
