"""Cross-utility contracts for inner/triple product operators."""

from __future__ import annotations

import copy

import pytest
import torch

from spflow.exceptions import ShapeError, UnsupportedOperationError
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.bernoulli import Bernoulli
from spflow.modules.leaves.categorical import Categorical
from spflow.modules.leaves.normal import Normal
from spflow.modules.ops.cat import Cat
from spflow.modules.products.product import Product
from spflow.modules.sums.sum import Sum
from spflow.utils.inner_product_core import (
    inner_product_matrix,
    log_self_inner_product_scalar,
    triple_product_scalar,
    triple_product_tensor,
)


def _normal(scope: list[int], loc: float = 0.0, scale: float = 1.0) -> Normal:
    f = len(scope)
    return Normal(
        scope=Scope(scope),
        out_channels=1,
        num_repetitions=1,
        loc=torch.full((f, 1, 1), loc),
        scale=torch.full((f, 1, 1), scale),
    )


def _bernoulli(scope: list[int], p: float = 0.5, out_channels: int = 1) -> Bernoulli:
    f = len(scope)
    return Bernoulli(
        scope=Scope(scope),
        out_channels=out_channels,
        num_repetitions=1,
        probs=torch.full((f, out_channels, 1), p),
    )


@pytest.mark.contract
def test_inner_product_matrix_contracts():
    pa = Product([_bernoulli([0], 0.2), _bernoulli([1], 0.4)])
    pb = Product([_bernoulli([0], 0.7), _bernoulli([1], 0.6)])
    k = inner_product_matrix(pa, pb)
    assert k.shape == (1, 1, 1, 1)
    assert torch.isfinite(k).all()

    sa = Sum(inputs=[_bernoulli([0], 0.3), _bernoulli([0], 0.8)], weights=torch.tensor([[[[0.2]], [[0.8]]]]))
    sb = Sum(inputs=[_bernoulli([0], 0.5), _bernoulli([0], 0.9)], weights=torch.tensor([[[[0.6]], [[0.4]]]]))
    ks = inner_product_matrix(sa, sb)
    assert ks.shape == (1, 1, 1, 1)
    assert torch.isfinite(ks).all()

    # Mixing a leaf with a composite node has no bilinear contract and must fail explicitly.
    with pytest.raises(UnsupportedOperationError):
        inner_product_matrix(_normal([0]), sa)


@pytest.mark.contract
def test_triple_product_operator_contracts():
    prod_a = Product([_bernoulli([0], 0.2), _bernoulli([1], 0.4)])
    prod_b = Product([_bernoulli([0], 0.5), _bernoulli([1], 0.6)])
    prod_c = Product([_bernoulli([0], 0.7), _bernoulli([1], 0.8)])
    tp = triple_product_tensor(prod_a, prod_b, prod_c)
    assert tp.shape == (1, 1, 1, 1, 1)
    assert torch.isfinite(tp).all()

    sum_a = Sum(
        inputs=[_bernoulli([0], 0.2), _bernoulli([0], 0.8)], weights=torch.tensor([[[[0.3]], [[0.7]]]])
    )
    sum_b = Sum(
        inputs=[_bernoulli([0], 0.5), _bernoulli([0], 0.6)], weights=torch.tensor([[[[0.4]], [[0.6]]]])
    )
    sum_c = Sum(
        inputs=[_bernoulli([0], 0.1), _bernoulli([0], 0.9)], weights=torch.tensor([[[[0.2]], [[0.8]]]])
    )
    ts = triple_product_tensor(sum_a, sum_b, sum_c)
    assert ts.shape == (1, 1, 1, 1, 1)
    assert torch.isfinite(ts).all()

    c0 = Cat(inputs=[_bernoulli([0]), _bernoulli([0])], dim=2)
    c1 = copy.deepcopy(c0)
    c2 = copy.deepcopy(c0)
    c1.dim = 1
    with pytest.raises(ShapeError):
        triple_product_tensor(c0, c1, c2)

    with pytest.raises(UnsupportedOperationError):
        triple_product_tensor(_normal([0]), sum_b, sum_c)


@pytest.mark.contract
def test_scalar_shape_contracts():
    # Scalar helpers are only defined for single-feature leaves.
    with pytest.raises(ShapeError):
        log_self_inner_product_scalar(_bernoulli([0, 1]))

    # Triple scalar integration also requires all operands to share the same single scope.
    with pytest.raises(ShapeError):
        triple_product_scalar(_bernoulli([0, 1]), _bernoulli([0]), _bernoulli([0]))


@pytest.mark.contract
@pytest.mark.numerical
def test_leaf_happy_paths_contract():
    na = _normal([0], loc=0.1, scale=0.7)
    nb = _normal([0], loc=-0.2, scale=1.1)
    nc = _normal([0], loc=0.4, scale=0.9)
    tn = triple_product_tensor(na, nb, nc)
    assert tn.shape == (1, 1, 1, 1, 1)
    assert torch.isfinite(tn).all()
    # Positive Gaussian overlap protects against sign/normalization regressions in closed forms.
    assert float(tn[0, 0, 0, 0, 0].item()) > 0.0

    c1 = Categorical(
        scope=Scope([0]), out_channels=1, num_repetitions=1, K=3, probs=torch.tensor([[[[0.1, 0.3, 0.6]]]])
    )
    c2 = Categorical(
        scope=Scope([0]), out_channels=1, num_repetitions=1, K=3, probs=torch.tensor([[[[0.5, 0.25, 0.25]]]])
    )
    c3 = Categorical(
        scope=Scope([0]), out_channels=1, num_repetitions=1, K=3, probs=torch.tensor([[[[0.2, 0.4, 0.4]]]])
    )
    tc = triple_product_tensor(c1, c2, c3)
    expected = torch.sum(c1.probs.to(torch.float64) * c2.probs.to(torch.float64) * c3.probs.to(torch.float64))
    torch.testing.assert_close(tc[0, 0, 0, 0, 0], expected, rtol=0.0, atol=0.0)
