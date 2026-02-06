import copy

import pytest
import torch

from spflow.exceptions import ShapeError, UnsupportedOperationError
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.bernoulli import Bernoulli
from spflow.modules.leaves.categorical import Categorical
from spflow.modules.leaves.cltree import CLTree
from spflow.modules.leaves.gamma import Gamma
from spflow.modules.leaves.hypergeometric import Hypergeometric
from spflow.modules.leaves.normal import Normal
from spflow.modules.leaves.piecewise_linear import PiecewiseLinear
from spflow.modules.module_shape import ModuleShape
from spflow.modules.ops.cat import Cat
from spflow.modules.products.product import Product
from spflow.modules.sums.sum import Sum
from spflow.utils.cache import Cache
from spflow.utils.domain import Domain
from spflow.utils.inner_product_core import (
    _neg_binom_logpmf,
    inner_product_matrix,
    leaf_inner_product,
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


def _piecewise(
    *,
    xs: torch.Tensor | None = None,
    ys: torch.Tensor | None = None,
    initialized: bool = True,
    domains: list[Domain] | None = None,
) -> PiecewiseLinear:
    leaf = PiecewiseLinear(scope=Scope([0]), out_channels=1, num_repetitions=1)
    if xs is not None and ys is not None:
        leaf.xs = [[[[xs.to(dtype=torch.float64)]]]]  # type: ignore[assignment]
        leaf.ys = [[[[ys.to(dtype=torch.float64)]]]]  # type: ignore[assignment]
    leaf.is_initialized = initialized
    leaf.domains = domains  # type: ignore[assignment]
    return leaf


def test_leaf_inner_product_validation_branches():
    a = _normal([0])
    b = _normal([1])
    with pytest.raises(ShapeError, match="Scopes must match"):
        leaf_inner_product(a, b)

    b = _normal([0])
    b.out_shape = ModuleShape(2, 1, 1)
    with pytest.raises(ShapeError, match="Leaf features must match"):
        leaf_inner_product(a, b)

    b = _normal([0])
    b.out_shape = ModuleShape(1, 1, 2)
    with pytest.raises(ShapeError, match="Leaf repetitions must match"):
        leaf_inner_product(a, b)


def test_leaf_inner_product_distribution_guardrails():
    c1 = Categorical(
        scope=Scope([0]), out_channels=1, num_repetitions=1, K=2, probs=torch.tensor([[[[0.3, 0.7]]]])
    )
    c2 = Categorical(
        scope=Scope([0]), out_channels=1, num_repetitions=1, K=3, probs=torch.tensor([[[[0.2, 0.3, 0.5]]]])
    )
    with pytest.raises(ShapeError, match="Categorical K mismatch"):
        leaf_inner_product(c1, c2)

    g1 = Gamma(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        concentration=torch.tensor([[[0.4]]]),
        rate=torch.tensor([[[1.0]]]),
    )
    g2 = Gamma(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        concentration=torch.tensor([[[0.5]]]),
        rate=torch.tensor([[[2.0]]]),
    )
    with pytest.raises(UnsupportedOperationError, match="concentration_a \\+ concentration_b > 1"):
        leaf_inner_product(g1, g2)

    h1 = Hypergeometric(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        K=torch.tensor([[[2.0]]]),
        N=torch.tensor([[[7.0]]]),
        n=torch.tensor([[[2.0]]]),
    )
    h2 = Hypergeometric(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        K=torch.tensor([[[3.0]]]),
        N=torch.tensor([[[8.0]]]),
        n=torch.tensor([[[2.0]]]),
    )
    with pytest.raises(ShapeError, match="matching N"):
        leaf_inner_product(h1, h2)


def test_leaf_inner_product_piecewise_guardrails_and_short_grid():
    a = _piecewise(initialized=False)
    b = _piecewise(initialized=False)
    with pytest.raises(UnsupportedOperationError, match="requires both leaves to be initialized"):
        leaf_inner_product(a, b)

    a = _piecewise(initialized=True, domains=None)
    b = _piecewise(initialized=True, domains=None)
    with pytest.raises(UnsupportedOperationError, match="requires domains"):
        leaf_inner_product(a, b)

    a = _piecewise(initialized=True, domains=[Domain.discrete_range(0, 1)])
    b = _piecewise(initialized=True, domains=[Domain.discrete_range(0, 1)])
    with pytest.raises(UnsupportedOperationError, match="supports continuous domains only"):
        leaf_inner_product(a, b)

    single = _piecewise(
        xs=torch.tensor([1.0], dtype=torch.float64),
        ys=torch.tensor([0.7], dtype=torch.float64),
        initialized=True,
        domains=[Domain.continuous_range(0.0, 2.0)],
    )
    out = leaf_inner_product(single, single)
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_leaf_inner_product_cltree_and_unsupported_branches():
    a = CLTree(scope=Scope([0, 1]), out_channels=1, num_repetitions=1, K=2)
    b = CLTree(scope=Scope([0, 1]), out_channels=1, num_repetitions=1, K=2)
    b.K = 3
    with pytest.raises(ShapeError, match="CLTree K mismatch"):
        leaf_inner_product(a, b)

    b = CLTree(scope=Scope([0, 1]), out_channels=1, num_repetitions=1, K=2)
    b.parents = a.parents.clone()
    b.parents[1] = -1 if int(a.parents[1].item()) != -1 else 0
    with pytest.raises(UnsupportedOperationError, match="identical tree structure"):
        leaf_inner_product(a, b)

    with pytest.raises(UnsupportedOperationError, match="not implemented"):
        leaf_inner_product(_normal([0]), _bernoulli([0]))


def test_inner_product_matrix_shape_and_cat_errors():
    a = _normal([0])
    b = _normal([0])
    b.out_shape = ModuleShape(2, 1, 1)
    with pytest.raises(ShapeError, match="Feature mismatch"):
        inner_product_matrix(a, b)

    b = _normal([0])
    b.out_shape = ModuleShape(1, 1, 2)
    with pytest.raises(ShapeError, match="Repetition mismatch"):
        inner_product_matrix(a, b)

    c0 = Cat(inputs=[_bernoulli([0]), _bernoulli([0])], dim=2)
    c1 = copy.deepcopy(c0)
    c1.dim = 1
    with pytest.raises(ShapeError, match="Cat dim mismatch"):
        inner_product_matrix(c0, c1)

    d0 = Cat(inputs=[_bernoulli([0, 1])], dim=1)
    d1 = Cat(inputs=[_bernoulli([0]), _bernoulli([1])], dim=1)
    with pytest.raises(ShapeError, match="Cat arity mismatch"):
        inner_product_matrix(d0, d1)

    c_bad = copy.deepcopy(c0)
    c_bad.dim = 3
    with pytest.raises(UnsupportedOperationError, match="does not support Cat"):
        inner_product_matrix(c_bad, c_bad)


def test_inner_product_matrix_product_sum_and_unsupported_paths():
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

    with pytest.raises(UnsupportedOperationError, match="not implemented"):
        inner_product_matrix(_normal([0]), sa)


def test_log_self_inner_product_scalar_shape_guard():
    with pytest.raises(ShapeError, match="Expected scalar output"):
        log_self_inner_product_scalar(_bernoulli([0, 1]))


def test_triple_product_cache_symmetry_and_memo_creation():
    a = _bernoulli([0], 0.2)
    b = _bernoulli([0], 0.8)
    c = _bernoulli([0], 0.5)
    cache = Cache()
    t_ab = triple_product_tensor(a, b, c, cache=cache)
    t_ba = triple_product_tensor(b, a, c, cache=cache)
    torch.testing.assert_close(t_ab, t_ba.permute(0, 2, 1, 3, 4))
    assert "_triple_product_memo" in cache.extras


def test_triple_product_shape_guards():
    a = _bernoulli([0])
    b = _bernoulli([0])
    c = _bernoulli([0])

    b.out_shape = ModuleShape(2, 1, 1)
    with pytest.raises(ShapeError, match="Feature mismatch for triple product"):
        triple_product_tensor(a, b, c)

    b = _bernoulli([0])
    b.out_shape = ModuleShape(1, 1, 2)
    with pytest.raises(ShapeError, match="Repetition mismatch for triple product"):
        triple_product_tensor(a, b, c)


def test_triple_product_leaf_and_piecewise_guardrails():
    c1 = Categorical(
        scope=Scope([0]), out_channels=1, num_repetitions=1, K=2, probs=torch.tensor([[[[0.4, 0.6]]]])
    )
    c2 = Categorical(
        scope=Scope([0]), out_channels=1, num_repetitions=1, K=2, probs=torch.tensor([[[[0.1, 0.9]]]])
    )
    c3 = Categorical(
        scope=Scope([0]), out_channels=1, num_repetitions=1, K=3, probs=torch.tensor([[[[0.1, 0.2, 0.7]]]])
    )
    with pytest.raises(ShapeError, match="Categorical K mismatch for triple product"):
        triple_product_tensor(c1, c2, c3)

    h1 = Hypergeometric(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        K=torch.tensor([[[2.0]]]),
        N=torch.tensor([[[7.0]]]),
        n=torch.tensor([[[2.0]]]),
    )
    h2 = Hypergeometric(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        K=torch.tensor([[[3.0]]]),
        N=torch.tensor([[[7.0]]]),
        n=torch.tensor([[[2.0]]]),
    )
    h3 = Hypergeometric(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        K=torch.tensor([[[3.0]]]),
        N=torch.tensor([[[8.0]]]),
        n=torch.tensor([[[2.0]]]),
    )
    with pytest.raises(ShapeError, match="requires matching N"):
        triple_product_tensor(h1, h2, h3)

    p = _piecewise(initialized=False)
    with pytest.raises(UnsupportedOperationError, match="requires all leaves to be initialized"):
        triple_product_tensor(p, p, p)

    p = _piecewise(initialized=True, domains=None)
    with pytest.raises(UnsupportedOperationError, match="requires domains"):
        triple_product_tensor(p, p, p)

    p = _piecewise(initialized=True, domains=[Domain.discrete_range(0, 1)])
    with pytest.raises(UnsupportedOperationError, match="supports continuous domains only"):
        triple_product_tensor(p, p, p)

    single = _piecewise(
        xs=torch.tensor([0.0], dtype=torch.float64),
        ys=torch.tensor([0.8], dtype=torch.float64),
        initialized=True,
        domains=[Domain.continuous_range(0.0, 1.0)],
    )
    out = triple_product_tensor(single, single, single)
    torch.testing.assert_close(out, torch.zeros_like(out))

    with pytest.raises(UnsupportedOperationError, match="Leaf triple product not implemented"):
        triple_product_tensor(_normal([0]), _bernoulli([0]), _bernoulli([0]))


def test_triple_product_cat_product_sum_and_fallback_paths():
    prod_a = Product([_bernoulli([0], 0.2), _bernoulli([1], 0.4)])
    prod_b = Product([_bernoulli([0], 0.5), _bernoulli([1], 0.6)])
    prod_c = Product([_bernoulli([0], 0.7), _bernoulli([1], 0.8)])
    tp = triple_product_tensor(prod_a, prod_b, prod_c)
    assert tp.shape == (1, 1, 1, 1, 1)
    assert torch.isfinite(tp).all()

    sum_a = Sum(
        inputs=[_bernoulli([0], 0.2), _bernoulli([0], 0.8)],
        weights=torch.tensor([[[[0.3]], [[0.7]]]]),
    )
    sum_b = Sum(
        inputs=[_bernoulli([0], 0.5), _bernoulli([0], 0.6)],
        weights=torch.tensor([[[[0.4]], [[0.6]]]]),
    )
    sum_c = Sum(
        inputs=[_bernoulli([0], 0.1), _bernoulli([0], 0.9)],
        weights=torch.tensor([[[[0.2]], [[0.8]]]]),
    )
    ts = triple_product_tensor(sum_a, sum_b, sum_c)
    assert ts.shape == (1, 1, 1, 1, 1)
    assert torch.isfinite(ts).all()

    c0 = Cat(inputs=[_bernoulli([0]), _bernoulli([0])], dim=2)
    c1 = copy.deepcopy(c0)
    c2 = copy.deepcopy(c0)
    c1.dim = 1
    with pytest.raises(ShapeError, match="Cat dim mismatch for triple product"):
        triple_product_tensor(c0, c1, c2)

    d0 = Cat(inputs=[_bernoulli([0, 1])], dim=1)
    d1 = Cat(inputs=[_bernoulli([0]), _bernoulli([1])], dim=1)
    d2 = Cat(inputs=[_bernoulli([0]), _bernoulli([1])], dim=1)
    with pytest.raises(ShapeError, match="Cat arity mismatch for triple product"):
        triple_product_tensor(d0, d1, d2)

    c_bad = copy.deepcopy(c0)
    c_bad.dim = 3
    with pytest.raises(UnsupportedOperationError, match="does not support Cat"):
        triple_product_tensor(c_bad, c_bad, c_bad)

    with pytest.raises(UnsupportedOperationError, match="triple_product_tensor not implemented"):
        triple_product_tensor(_normal([0]), sum_b, sum_c)


def test_triple_product_scalar_shape_guard():
    with pytest.raises(ShapeError, match="expects all modules to have out_shape == \\(1,1,1\\)"):
        triple_product_scalar(_bernoulli([0, 1]), _bernoulli([0]), _bernoulli([0]))


def test_private_neg_binom_logpmf_matches_torch():
    ks = torch.arange(0, 6, dtype=torch.float64)
    r = torch.tensor(3.0, dtype=torch.float64)
    p = torch.tensor(0.7, dtype=torch.float64)
    got = _neg_binom_logpmf(ks, r, p)
    ref = torch.distributions.NegativeBinomial(total_count=r, probs=p).log_prob(ks)
    torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)


def test_inner_product_product_cache_write_branch():
    pa = Product([_bernoulli([0], 0.2), _bernoulli([1], 0.3)])
    pb = Product([_bernoulli([0], 0.8), _bernoulli([1], 0.6)])
    cache = Cache()
    k1 = inner_product_matrix(pa, pb, cache=cache)
    k2 = inner_product_matrix(pa, pb, cache=cache)
    torch.testing.assert_close(k1, k2)


def test_triple_product_leaf_normal_and_categorical_happy_paths():
    na = _normal([0], loc=0.1, scale=0.7)
    nb = _normal([0], loc=-0.2, scale=1.1)
    nc = _normal([0], loc=0.4, scale=0.9)
    tn = triple_product_tensor(na, nb, nc)
    assert tn.shape == (1, 1, 1, 1, 1)
    assert torch.isfinite(tn).all()
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


def test_triple_product_cache_direct_hit_and_cache_write_paths():
    a = _bernoulli([0], 0.2)
    b = _bernoulli([0], 0.6)
    c = _bernoulli([0], 0.8)
    cache = Cache()
    t1 = triple_product_tensor(a, b, c, cache=cache)
    t2 = triple_product_tensor(a, b, c, cache=cache)
    torch.testing.assert_close(t1, t2)

    cat1_a = Cat(inputs=[_bernoulli([0], 0.2), _bernoulli([1], 0.3)], dim=1)
    cat1_b = Cat(inputs=[_bernoulli([0], 0.4), _bernoulli([1], 0.5)], dim=1)
    cat1_c = Cat(inputs=[_bernoulli([0], 0.6), _bernoulli([1], 0.7)], dim=1)
    t_cat1 = triple_product_tensor(cat1_a, cat1_b, cat1_c, cache=cache)
    assert t_cat1.shape == (2, 1, 1, 1, 1)

    cat2_a = Cat(inputs=[_bernoulli([0], 0.2, out_channels=1), _bernoulli([0], 0.3, out_channels=1)], dim=2)
    cat2_b = Cat(inputs=[_bernoulli([0], 0.4, out_channels=1), _bernoulli([0], 0.5, out_channels=1)], dim=2)
    cat2_c = Cat(inputs=[_bernoulli([0], 0.6, out_channels=1), _bernoulli([0], 0.7, out_channels=1)], dim=2)
    t_cat2 = triple_product_tensor(cat2_a, cat2_b, cat2_c, cache=cache)
    assert t_cat2.shape == (1, 2, 2, 2, 1)

    prod_a = Product([_bernoulli([0], 0.2), _bernoulli([1], 0.2)])
    prod_b = Product([_bernoulli([0], 0.3), _bernoulli([1], 0.3)])
    prod_c = Product([_bernoulli([0], 0.4), _bernoulli([1], 0.4)])
    t_prod = triple_product_tensor(prod_a, prod_b, prod_c, cache=cache)
    assert t_prod.shape == (1, 1, 1, 1, 1)

    sum_a = Sum(
        inputs=[_bernoulli([0], 0.2), _bernoulli([0], 0.8)], weights=torch.tensor([[[[0.3]], [[0.7]]]])
    )
    sum_b = Sum(
        inputs=[_bernoulli([0], 0.1), _bernoulli([0], 0.9)], weights=torch.tensor([[[[0.4]], [[0.6]]]])
    )
    sum_c = Sum(
        inputs=[_bernoulli([0], 0.5), _bernoulli([0], 0.6)], weights=torch.tensor([[[[0.2]], [[0.8]]]])
    )
    t_sum = triple_product_tensor(sum_a, sum_b, sum_c, cache=cache)
    assert t_sum.shape == (1, 1, 1, 1, 1)
