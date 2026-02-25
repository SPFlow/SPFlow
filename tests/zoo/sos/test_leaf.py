"""SOS integration/math checks for leaf-related inner/triple products.

This suite intentionally remains outside `tests/modules/leaves`.
"""

import math

import torch

from spflow.meta.data.scope import Scope
from spflow.modules.leaves.binomial import Binomial
from spflow.modules.leaves.histogram import Histogram
from spflow.modules.leaves.hypergeometric import Hypergeometric
from spflow.modules.leaves.negative_binomial import NegativeBinomial
from spflow.modules.leaves.piecewise_linear import PiecewiseLinear, interp
from spflow.utils.domain import Domain
from spflow.zoo.sos import inner_product_matrix, triple_product_scalar


def _binomial_pmf(ks: torch.Tensor, *, n: int, p: float) -> torch.Tensor:
    dist = torch.distributions.Binomial(
        total_count=torch.tensor(float(n), dtype=torch.float64),
        probs=torch.tensor(float(p), dtype=torch.float64),
    )
    return torch.exp(dist.log_prob(ks.to(dtype=torch.float64)))


def test_binomial_inner_and_triple_matches_exact_enumeration():
    n1, p1 = 6, 0.2
    n2, p2 = 8, 0.75
    n3, p3 = 5, 0.55

    a = Binomial(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        total_count=torch.tensor([[[float(n1)]]]),
        probs=torch.tensor([[[p1]]]),
    )
    b = Binomial(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        total_count=torch.tensor([[[float(n2)]]]),
        probs=torch.tensor([[[p2]]]),
    )
    c = Binomial(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        total_count=torch.tensor([[[float(n3)]]]),
        probs=torch.tensor([[[p3]]]),
    )

    ip = inner_product_matrix(a, b)[0, 0, 0, 0]
    ks_ip = torch.arange(0, min(n1, n2) + 1, dtype=torch.float64)
    expected_ip = torch.sum(_binomial_pmf(ks_ip, n=n1, p=p1) * _binomial_pmf(ks_ip, n=n2, p=p2))
    torch.testing.assert_close(ip, expected_ip.to(dtype=ip.dtype, device=ip.device), rtol=1e-6, atol=1e-9)

    tp = triple_product_scalar(a, b, c)
    ks_tp = torch.arange(0, min(n1, n2, n3) + 1, dtype=torch.float64)
    expected_tp = torch.sum(
        _binomial_pmf(ks_tp, n=n1, p=p1) * _binomial_pmf(ks_tp, n=n2, p=p2) * _binomial_pmf(ks_tp, n=n3, p=p3)
    )
    torch.testing.assert_close(tp, expected_tp.to(dtype=tp.dtype, device=tp.device), rtol=1e-6, atol=1e-9)


def _hypergeo_pmf(*, k: int, K: int, N: int, n: int) -> float:
    return (math.comb(K, k) * math.comb(N - K, n - k)) / math.comb(N, n)


def test_hypergeometric_inner_and_triple_matches_exact_enumeration():
    # The closed-form overlap used by the implementation assumes a shared population size N.
    N = 12
    params = [(5, 4), (7, 6), (4, 5)]  # (K, n)

    leaves: list[Hypergeometric] = []
    for K, n in params:
        leaves.append(
            Hypergeometric(
                scope=Scope([0]),
                out_channels=1,
                num_repetitions=1,
                K=torch.tensor([[[float(K)]]]),
                N=torch.tensor([[[float(N)]]]),
                n=torch.tensor([[[float(n)]]]),
            )
        )

    a, b, c = leaves

    ip = inner_product_matrix(a, b)[0, 0, 0, 0]
    min_k = max(0, params[0][1] + params[0][0] - N, params[1][1] + params[1][0] - N)
    max_k = min(params[0][1], params[0][0], params[1][1], params[1][0])
    expected_ip = sum(
        _hypergeo_pmf(k=k, K=params[0][0], N=N, n=params[0][1])
        * _hypergeo_pmf(k=k, K=params[1][0], N=N, n=params[1][1])
        for k in range(min_k, max_k + 1)
    )
    torch.testing.assert_close(
        ip, torch.tensor(expected_ip, dtype=ip.dtype, device=ip.device), rtol=1e-12, atol=0.0
    )

    tp = triple_product_scalar(a, b, c)
    min_k = max(
        0,
        params[0][1] + params[0][0] - N,
        params[1][1] + params[1][0] - N,
        params[2][1] + params[2][0] - N,
    )
    max_k = min(
        params[0][1],
        params[0][0],
        params[1][1],
        params[1][0],
        params[2][1],
        params[2][0],
    )
    expected_tp = sum(
        _hypergeo_pmf(k=k, K=params[0][0], N=N, n=params[0][1])
        * _hypergeo_pmf(k=k, K=params[1][0], N=N, n=params[1][1])
        * _hypergeo_pmf(k=k, K=params[2][0], N=N, n=params[2][1])
        for k in range(min_k, max_k + 1)
    )
    torch.testing.assert_close(
        tp, torch.tensor(expected_tp, dtype=tp.dtype, device=tp.device), rtol=1e-12, atol=0.0
    )


def _neg_binom_pmf(ks: torch.Tensor, *, r: int, p: float) -> torch.Tensor:
    dist = torch.distributions.NegativeBinomial(
        total_count=torch.tensor(float(r), dtype=torch.float64),
        probs=torch.tensor(float(p), dtype=torch.float64),
    )
    return torch.exp(dist.log_prob(ks.to(dtype=torch.float64)))


def test_negative_binomial_inner_and_triple_matches_truncated_enumeration():
    # High success probabilities push tail mass down so finite truncation is a trustworthy oracle.
    r1, p1 = 2, 0.8
    r2, p2 = 3, 0.7
    r3, p3 = 4, 0.85
    k_max = 220

    a = NegativeBinomial(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        total_count=torch.tensor([[[float(r1)]]]),
        probs=torch.tensor([[[p1]]]),
    )
    b = NegativeBinomial(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        total_count=torch.tensor([[[float(r2)]]]),
        probs=torch.tensor([[[p2]]]),
    )
    c = NegativeBinomial(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        total_count=torch.tensor([[[float(r3)]]]),
        probs=torch.tensor([[[p3]]]),
    )

    ks = torch.arange(0, k_max + 1, dtype=torch.float64)
    pmf1 = _neg_binom_pmf(ks, r=r1, p=p1)
    pmf2 = _neg_binom_pmf(ks, r=r2, p=p2)
    pmf3 = _neg_binom_pmf(ks, r=r3, p=p3)

    expected_ip = torch.sum(pmf1 * pmf2)
    ip = inner_product_matrix(a, b)[0, 0, 0, 0]
    torch.testing.assert_close(ip, expected_ip.to(dtype=ip.dtype, device=ip.device), rtol=2e-8, atol=1e-9)

    expected_tp = torch.sum(pmf1 * pmf2 * pmf3)
    tp = triple_product_scalar(a, b, c)
    torch.testing.assert_close(tp, expected_tp.to(dtype=tp.dtype, device=tp.device), rtol=2e-8, atol=1e-9)


def test_histogram_inner_and_triple_matches_closed_form_same_bins():
    edges = torch.tensor([0.0, 1.0, 3.0, 4.0], dtype=torch.float32)
    widths = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)

    p1 = torch.tensor([[[[0.2, 0.3, 0.5]]]], dtype=torch.float32)
    p2 = torch.tensor([[[[0.1, 0.8, 0.1]]]], dtype=torch.float32)
    p3 = torch.tensor([[[[0.25, 0.25, 0.5]]]], dtype=torch.float32)

    a = Histogram(scope=Scope([0]), out_channels=1, num_repetitions=1, bin_edges=edges, probs=p1)
    b = Histogram(scope=Scope([0]), out_channels=1, num_repetitions=1, bin_edges=edges, probs=p2)
    c = Histogram(scope=Scope([0]), out_channels=1, num_repetitions=1, bin_edges=edges, probs=p3)

    ip = inner_product_matrix(a, b)[0, 0, 0, 0]
    expected_ip = torch.sum((p1.view(-1).to(torch.float64) * p2.view(-1).to(torch.float64)) / widths)
    torch.testing.assert_close(ip, expected_ip.to(dtype=ip.dtype, device=ip.device), rtol=1e-8, atol=5e-9)

    tp = triple_product_scalar(a, b, c)
    expected_tp = torch.sum(
        (p1.view(-1).to(torch.float64) * p2.view(-1).to(torch.float64) * p3.view(-1).to(torch.float64))
        / widths.pow(2)
    )
    torch.testing.assert_close(tp, expected_tp.to(dtype=tp.dtype, device=tp.device), rtol=1e-8, atol=5e-9)


def _make_piecewise_linear(
    *, xs: torch.Tensor, ys: torch.Tensor, scope: Scope, domain: Domain, out_channels: int = 1
) -> PiecewiseLinear:
    leaf = PiecewiseLinear(scope=scope, out_channels=out_channels, num_repetitions=1)

    # Build internals in the exact nested layout expected by PiecewiseLinear internals.
    leaf.xs = [[[[xs]]]]  # type: ignore[assignment]
    leaf.ys = [[[[ys]]]]  # type: ignore[assignment]
    leaf.domains = [domain]  # type: ignore[assignment]
    leaf.is_initialized = True
    return leaf


def test_piecewise_linear_inner_and_triple_matches_dense_trapezoid():
    scope = Scope([0])
    dom = Domain.continuous_range(0.0, 2.0)

    # Asymmetric supports/slopes make interpolation integration regressions easier to surface.
    xa = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    ya = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)  # Unit-mass baseline profile.
    xb = torch.tensor([0.0, 0.5, 1.5, 2.0], dtype=torch.float64)
    yb = torch.tensor([0.0, 0.6, 0.6, 0.0], dtype=torch.float64)  # Flat plateau stresses overlap handling.
    xc = torch.tensor([0.0, 0.7, 2.0], dtype=torch.float64)
    yc = torch.tensor([0.0, 0.9, 0.0], dtype=torch.float64)

    a = _make_piecewise_linear(xs=xa, ys=ya, scope=scope, domain=dom)
    b = _make_piecewise_linear(xs=xb, ys=yb, scope=scope, domain=dom)
    c = _make_piecewise_linear(xs=xc, ys=yc, scope=scope, domain=dom)

    ip = inner_product_matrix(a, b)[0, 0, 0, 0]
    tp = triple_product_scalar(a, b, c)

    grid = torch.linspace(0.0, 2.0, steps=200_001, dtype=torch.float64)
    fa = interp(grid, xa, ya, extrapolate="constant")
    fb = interp(grid, xb, yb, extrapolate="constant")
    fc = interp(grid, xc, yc, extrapolate="constant")
    expected_ip = torch.trapz(fa * fb, grid)
    expected_tp = torch.trapz(fa * fb * fc, grid)

    torch.testing.assert_close(ip, expected_ip.to(dtype=ip.dtype, device=ip.device), rtol=3e-4, atol=2e-5)
    torch.testing.assert_close(tp, expected_tp.to(dtype=tp.dtype, device=tp.device), rtol=3e-4, atol=2e-5)
