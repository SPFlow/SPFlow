"""Shared inner-/triple-product utilities for probabilistic circuits.

This module implements the dynamic programs used by SOS/SOCS normalization:

- Pairwise inner products:  ∫ a(x) b(x) dx
- Triple products:          ∫ a(x) b(x) c(x) dx

Both are computed bottom-up using circuit structure (Cat/Product/(Signed)Sum)
and analytic leaf integrals when available.

Two wrappers use this core:
- `spflow.utils.inner_product` (for `spflow.modules.sos`)
- `spflow.utils.inner_product` (canonical SOS/SOCS entry point)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from einops import rearrange
from torch import Tensor

from spflow.exceptions import ShapeError, UnsupportedOperationError
from spflow.modules.leaves.bernoulli import Bernoulli
from spflow.modules.leaves.binomial import Binomial
from spflow.modules.leaves.categorical import Categorical
from spflow.modules.leaves.cltree import CLTree
from spflow.modules.leaves.exponential import Exponential
from spflow.modules.leaves.gamma import Gamma
from spflow.modules.leaves.geometric import Geometric
from spflow.modules.leaves.histogram import Histogram
from spflow.modules.leaves.hypergeometric import Hypergeometric
from spflow.modules.leaves.laplace import Laplace
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.leaves.log_normal import LogNormal
from spflow.modules.leaves.negative_binomial import NegativeBinomial
from spflow.modules.leaves.normal import Normal
from spflow.modules.leaves.piecewise_linear import PiecewiseLinear
from spflow.modules.leaves.poisson import Poisson
from spflow.modules.leaves.uniform import Uniform
from spflow.modules.module import Module
from spflow.modules.ops.cat import Cat
from spflow.modules.products.product import Product
from spflow.modules.sums.sum import Sum
from spflow.utils.cache import Cache
from spflow.utils.domain import DataType


def _ensure_same_scope(a: Module, b: Module) -> None:
    if a.scope != b.scope:
        raise ShapeError(f"Scopes must match: {a.scope} vs {b.scope}.")


def _leaf_event_shape_ok(a: LeafModule, b: LeafModule) -> None:
    if a.out_shape.features != b.out_shape.features:
        raise ShapeError("Leaf features must match for inner product.")
    if a.out_shape.repetitions != b.out_shape.repetitions:
        raise ShapeError("Leaf repetitions must match for inner product.")


def _binomial_logpmf(k: Tensor, n: Tensor, p: Tensor) -> Tensor:
    # k, n, p are broadcastable tensors (float64). Masking for k outside [0,n] handled by caller.
    # log C(n,k) + k log p + (n-k) log(1-p)
    logc = torch.lgamma(n + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(n - k + 1.0)
    return logc + k * torch.log(p.clamp_min(1e-30)) + (n - k) * torch.log((1.0 - p).clamp_min(1e-30))


def _hypergeo_logpmf(k: Tensor, K: Tensor, N: Tensor, n: Tensor) -> Tensor:
    # log [C(K,k) C(N-K, n-k) / C(N,n)]
    log_c_K_k = torch.lgamma(K + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(K - k + 1.0)
    NK = N - K
    nk = n - k
    log_c_NK_nk = torch.lgamma(NK + 1.0) - torch.lgamma(nk + 1.0) - torch.lgamma(NK - nk + 1.0)
    log_c_N_n = torch.lgamma(N + 1.0) - torch.lgamma(n + 1.0) - torch.lgamma(N - n + 1.0)
    return log_c_K_k + log_c_NK_nk - log_c_N_n


def _neg_binom_logpmf(k: Tensor, r: Tensor, p: Tensor) -> Tensor:
    # Torch NegativeBinomial: number of successes k >= 0 before total_count=r failures, probs=p:
    # pmf = C(k+r-1,k) p^k (1-p)^r
    return (
        torch.lgamma(k + r)
        - torch.lgamma(k + 1.0)
        - torch.lgamma(r)
        + k * torch.log(p.clamp_min(1e-30))
        + r * torch.log((1.0 - p).clamp_min(1e-30))
    )


def _series_logsumexp(
    *,
    log_terms_fn: callable,
    max_k: int,
    tol: float,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    # Generic positive-series accumulator in log-space. log_terms_fn(k) -> log term tensor.
    logS = torch.full((), float("-inf"), dtype=dtype, device=device)
    for k in range(max_k + 1):
        lt = log_terms_fn(k)
        logS = torch.logaddexp(logS, lt)
        if k >= 32:
            # Relative contribution bound: exp(lt-logS) < tol
            if torch.all((lt - logS) < torch.log(torch.tensor(tol, dtype=dtype, device=device))):
                break
    return logS


def leaf_inner_product(a: Module, b: Module) -> Tensor:
    """Compute per-feature/channel inner products ∫ f_a(x) f_b(x) dx for leaves."""
    _ensure_same_scope(a, b)
    _leaf_event_shape_ok(a, b)
    try:
        from spflow.zoo.sos.signed_categorical import SignedCategorical as _SignedCategorical
    except Exception:  # pragma: no cover - optional zoo import
        _SignedCategorical = None  # type: ignore[assignment]

    if isinstance(a, Normal) and isinstance(b, Normal):
        mu1 = rearrange(a.loc.to(dtype=torch.float64), "f ci r -> f ci 1 r")
        mu2 = rearrange(b.loc.to(dtype=torch.float64), "f co r -> f 1 co r")
        s1 = rearrange(a.scale.to(dtype=torch.float64), "f ci r -> f ci 1 r")
        s2 = rearrange(b.scale.to(dtype=torch.float64), "f co r -> f 1 co r")
        var = s1.pow(2) + s2.pow(2)
        log_coeff = -0.5 * torch.log(2.0 * torch.pi * var)
        quad = -(mu1 - mu2).pow(2) / (2.0 * var)
        return torch.exp(log_coeff + quad)

    if isinstance(a, Bernoulli) and isinstance(b, Bernoulli):
        p1 = rearrange(a.probs.to(dtype=torch.float64), "f ci r -> f ci 1 r")
        p2 = rearrange(b.probs.to(dtype=torch.float64), "f co r -> f 1 co r")
        return p1 * p2 + (1.0 - p1) * (1.0 - p2)

    if isinstance(a, Categorical) and isinstance(b, Categorical):
        if a.K != b.K:
            raise ShapeError(f"Categorical K mismatch: {a.K} vs {b.K}.")
        p1 = rearrange(a.probs.to(dtype=torch.float64), "f ci r k -> f ci 1 r k")
        p2 = rearrange(b.probs.to(dtype=torch.float64), "f co r k -> f 1 co r k")
        return torch.sum(p1 * p2, dim=-1)

    if _SignedCategorical is not None:
        if isinstance(a, _SignedCategorical) and isinstance(b, _SignedCategorical):
            if a.K != b.K:
                raise ShapeError(f"SignedCategorical K mismatch: {a.K} vs {b.K}.")
            w1 = rearrange(a.weights.to(dtype=torch.float64), "f ci r k -> f ci 1 r k")
            w2 = rearrange(b.weights.to(dtype=torch.float64), "f co r k -> f 1 co r k")
            return torch.sum(w1 * w2, dim=-1)

        if isinstance(a, _SignedCategorical) and isinstance(b, Categorical):
            if a.K != b.K:
                raise ShapeError(f"Categorical K mismatch: {a.K} vs {b.K}.")
            w1 = rearrange(a.weights.to(dtype=torch.float64), "f ci r k -> f ci 1 r k")
            p2 = rearrange(b.probs.to(dtype=torch.float64), "f co r k -> f 1 co r k")
            return torch.sum(w1 * p2, dim=-1)

        if isinstance(a, Categorical) and isinstance(b, _SignedCategorical):
            if a.K != b.K:
                raise ShapeError(f"Categorical K mismatch: {a.K} vs {b.K}.")
            p1 = rearrange(a.probs.to(dtype=torch.float64), "f ci r k -> f ci 1 r k")
            w2 = rearrange(b.weights.to(dtype=torch.float64), "f co r k -> f 1 co r k")
            return torch.sum(p1 * w2, dim=-1)

    if isinstance(a, Exponential) and isinstance(b, Exponential):
        r1 = rearrange(a.rate.to(dtype=torch.float64), "f ci r -> f ci 1 r")
        r2 = rearrange(b.rate.to(dtype=torch.float64), "f co r -> f 1 co r")
        return (r1 * r2) / (r1 + r2).clamp_min(1e-30)

    if isinstance(a, Laplace) and isinstance(b, Laplace):
        mu1 = rearrange(a.loc.to(dtype=torch.float64), "f ci r -> f ci 1 r")
        mu2 = rearrange(b.loc.to(dtype=torch.float64), "f co r -> f 1 co r")
        b1 = rearrange(a.scale.to(dtype=torch.float64), "f ci r -> f ci 1 r").clamp_min(1e-30)
        b2 = rearrange(b.scale.to(dtype=torch.float64), "f co r -> f 1 co r").clamp_min(1e-30)
        d = torch.abs(mu1 - mu2)
        exp1 = torch.exp(-d / b1)
        exp2 = torch.exp(-d / b2)
        term_tails = (exp1 + exp2) / (4.0 * (b1 + b2))
        same = torch.isclose(b1, b2)
        term_mid = (exp1 - exp2) / (4.0 * (b1 - b2))
        term_mid_same = torch.exp(-d / b1) * d / (4.0 * b1.pow(2))
        return torch.where(same, term_tails + term_mid_same, term_tails + term_mid)

    if isinstance(a, LogNormal) and isinstance(b, LogNormal):
        mu1 = rearrange(a.loc.to(dtype=torch.float64), "f ci r -> f ci 1 r")
        mu2 = rearrange(b.loc.to(dtype=torch.float64), "f co r -> f 1 co r")
        s1 = rearrange(a.scale.to(dtype=torch.float64), "f ci r -> f ci 1 r").clamp_min(1e-30)
        s2 = rearrange(b.scale.to(dtype=torch.float64), "f co r -> f 1 co r").clamp_min(1e-30)
        a1 = 1.0 / s1.pow(2)
        a2 = 1.0 / s2.pow(2)
        A = a1 + a2
        D = (a1 * mu1 + a2 * mu2) - 1.0
        E = -0.5 * (a1 * mu1.pow(2) + a2 * mu2.pow(2))
        log_pref = -0.5 * torch.log(2.0 * torch.pi * (s1.pow(2) + s2.pow(2)))
        return torch.exp(log_pref + E + (D.pow(2) / (2.0 * A)))

    if isinstance(a, Poisson) and isinstance(b, Poisson):
        l1 = rearrange(a.rate.to(dtype=torch.float64), "f ci r -> f ci 1 r").clamp_min(0.0)
        l2 = rearrange(b.rate.to(dtype=torch.float64), "f co r -> f 1 co r").clamp_min(0.0)
        z = 2.0 * torch.sqrt((l1 * l2).clamp_min(0.0))
        i0 = getattr(torch, "i0", torch.special.i0)
        return torch.exp(-(l1 + l2)) * i0(z)

    if isinstance(a, Gamma) and isinstance(b, Gamma):
        a1 = rearrange(a.concentration.to(dtype=torch.float64), "f ci r -> f ci 1 r").clamp_min(1e-30)
        a2 = rearrange(b.concentration.to(dtype=torch.float64), "f co r -> f 1 co r").clamp_min(1e-30)
        b1 = rearrange(a.rate.to(dtype=torch.float64), "f ci r -> f ci 1 r").clamp_min(1e-30)
        b2 = rearrange(b.rate.to(dtype=torch.float64), "f co r -> f 1 co r").clamp_min(1e-30)
        s = a1 + a2 - 1.0
        if (s <= 0.0).any():
            raise UnsupportedOperationError(
                "Gamma inner product requires concentration_a + concentration_b > 1 for integrability."
            )
        log_ip = (
            a1 * torch.log(b1)
            + a2 * torch.log(b2)
            + torch.lgamma(s)
            - torch.lgamma(a1)
            - torch.lgamma(a2)
            - s * torch.log(b1 + b2)
        )
        return torch.exp(log_ip)

    if isinstance(a, Uniform) and isinstance(b, Uniform):
        a1 = rearrange(a.low.to(dtype=torch.float64), "f ci r -> f ci 1 r")
        b1 = rearrange(a.high.to(dtype=torch.float64), "f ci r -> f ci 1 r")
        a2 = rearrange(b.low.to(dtype=torch.float64), "f co r -> f 1 co r")
        b2 = rearrange(b.high.to(dtype=torch.float64), "f co r -> f 1 co r")
        len1 = (b1 - a1).clamp_min(1e-30)
        len2 = (b2 - a2).clamp_min(1e-30)
        left = torch.maximum(a1, a2)
        right = torch.minimum(b1, b2)
        overlap = (right - left).clamp_min(0.0)
        return overlap / (len1 * len2)

    if isinstance(a, Geometric) and isinstance(b, Geometric):
        p1 = rearrange(a.probs.to(dtype=torch.float64), "f ci r -> f ci 1 r").clamp_min(0.0).clamp_max(1.0)
        p2 = rearrange(b.probs.to(dtype=torch.float64), "f co r -> f 1 co r").clamp_min(0.0).clamp_max(1.0)
        q1 = 1.0 - p1
        q2 = 1.0 - p2
        denom = 1.0 - (q1 * q2)
        return (p1 * p2) / denom.clamp_min(1e-30)

    if isinstance(a, Binomial) and isinstance(b, Binomial):
        n1 = rearrange(a.total_count.to(dtype=torch.float64), "f ci r -> f ci 1 r")
        n2 = rearrange(b.total_count.to(dtype=torch.float64), "f co r -> f 1 co r")
        p1 = rearrange(a.probs.to(dtype=torch.float64), "f ci r -> f ci 1 r")
        p2 = rearrange(b.probs.to(dtype=torch.float64), "f co r -> f 1 co r")
        max_n = int(torch.max(torch.maximum(n1, n2)).item())
        ks = rearrange(torch.arange(0, max_n + 1, dtype=torch.float64, device=p1.device), "k -> k 1 1 1 1")
        n1b = rearrange(n1, "f ci co r -> 1 f ci co r")
        n2b = rearrange(n2, "f ci co r -> 1 f ci co r")
        lp1 = _binomial_logpmf(ks, n1b, rearrange(p1, "f ci co r -> 1 f ci co r"))
        lp2 = _binomial_logpmf(ks, n2b, rearrange(p2, "f ci co r -> 1 f ci co r"))
        mask = (ks <= n1b) & (ks <= n2b)
        lsum = torch.logsumexp(torch.where(mask, lp1 + lp2, torch.full_like(lp1, float("-inf"))), dim=0)
        return torch.exp(lsum)

    if isinstance(a, Hypergeometric) and isinstance(b, Hypergeometric):
        K1 = rearrange(a.K.to(dtype=torch.float64), "f ci r -> f ci 1 r")
        N1 = rearrange(a.N.to(dtype=torch.float64), "f ci r -> f ci 1 r")
        n1 = rearrange(a.n.to(dtype=torch.float64), "f ci r -> f ci 1 r")
        K2 = rearrange(b.K.to(dtype=torch.float64), "f co r -> f 1 co r")
        N2 = rearrange(b.N.to(dtype=torch.float64), "f co r -> f 1 co r")
        n2 = rearrange(b.n.to(dtype=torch.float64), "f co r -> f 1 co r")
        if not torch.allclose(N1, N2):
            raise ShapeError("Hypergeometric inner product requires matching N (population size).")
        N = N1
        max_k = int(torch.max(torch.minimum(torch.minimum(n1, K1), torch.minimum(n2, K2))).item())
        ks = rearrange(torch.arange(0, max_k + 1, dtype=torch.float64, device=N.device), "k -> k 1 1 1 1")
        K1b = rearrange(K1, "f ci co r -> 1 f ci co r")
        K2b = rearrange(K2, "f ci co r -> 1 f ci co r")
        Nb = rearrange(N, "f ci co r -> 1 f ci co r")
        n1b = rearrange(n1, "f ci co r -> 1 f ci co r")
        n2b = rearrange(n2, "f ci co r -> 1 f ci co r")
        lp1 = _hypergeo_logpmf(ks, K1b, Nb, n1b)
        lp2 = _hypergeo_logpmf(ks, K2b, Nb, n2b)
        min1 = rearrange(torch.maximum(torch.zeros_like(N), n1 + K1 - N), "f ci co r -> 1 f ci co r")
        max1 = rearrange(torch.minimum(n1, K1), "f ci co r -> 1 f ci co r")
        min2 = rearrange(torch.maximum(torch.zeros_like(N), n2 + K2 - N), "f ci co r -> 1 f ci co r")
        max2 = rearrange(torch.minimum(n2, K2), "f ci co r -> 1 f ci co r")
        mask = (ks >= min1) & (ks <= max1) & (ks >= min2) & (ks <= max2)
        lsum = torch.logsumexp(torch.where(mask, lp1 + lp2, torch.full_like(lp1, float("-inf"))), dim=0)
        return torch.exp(lsum)

    if isinstance(a, NegativeBinomial) and isinstance(b, NegativeBinomial):
        r1 = rearrange(a.total_count.to(dtype=torch.float64), "f ci r -> f ci 1 r")
        r2 = rearrange(b.total_count.to(dtype=torch.float64), "f co r -> f 1 co r")
        p1 = rearrange(a.probs.to(dtype=torch.float64), "f ci r -> f ci 1 r").clamp_min(1e-30).clamp_max(1.0)
        p2 = rearrange(b.probs.to(dtype=torch.float64), "f co r -> f 1 co r").clamp_min(1e-30).clamp_max(1.0)
        q = p1 * p2

        def log_term(k: int) -> Tensor:
            kk = torch.tensor(float(k), dtype=torch.float64, device=q.device)
            # (r1)_k (r2)_k / (k!)^2 * (p1 p2)^k * (1-p1)^r1 (1-p2)^r2
            lt = (
                torch.lgamma(r1 + kk)
                - torch.lgamma(r1)
                + torch.lgamma(r2 + kk)
                - torch.lgamma(r2)
                - 2.0 * torch.lgamma(kk + 1.0)
                + kk * torch.log(q.clamp_min(1e-30))
            )
            const = r1 * torch.log((1.0 - p1).clamp_min(1e-30)) + r2 * torch.log((1.0 - p2).clamp_min(1e-30))
            return lt + const

        logS = _series_logsumexp(log_terms_fn=log_term, max_k=4096, tol=1e-12, device=q.device)
        return torch.exp(logS)

    if isinstance(a, Histogram) and isinstance(b, Histogram):
        # Univariate leaf: per-feature inner product is computed independently.
        edges1 = a.bin_edges.to(dtype=torch.float64, device=a.device)
        edges2 = b.bin_edges.to(dtype=torch.float64, device=b.device)
        u_edges = torch.unique(torch.cat([edges1, edges2])).to(dtype=torch.float64)
        u_edges, _ = torch.sort(u_edges)
        seg_left = u_edges[:-1]
        seg_right = u_edges[1:]
        seg_len = (seg_right - seg_left).clamp_min(0.0)
        mids = (seg_left + seg_right) / 2.0

        widths1 = (edges1[1:] - edges1[:-1]).to(dtype=torch.float64)
        widths2 = (edges2[1:] - edges2[:-1]).to(dtype=torch.float64)
        dens1 = rearrange(
            a.probs.to(dtype=torch.float64) / rearrange(widths1, "k -> 1 1 1 k"),
            "f ci r b1 -> f ci 1 r b1",
        )
        dens2 = rearrange(
            b.probs.to(dtype=torch.float64) / rearrange(widths2, "k -> 1 1 1 k"),
            "f co r b2 -> f 1 co r b2",
        )

        idx1 = (torch.bucketize(mids, edges1, right=True) - 1).clamp(0, widths1.numel() - 1)
        idx2 = (torch.bucketize(mids, edges2, right=True) - 1).clamp(0, widths2.numel() - 1)
        in1 = (mids >= edges1[0]) & (mids < edges1[-1])
        in2 = (mids >= edges2[0]) & (mids < edges2[-1])
        mask = (in1 & in2).to(dtype=torch.float64)

        d1 = dens1.index_select(-1, idx1).squeeze(-1)  # (F,Ca,1,R,S)
        d2 = dens2.index_select(-1, idx2).squeeze(-1)  # (F,1,Cb,R,S)
        prod = d1 * d2  # (F,Ca,Cb,R,S)
        out = torch.sum(prod * rearrange(seg_len * mask, "s -> 1 1 1 1 s"), dim=-1)
        return out

    if isinstance(a, PiecewiseLinear) and isinstance(b, PiecewiseLinear):
        if not a.is_initialized or not b.is_initialized:
            raise UnsupportedOperationError(
                "PiecewiseLinear inner product requires both leaves to be initialized."
            )
        if a.domains is None or b.domains is None:
            raise UnsupportedOperationError("PiecewiseLinear inner product requires domains.")

        # Only support continuous domains for now.
        for dom in a.domains:
            if dom.data_type != DataType.CONTINUOUS:
                raise UnsupportedOperationError(
                    "PiecewiseLinear inner product currently supports continuous domains only."
                )

        dist_a = a.distribution()
        dist_b = b.distribution()
        F, Ca, Cb, R = (
            a.out_shape.features,
            a.out_shape.channels,
            b.out_shape.channels,
            a.out_shape.repetitions,
        )
        out = torch.empty((F, Ca, Cb, R), dtype=torch.float64, device=a.device)

        def _get_knots(dist, r: int, leaf_idx: int, f: int) -> tuple[Tensor, Tensor]:
            xs = dist.xs[r][leaf_idx][f][0]
            ys = dist.ys[r][leaf_idx][f][0]
            return xs.to(dtype=torch.float64), ys.to(dtype=torch.float64)

        for r in range(R):
            for ca in range(Ca):
                for cb in range(Cb):
                    for f in range(F):
                        xa, ya = _get_knots(dist_a, r, ca, f)
                        xb, yb = _get_knots(dist_b, r, cb, f)
                        grid = torch.unique(torch.cat([xa, xb]))
                        grid, _ = torch.sort(grid)
                        if grid.numel() < 2:
                            out[f, ca, cb, r] = 0.0
                            continue
                        # Evaluate at grid points using the leaf's interpolation helper.
                        from spflow.modules.leaves.piecewise_linear import interp  # local import

                        fa = interp(grid, xa, ya, extrapolate="constant")
                        fb = interp(grid, xb, yb, extrapolate="constant")
                        h = (grid[1:] - grid[:-1]).clamp_min(0.0)
                        f0, f1 = fa[:-1], fa[1:]
                        g0, g1 = fb[:-1], fb[1:]
                        integral = torch.sum(h / 6.0 * (2 * f0 * g0 + f0 * g1 + f1 * g0 + 2 * f1 * g1))
                        out[f, ca, cb, r] = integral
        return out

    if isinstance(a, CLTree) and isinstance(b, CLTree):
        if a.K != b.K:
            raise ShapeError(f"CLTree K mismatch: {a.K} vs {b.K}.")
        if not torch.equal(a.parents, b.parents):
            raise UnsupportedOperationError(
                "CLTree inner product requires identical tree structure (parents)."
            )

        parents = a.parents.tolist()
        root = parents.index(-1)
        children: list[list[int]] = [[] for _ in range(a.out_shape.features)]
        for child, parent in enumerate(parents):
            if parent == -1:
                continue
            children[parent].append(child)

        log_cpt_a = a.log_cpt.to(dtype=torch.float64)
        log_cpt_b = b.log_cpt.to(dtype=torch.float64)

        C1, C2 = a.out_shape.channels, b.out_shape.channels
        R = a.out_shape.repetitions
        K = a.K
        F = a.out_shape.features

        pa_root = torch.exp(log_cpt_a[root, :, :, :, 0])
        pb_root = torch.exp(log_cpt_b[root, :, :, :, 0])

        msg = torch.ones((F, C1, C2, R, K), dtype=torch.float64, device=log_cpt_a.device)

        post_order = a.post_order.tolist()
        for i in post_order:
            p = parents[i]
            if p == -1:
                continue

            prod_child = torch.ones((C1, C2, R, K), dtype=torch.float64, device=log_cpt_a.device)
            for ch in children[i]:
                prod_child = prod_child * msg[ch]

            pa = torch.exp(log_cpt_a[i])
            pb = torch.exp(log_cpt_b[i])
            phi = rearrange(pa, "ca r i o -> ca 1 r i o") * rearrange(pb, "cb r i o -> 1 cb r i o")
            msg_i = torch.einsum("abri,abrio->abro", prod_child, phi)
            msg[i] = msg_i

        prod_root = torch.ones((C1, C2, R, K), dtype=torch.float64, device=log_cpt_a.device)
        for ch in children[root]:
            prod_root = prod_root * msg[ch]

        phi_root = rearrange(pa_root, "ca r i -> ca 1 r i") * rearrange(pb_root, "cb r i -> 1 cb r i")
        z = torch.sum(phi_root * prod_root, dim=-1)

        out = torch.ones((F, C1, C2, R), dtype=torch.float64, device=log_cpt_a.device)
        out[0] = z
        return out

    raise UnsupportedOperationError(
        f"Leaf inner product not implemented for {type(a).__name__} × {type(b).__name__}. "
        "Supported: Normal, Bernoulli, Categorical, Exponential, Laplace, LogNormal, Poisson, Gamma, "
        "Uniform, Geometric, Binomial, Hypergeometric, NegativeBinomial, Histogram, PiecewiseLinear, CLTree."
    )


def _get_pair_memo(cache: Cache, *, memo_key: str) -> dict[tuple[int, int], Tensor]:
    memo = cache.extras.get(memo_key)
    if memo is None:
        memo = {}
        cache.extras[memo_key] = memo
    return cast(dict[tuple[int, int], Tensor], memo)


def inner_product_matrix(
    a: Module,
    b: Module,
    *,
    cache: Cache | None = None,
    signed_sum_types: Sequence[type[Module]] = (),
    memo_key: str = "_inner_product_memo",
) -> Tensor:
    if cache is not None:
        memo = _get_pair_memo(cache, memo_key=memo_key)
        key = (id(a), id(b))
        cached = memo.get(key)
        if cached is not None:
            return cached
        rev = memo.get((id(b), id(a)))
        if rev is not None:
            out = rearrange(rev, "f ci co r -> f co ci r").contiguous()
            memo[key] = out
            return out

    _ensure_same_scope(a, b)
    if a.out_shape.features != b.out_shape.features:
        raise ShapeError(f"Feature mismatch: {a.out_shape.features} vs {b.out_shape.features}.")
    if a.out_shape.repetitions != b.out_shape.repetitions:
        raise ShapeError("Repetition mismatch for inner product.")

    try:
        from spflow.zoo.sos.signed_categorical import SignedCategorical as _SignedCategorical
    except Exception:  # pragma: no cover - optional zoo import
        _SignedCategorical = None  # type: ignore[assignment]

    a_is_leaf_like = isinstance(a, LeafModule) or (
        _SignedCategorical is not None and isinstance(a, _SignedCategorical)
    )
    b_is_leaf_like = isinstance(b, LeafModule) or (
        _SignedCategorical is not None and isinstance(b, _SignedCategorical)
    )

    if a_is_leaf_like and b_is_leaf_like:
        out = leaf_inner_product(a, b)
        if cache is not None:
            memo[(id(a), id(b))] = out
        return out

    if isinstance(a, Cat) and isinstance(b, Cat):
        if a.dim != b.dim:
            raise ShapeError("Cat dim mismatch for inner product.")

        if a.dim == 1:
            if len(a.inputs) != len(b.inputs):
                raise ShapeError("Cat arity mismatch for inner product.")
            parts = [
                inner_product_matrix(
                    cast(Module, ai), cast(Module, bi), cache=cache, signed_sum_types=signed_sum_types
                )
                for ai, bi in zip(a.inputs, b.inputs)
            ]
            out = torch.cat(parts, dim=0)
            if cache is not None:
                memo[(id(a), id(b))] = out
            return out

        if a.dim == 2:
            F = a.out_shape.features
            R = a.out_shape.repetitions
            Ca = sum(cast(Module, ai).out_shape.channels for ai in a.inputs)
            Cb = sum(cast(Module, bi).out_shape.channels for bi in b.inputs)

            blocks: list[list[Tensor]] = []
            for ai in a.inputs:
                row: list[Tensor] = []
                for bi in b.inputs:
                    row.append(
                        inner_product_matrix(
                            cast(Module, ai),
                            cast(Module, bi),
                            cache=cache,
                            signed_sum_types=signed_sum_types,
                        )
                    )
                blocks.append(row)

            out = torch.empty((F, Ca, Cb, R), dtype=torch.float64, device=blocks[0][0].device)
            a_off = 0
            for i, ai in enumerate(a.inputs):
                ai_mod = cast(Module, ai)
                a_ch = ai_mod.out_shape.channels
                b_off = 0
                for j, bi in enumerate(b.inputs):
                    bi_mod = cast(Module, bi)
                    b_ch = bi_mod.out_shape.channels
                    out[:, a_off : a_off + a_ch, b_off : b_off + b_ch, :] = blocks[i][j]
                    b_off += b_ch
                a_off += a_ch

            if cache is not None:
                memo[(id(a), id(b))] = out
            return out

        raise UnsupportedOperationError(f"inner_product does not support Cat(dim={a.dim}).")

    if isinstance(a, Product) and isinstance(b, Product):
        child_k = inner_product_matrix(
            cast(Module, a.inputs), cast(Module, b.inputs), cache=cache, signed_sum_types=signed_sum_types
        )
        out = torch.prod(child_k, dim=0, keepdim=True)
        if cache is not None:
            memo[(id(a), id(b))] = out
        return out

    sum_types = (Sum, *signed_sum_types)
    if isinstance(a, sum_types) and isinstance(b, sum_types):
        child_k = inner_product_matrix(
            cast(Module, a.inputs), cast(Module, b.inputs), cache=cache, signed_sum_types=signed_sum_types
        )
        wa = a.weights.to(dtype=torch.float64)  # type: ignore[attr-defined]
        wb = b.weights.to(dtype=torch.float64)  # type: ignore[attr-defined]
        out = torch.einsum("fiar,fijr,fjbr->fabr", wa, child_k, wb)
        if cache is not None:
            memo[(id(a), id(b))] = out
        return out

    raise UnsupportedOperationError(
        f"inner_product_matrix not implemented for {type(a).__name__} × {type(b).__name__}."
    )


def log_self_inner_product_scalar(
    module: Module,
    *,
    cache: Cache | None = None,
    signed_sum_types: Sequence[type[Module]] = (),
    memo_key: str = "_inner_product_memo",
) -> Tensor:
    if tuple(module.out_shape) != (1, 1, 1):
        raise ShapeError(f"Expected scalar output (1,1,1), got {tuple(module.out_shape)}.")
    k = inner_product_matrix(
        module, module, cache=cache, signed_sum_types=signed_sum_types, memo_key=memo_key
    )
    z = torch.clamp(k[0, 0, 0, 0], min=0.0)
    return torch.log(z.clamp_min(1e-30))


def _get_triple_memo(cache: Cache, *, memo_key: str) -> dict[tuple[int, int, int], Tensor]:
    memo = cache.extras.get(memo_key)
    if memo is None:
        memo = {}
        cache.extras[memo_key] = memo
    return cast(dict[tuple[int, int, int], Tensor], memo)


def triple_product_tensor(
    a: Module,
    b: Module,
    c: Module,
    *,
    cache: Cache | None = None,
    signed_sum_types: Sequence[type[Module]] = (),
    memo_key: str = "_triple_product_memo",
) -> Tensor:
    if cache is not None:
        memo = _get_triple_memo(cache, memo_key=memo_key)
        key = (id(a), id(b), id(c))
        cached = memo.get(key)
        if cached is not None:
            return cached
        swapped = memo.get((id(b), id(a), id(c)))
        if swapped is not None:
            out = rearrange(swapped, "f ci co cj r -> f co ci cj r").contiguous()
            memo[key] = out
            return out

    _ensure_same_scope(a, b)
    _ensure_same_scope(a, c)
    if a.out_shape.features != b.out_shape.features or a.out_shape.features != c.out_shape.features:
        raise ShapeError("Feature mismatch for triple product.")
    if (
        a.out_shape.repetitions != b.out_shape.repetitions
        or a.out_shape.repetitions != c.out_shape.repetitions
    ):
        raise ShapeError("Repetition mismatch for triple product.")

    try:
        from spflow.zoo.sos.signed_categorical import SignedCategorical as _SignedCategorical
    except Exception:  # pragma: no cover - optional zoo import
        _SignedCategorical = None  # type: ignore[assignment]

    a_is_leaf_like = isinstance(a, LeafModule) or (
        _SignedCategorical is not None and isinstance(a, _SignedCategorical)
    )
    b_is_leaf_like = isinstance(b, LeafModule) or (
        _SignedCategorical is not None and isinstance(b, _SignedCategorical)
    )
    c_is_leaf_like = isinstance(c, LeafModule) or (
        _SignedCategorical is not None and isinstance(c, _SignedCategorical)
    )

    if a_is_leaf_like and b_is_leaf_like and c_is_leaf_like:
        # Handle a subset of leaf triples explicitly; otherwise reduce to finite sums/interval overlap where possible.
        if (
            _SignedCategorical is not None
            and isinstance(a, (Categorical, _SignedCategorical))
            and isinstance(b, (Categorical, _SignedCategorical))
            and isinstance(c, (Categorical, _SignedCategorical))
        ):
            Ks = (a.K, b.K, c.K)  # type: ignore[attr-defined]
            if len(set(Ks)) != 1:
                raise ShapeError(f"Categorical K mismatch for triple product: {Ks}.")

            def _cat_tensor(x: LeafModule) -> Tensor:
                if isinstance(x, Categorical):
                    return x.probs.to(dtype=torch.float64)
                return cast(_SignedCategorical, x).weights.to(dtype=torch.float64)

            p1 = rearrange(_cat_tensor(a), "f ci r k -> f ci 1 1 r k")
            p2 = rearrange(_cat_tensor(b), "f cj r k -> f 1 cj 1 r k")
            p3 = rearrange(_cat_tensor(c), "f ck r k -> f 1 1 ck r k")
            out = torch.sum(p1 * p2 * p3, dim=-1)
        elif isinstance(a, Normal) and isinstance(b, Normal) and isinstance(c, Normal):
            mu1 = rearrange(a.loc.to(dtype=torch.float64), "f ci r -> f ci 1 1 r")
            mu2 = rearrange(b.loc.to(dtype=torch.float64), "f cj r -> f 1 cj 1 r")
            mu3 = rearrange(c.loc.to(dtype=torch.float64), "f ck r -> f 1 1 ck r")
            s1 = rearrange(a.scale.to(dtype=torch.float64).clamp_min(1e-30), "f ci r -> f ci 1 1 r")
            s2 = rearrange(b.scale.to(dtype=torch.float64).clamp_min(1e-30), "f cj r -> f 1 cj 1 r")
            s3 = rearrange(c.scale.to(dtype=torch.float64).clamp_min(1e-30), "f ck r -> f 1 1 ck r")
            tau1 = 1.0 / s1.pow(2)
            tau2 = 1.0 / s2.pow(2)
            tau3 = 1.0 / s3.pow(2)
            tau = tau1 + tau2 + tau3
            m = (tau1 * mu1 + tau2 * mu2 + tau3 * mu3) / tau
            quad = (tau1 * mu1.pow(2) + tau2 * mu2.pow(2) + tau3 * mu3.pow(2)) - tau * m.pow(2)
            log_pref = -torch.log(torch.tensor(2.0 * torch.pi, dtype=torch.float64, device=mu1.device))
            log_pref = log_pref - (torch.log(s1) + torch.log(s2) + torch.log(s3)) - 0.5 * torch.log(tau)
            out = torch.exp(log_pref - 0.5 * quad)
        elif isinstance(a, Bernoulli) and isinstance(b, Bernoulli) and isinstance(c, Bernoulli):
            p1 = rearrange(a.probs.to(dtype=torch.float64), "f ci r -> f ci 1 1 r")
            p2 = rearrange(b.probs.to(dtype=torch.float64), "f cj r -> f 1 cj 1 r")
            p3 = rearrange(c.probs.to(dtype=torch.float64), "f ck r -> f 1 1 ck r")
            q1 = 1.0 - p1
            q2 = 1.0 - p2
            q3 = 1.0 - p3
            out = (q1 * q2 * q3) + (p1 * p2 * p3)
        elif isinstance(a, Categorical) and isinstance(b, Categorical) and isinstance(c, Categorical):
            if a.K != b.K or a.K != c.K:
                raise ShapeError("Categorical K mismatch for triple product.")
            p1 = rearrange(a.probs.to(dtype=torch.float64), "f ci r k -> f ci 1 1 r k")
            p2 = rearrange(b.probs.to(dtype=torch.float64), "f cj r k -> f 1 cj 1 r k")
            p3 = rearrange(c.probs.to(dtype=torch.float64), "f ck r k -> f 1 1 ck r k")
            out = torch.sum(p1 * p2 * p3, dim=-1)
        elif isinstance(a, Uniform) and isinstance(b, Uniform) and isinstance(c, Uniform):
            a1 = rearrange(a.low.to(dtype=torch.float64), "f ci r -> f ci 1 1 r")
            b1 = rearrange(a.high.to(dtype=torch.float64), "f ci r -> f ci 1 1 r")
            a2 = rearrange(b.low.to(dtype=torch.float64), "f cj r -> f 1 cj 1 r")
            b2 = rearrange(b.high.to(dtype=torch.float64), "f cj r -> f 1 cj 1 r")
            a3 = rearrange(c.low.to(dtype=torch.float64), "f ck r -> f 1 1 ck r")
            b3 = rearrange(c.high.to(dtype=torch.float64), "f ck r -> f 1 1 ck r")
            len1 = (b1 - a1).clamp_min(1e-30)
            len2 = (b2 - a2).clamp_min(1e-30)
            len3 = (b3 - a3).clamp_min(1e-30)
            left = torch.maximum(torch.maximum(a1, a2), a3)
            right = torch.minimum(torch.minimum(b1, b2), b3)
            overlap = (right - left).clamp_min(0.0)
            out = overlap / (len1 * len2 * len3)
        elif isinstance(a, Geometric) and isinstance(b, Geometric) and isinstance(c, Geometric):
            p1 = (
                rearrange(a.probs.to(dtype=torch.float64), "f ci r -> f ci 1 1 r")
                .clamp_min(0.0)
                .clamp_max(1.0)
            )
            p2 = (
                rearrange(b.probs.to(dtype=torch.float64), "f cj r -> f 1 cj 1 r")
                .clamp_min(0.0)
                .clamp_max(1.0)
            )
            p3 = (
                rearrange(c.probs.to(dtype=torch.float64), "f ck r -> f 1 1 ck r")
                .clamp_min(0.0)
                .clamp_max(1.0)
            )
            qprod = (1.0 - p1) * (1.0 - p2) * (1.0 - p3)
            out = (p1 * p2 * p3) / (1.0 - qprod).clamp_min(1e-30)
        elif isinstance(a, Binomial) and isinstance(b, Binomial) and isinstance(c, Binomial):
            n1 = rearrange(a.total_count.to(dtype=torch.float64), "f ci r -> f ci 1 1 r")
            n2 = rearrange(b.total_count.to(dtype=torch.float64), "f cj r -> f 1 cj 1 r")
            n3 = rearrange(c.total_count.to(dtype=torch.float64), "f ck r -> f 1 1 ck r")
            p1 = rearrange(a.probs.to(dtype=torch.float64), "f ci r -> f ci 1 1 r")
            p2 = rearrange(b.probs.to(dtype=torch.float64), "f cj r -> f 1 cj 1 r")
            p3 = rearrange(c.probs.to(dtype=torch.float64), "f ck r -> f 1 1 ck r")
            max_n = int(torch.max(torch.maximum(torch.maximum(n1, n2), n3)).item())
            ks = rearrange(
                torch.arange(0, max_n + 1, dtype=torch.float64, device=p1.device), "k -> k 1 1 1 1 1"
            )
            n1b = rearrange(n1, "f ci cj ck r -> 1 f ci cj ck r")
            n2b = rearrange(n2, "f ci cj ck r -> 1 f ci cj ck r")
            n3b = rearrange(n3, "f ci cj ck r -> 1 f ci cj ck r")
            lp1 = _binomial_logpmf(ks, n1b, rearrange(p1, "f ci cj ck r -> 1 f ci cj ck r"))
            lp2 = _binomial_logpmf(ks, n2b, rearrange(p2, "f ci cj ck r -> 1 f ci cj ck r"))
            lp3 = _binomial_logpmf(ks, n3b, rearrange(p3, "f ci cj ck r -> 1 f ci cj ck r"))
            mask = (ks <= n1b) & (ks <= n2b) & (ks <= n3b)
            lsum = torch.logsumexp(
                torch.where(mask, lp1 + lp2 + lp3, torch.full_like(lp1, float("-inf"))), dim=0
            )
            out = torch.exp(lsum)
        elif (
            isinstance(a, Hypergeometric) and isinstance(b, Hypergeometric) and isinstance(c, Hypergeometric)
        ):
            K1 = rearrange(a.K.to(dtype=torch.float64), "f ci r -> f ci 1 1 r")
            N1 = rearrange(a.N.to(dtype=torch.float64), "f ci r -> f ci 1 1 r")
            n1 = rearrange(a.n.to(dtype=torch.float64), "f ci r -> f ci 1 1 r")
            K2 = rearrange(b.K.to(dtype=torch.float64), "f cj r -> f 1 cj 1 r")
            N2 = rearrange(b.N.to(dtype=torch.float64), "f cj r -> f 1 cj 1 r")
            n2 = rearrange(b.n.to(dtype=torch.float64), "f cj r -> f 1 cj 1 r")
            K3 = rearrange(c.K.to(dtype=torch.float64), "f ck r -> f 1 1 ck r")
            N3 = rearrange(c.N.to(dtype=torch.float64), "f ck r -> f 1 1 ck r")
            n3 = rearrange(c.n.to(dtype=torch.float64), "f ck r -> f 1 1 ck r")
            if not (torch.allclose(N1, N2) and torch.allclose(N1, N3)):
                raise ShapeError("Hypergeometric triple product requires matching N.")
            N = N1
            max_k = int(
                torch.max(
                    torch.minimum(
                        torch.minimum(n1, K1), torch.minimum(torch.minimum(n2, K2), torch.minimum(n3, K3))
                    )
                ).item()
            )
            ks = rearrange(
                torch.arange(0, max_k + 1, dtype=torch.float64, device=N.device), "k -> k 1 1 1 1 1"
            )
            K1b = rearrange(K1, "f ci cj ck r -> 1 f ci cj ck r")
            K2b = rearrange(K2, "f ci cj ck r -> 1 f ci cj ck r")
            K3b = rearrange(K3, "f ci cj ck r -> 1 f ci cj ck r")
            Nb = rearrange(N, "f ci cj ck r -> 1 f ci cj ck r")
            n1b = rearrange(n1, "f ci cj ck r -> 1 f ci cj ck r")
            n2b = rearrange(n2, "f ci cj ck r -> 1 f ci cj ck r")
            n3b = rearrange(n3, "f ci cj ck r -> 1 f ci cj ck r")
            lp1 = _hypergeo_logpmf(ks, K1b, Nb, n1b)
            lp2 = _hypergeo_logpmf(ks, K2b, Nb, n2b)
            lp3 = _hypergeo_logpmf(ks, K3b, Nb, n3b)
            min1 = rearrange(
                torch.maximum(torch.zeros_like(N), n1 + K1 - N), "f ci cj ck r -> 1 f ci cj ck r"
            )
            max1 = rearrange(torch.minimum(n1, K1), "f ci cj ck r -> 1 f ci cj ck r")
            min2 = rearrange(
                torch.maximum(torch.zeros_like(N), n2 + K2 - N), "f ci cj ck r -> 1 f ci cj ck r"
            )
            max2 = rearrange(torch.minimum(n2, K2), "f ci cj ck r -> 1 f ci cj ck r")
            min3 = rearrange(
                torch.maximum(torch.zeros_like(N), n3 + K3 - N), "f ci cj ck r -> 1 f ci cj ck r"
            )
            max3 = rearrange(torch.minimum(n3, K3), "f ci cj ck r -> 1 f ci cj ck r")
            mask = (ks >= min1) & (ks <= max1) & (ks >= min2) & (ks <= max2) & (ks >= min3) & (ks <= max3)
            lsum = torch.logsumexp(
                torch.where(mask, lp1 + lp2 + lp3, torch.full_like(lp1, float("-inf"))), dim=0
            )
            out = torch.exp(lsum)
        elif (
            isinstance(a, NegativeBinomial)
            and isinstance(b, NegativeBinomial)
            and isinstance(c, NegativeBinomial)
        ):
            r1 = rearrange(a.total_count.to(dtype=torch.float64), "f ci r -> f ci 1 1 r")
            r2 = rearrange(b.total_count.to(dtype=torch.float64), "f cj r -> f 1 cj 1 r")
            r3 = rearrange(c.total_count.to(dtype=torch.float64), "f ck r -> f 1 1 ck r")
            p1 = (
                rearrange(a.probs.to(dtype=torch.float64), "f ci r -> f ci 1 1 r")
                .clamp_min(1e-30)
                .clamp_max(1.0)
            )
            p2 = (
                rearrange(b.probs.to(dtype=torch.float64), "f cj r -> f 1 cj 1 r")
                .clamp_min(1e-30)
                .clamp_max(1.0)
            )
            p3 = (
                rearrange(c.probs.to(dtype=torch.float64), "f ck r -> f 1 1 ck r")
                .clamp_min(1e-30)
                .clamp_max(1.0)
            )
            q = p1 * p2 * p3

            def log_term(k: int) -> Tensor:
                kk = torch.tensor(float(k), dtype=torch.float64, device=q.device)
                lt = (
                    torch.lgamma(r1 + kk)
                    - torch.lgamma(r1)
                    + torch.lgamma(r2 + kk)
                    - torch.lgamma(r2)
                    + torch.lgamma(r3 + kk)
                    - torch.lgamma(r3)
                    - 3.0 * torch.lgamma(kk + 1.0)
                    + kk * torch.log(q.clamp_min(1e-30))
                )
                const = (
                    r1 * torch.log((1.0 - p1).clamp_min(1e-30))
                    + r2 * torch.log((1.0 - p2).clamp_min(1e-30))
                    + r3 * torch.log((1.0 - p3).clamp_min(1e-30))
                )
                return lt + const

            logS = _series_logsumexp(log_terms_fn=log_term, max_k=4096, tol=1e-12, device=q.device)
            out = torch.exp(logS)
        elif isinstance(a, Histogram) and isinstance(b, Histogram) and isinstance(c, Histogram):
            edges1 = a.bin_edges.to(dtype=torch.float64, device=a.device)
            edges2 = b.bin_edges.to(dtype=torch.float64, device=b.device)
            edges3 = c.bin_edges.to(dtype=torch.float64, device=c.device)
            u_edges = torch.unique(torch.cat([edges1, edges2, edges3])).to(dtype=torch.float64)
            u_edges, _ = torch.sort(u_edges)
            seg_left = u_edges[:-1]
            seg_right = u_edges[1:]
            seg_len = (seg_right - seg_left).clamp_min(0.0)
            mids = (seg_left + seg_right) / 2.0

            widths1 = (edges1[1:] - edges1[:-1]).to(dtype=torch.float64)
            widths2 = (edges2[1:] - edges2[:-1]).to(dtype=torch.float64)
            widths3 = (edges3[1:] - edges3[:-1]).to(dtype=torch.float64)

            dens1 = a.probs.to(dtype=torch.float64) / rearrange(widths1, "b1 -> 1 1 1 b1")
            dens2 = b.probs.to(dtype=torch.float64) / rearrange(widths2, "b2 -> 1 1 1 b2")
            dens3 = c.probs.to(dtype=torch.float64) / rearrange(widths3, "b3 -> 1 1 1 b3")

            idx1 = (torch.bucketize(mids, edges1, right=True) - 1).clamp(0, widths1.numel() - 1)
            idx2 = (torch.bucketize(mids, edges2, right=True) - 1).clamp(0, widths2.numel() - 1)
            idx3 = (torch.bucketize(mids, edges3, right=True) - 1).clamp(0, widths3.numel() - 1)
            in1 = (mids >= edges1[0]) & (mids < edges1[-1])
            in2 = (mids >= edges2[0]) & (mids < edges2[-1])
            in3 = (mids >= edges3[0]) & (mids < edges3[-1])
            mask = (in1 & in2 & in3).to(dtype=torch.float64)

            d1 = dens1.index_select(-1, idx1)  # (F,Ca,R,S)
            d2 = dens2.index_select(-1, idx2)  # (F,Cb,R,S)
            d3 = dens3.index_select(-1, idx3)  # (F,Cc,R,S)
            prod = (
                rearrange(d1, "f ci r s -> f ci 1 1 r s")
                * rearrange(d2, "f cj r s -> f 1 cj 1 r s")
                * rearrange(d3, "f ck r s -> f 1 1 ck r s")
            )
            out = torch.sum(prod * rearrange(seg_len * mask, "s -> 1 1 1 1 1 s"), dim=-1)
        elif (
            isinstance(a, PiecewiseLinear)
            and isinstance(b, PiecewiseLinear)
            and isinstance(c, PiecewiseLinear)
        ):
            if not (a.is_initialized and b.is_initialized and c.is_initialized):
                raise UnsupportedOperationError(
                    "PiecewiseLinear triple product requires all leaves to be initialized."
                )
            if a.domains is None:
                raise UnsupportedOperationError("PiecewiseLinear triple product requires domains.")
            for dom in a.domains:
                if dom.data_type != DataType.CONTINUOUS:
                    raise UnsupportedOperationError(
                        "PiecewiseLinear triple product currently supports continuous domains only."
                    )

            dist_a = a.distribution()
            dist_b = b.distribution()
            dist_c = c.distribution()
            F, Ca, Cb, Cc, R = (
                a.out_shape.features,
                a.out_shape.channels,
                b.out_shape.channels,
                c.out_shape.channels,
                a.out_shape.repetitions,
            )
            out = torch.empty((F, Ca, Cb, Cc, R), dtype=torch.float64, device=a.device)

            def _get_knots(dist, r: int, leaf_idx: int, f: int) -> tuple[Tensor, Tensor]:
                xs = dist.xs[r][leaf_idx][f][0]
                ys = dist.ys[r][leaf_idx][f][0]
                return xs.to(dtype=torch.float64), ys.to(dtype=torch.float64)

            from spflow.modules.leaves.piecewise_linear import interp  # local import

            u1 = 0.5 - 0.5 / torch.sqrt(torch.tensor(3.0, dtype=torch.float64, device=a.device))
            u2 = 0.5 + 0.5 / torch.sqrt(torch.tensor(3.0, dtype=torch.float64, device=a.device))

            for r in range(R):
                for ca in range(Ca):
                    for cb in range(Cb):
                        for cc in range(Cc):
                            for f in range(F):
                                xa, ya = _get_knots(dist_a, r, ca, f)
                                xb, yb = _get_knots(dist_b, r, cb, f)
                                xc, yc = _get_knots(dist_c, r, cc, f)
                                grid = torch.unique(torch.cat([xa, xb, xc]))
                                grid, _ = torch.sort(grid)
                                if grid.numel() < 2:
                                    out[f, ca, cb, cc, r] = 0.0
                                    continue
                                fa = interp(grid, xa, ya, extrapolate="constant")
                                fb = interp(grid, xb, yb, extrapolate="constant")
                                fc = interp(grid, xc, yc, extrapolate="constant")
                                h = (grid[1:] - grid[:-1]).clamp_min(0.0)
                                a0, a1 = fa[:-1], fa[1:]
                                b0, b1 = fb[:-1], fb[1:]
                                c0, c1 = fc[:-1], fc[1:]
                                au1 = a0 + (a1 - a0) * u1
                                au2 = a0 + (a1 - a0) * u2
                                bu1 = b0 + (b1 - b0) * u1
                                bu2 = b0 + (b1 - b0) * u2
                                cu1 = c0 + (c1 - c0) * u1
                                cu2 = c0 + (c1 - c0) * u2
                                integral = torch.sum(h / 2.0 * (au1 * bu1 * cu1 + au2 * bu2 * cu2))
                                out[f, ca, cb, cc, r] = integral
        else:
            raise UnsupportedOperationError(
                f"Leaf triple product not implemented for {type(a).__name__} × {type(b).__name__} × {type(c).__name__}."
            )

        if cache is not None:
            memo[key] = out
        return out

    if isinstance(a, Cat) and isinstance(b, Cat) and isinstance(c, Cat):
        if a.dim != b.dim or a.dim != c.dim:
            raise ShapeError("Cat dim mismatch for triple product.")

        if a.dim == 1:
            if len(a.inputs) != len(b.inputs) or len(a.inputs) != len(c.inputs):
                raise ShapeError("Cat arity mismatch for triple product.")
            parts = [
                triple_product_tensor(
                    cast(Module, ai),
                    cast(Module, bi),
                    cast(Module, ci),
                    cache=cache,
                    signed_sum_types=signed_sum_types,
                    memo_key=memo_key,
                )
                for ai, bi, ci in zip(a.inputs, b.inputs, c.inputs)
            ]
            out = torch.cat(parts, dim=0)
            if cache is not None:
                memo[key] = out
            return out

        if a.dim == 2:
            F = a.out_shape.features
            R = a.out_shape.repetitions
            Ca = sum(cast(Module, ai).out_shape.channels for ai in a.inputs)
            Cb = sum(cast(Module, bi).out_shape.channels for bi in b.inputs)
            Cc = sum(cast(Module, ci).out_shape.channels for ci in c.inputs)
            out = torch.empty(
                (F, Ca, Cb, Cc, R), dtype=torch.float64, device=cast(Module, a.inputs[0]).device
            )
            a_off = 0
            for ai in a.inputs:
                ai_mod = cast(Module, ai)
                a_ch = ai_mod.out_shape.channels
                b_off = 0
                for bi in b.inputs:
                    bi_mod = cast(Module, bi)
                    b_ch = bi_mod.out_shape.channels
                    c_off = 0
                    for ci in c.inputs:
                        ci_mod = cast(Module, ci)
                        c_ch = ci_mod.out_shape.channels
                        out[
                            :, a_off : a_off + a_ch, b_off : b_off + b_ch, c_off : c_off + c_ch, :
                        ] = triple_product_tensor(
                            ai_mod,
                            bi_mod,
                            ci_mod,
                            cache=cache,
                            signed_sum_types=signed_sum_types,
                            memo_key=memo_key,
                        )
                        c_off += c_ch
                    b_off += b_ch
                a_off += a_ch
            if cache is not None:
                memo[key] = out
            return out

        raise UnsupportedOperationError(f"triple_product does not support Cat(dim={a.dim}).")

    if isinstance(a, Product) and isinstance(b, Product) and isinstance(c, Product):
        child_t = triple_product_tensor(
            cast(Module, a.inputs),
            cast(Module, b.inputs),
            cast(Module, c.inputs),
            cache=cache,
            signed_sum_types=signed_sum_types,
            memo_key=memo_key,
        )
        out = torch.prod(child_t, dim=0, keepdim=True)
        if cache is not None:
            memo[key] = out
        return out

    sum_types = (Sum, *signed_sum_types)
    if isinstance(a, sum_types) and isinstance(b, sum_types) and isinstance(c, sum_types):
        child_t = triple_product_tensor(
            cast(Module, a.inputs),
            cast(Module, b.inputs),
            cast(Module, c.inputs),
            cache=cache,
            signed_sum_types=signed_sum_types,
            memo_key=memo_key,
        )
        wa = a.weights.to(dtype=torch.float64)  # type: ignore[attr-defined]
        wb = b.weights.to(dtype=torch.float64)  # type: ignore[attr-defined]
        wc = c.weights.to(dtype=torch.float64)  # type: ignore[attr-defined]
        out = torch.einsum("fiar,fjbr,fkcr,fijkr->fabcr", wa, wb, wc, child_t)
        if cache is not None:
            memo[key] = out
        return out

    raise UnsupportedOperationError(
        f"triple_product_tensor not implemented for {type(a).__name__} × {type(b).__name__} × {type(c).__name__}."
    )


def triple_product_scalar(
    a: Module,
    b: Module,
    c: Module,
    *,
    cache: Cache | None = None,
    signed_sum_types: Sequence[type[Module]] = (),
    memo_key: str = "_triple_product_memo",
) -> Tensor:
    if tuple(a.out_shape) != (1, 1, 1) or tuple(b.out_shape) != (1, 1, 1) or tuple(c.out_shape) != (1, 1, 1):
        raise ShapeError("triple_product_scalar expects all modules to have out_shape == (1,1,1).")
    t = triple_product_tensor(a, b, c, cache=cache, signed_sum_types=signed_sum_types, memo_key=memo_key)
    return t[0, 0, 0, 0, 0]
