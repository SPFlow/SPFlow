"""Exact inner-product utilities for probabilistic circuits.

This module provides a small set of routines needed by SOCS (sum of squares)
to compute exact normalization constants of the form:

    Z_i = ∫ c_i(x)^2 dx

via a dynamic program that computes channel-wise inner products bottom-up.

The implementation is intentionally conservative and only supports a subset
of leaves initially (Normal, Bernoulli, Categorical).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from spflow.exceptions import ShapeError, UnsupportedOperationError
from spflow.modules.leaves.bernoulli import Bernoulli
from spflow.modules.leaves.categorical import Categorical
from spflow.modules.leaves.cltree import CLTree
from spflow.modules.leaves.exponential import Exponential
from spflow.modules.leaves.gamma import Gamma
from spflow.modules.leaves.laplace import Laplace
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.leaves.log_normal import LogNormal
from spflow.modules.leaves.normal import Normal
from spflow.modules.leaves.poisson import Poisson
from spflow.modules.module import Module
from spflow.modules.ops.cat import Cat
from spflow.modules.products.product import Product
from spflow.modules.sums.signed_sum import SignedSum
from spflow.modules.sums.sum import Sum


def _ensure_same_scope(a: Module, b: Module) -> None:
    if a.scope != b.scope:
        raise ShapeError(f"Scopes must match for inner product: {a.scope} vs {b.scope}.")


def leaf_inner_product(a: LeafModule, b: LeafModule) -> Tensor:
    """Compute per-feature/channel inner products ∫ f_a(x) f_b(x) dx for leaves.

    Returns:
        Tensor of shape (features, Ca, Cb, repetitions) in float64.
    """
    _ensure_same_scope(a, b)
    if a.out_shape.features != b.out_shape.features:
        raise ShapeError("Leaf features must match for inner product.")
    if a.out_shape.repetitions != b.out_shape.repetitions:
        raise ShapeError("Leaf repetitions must match for inner product.")

    if isinstance(a, Normal) and isinstance(b, Normal):
        # Shapes: (F, C, R)
        mu1 = a.loc.to(dtype=torch.float64)
        mu2 = b.loc.to(dtype=torch.float64)
        s1 = a.scale.to(dtype=torch.float64)
        s2 = b.scale.to(dtype=torch.float64)

        mu1 = mu1.unsqueeze(2)  # (F, Ca, 1, R)
        mu2 = mu2.unsqueeze(1)  # (F, 1, Cb, R)
        s1 = s1.unsqueeze(2)
        s2 = s2.unsqueeze(1)

        var = s1.pow(2) + s2.pow(2)
        log_coeff = -0.5 * torch.log(2.0 * torch.pi * var)
        quad = -(mu1 - mu2).pow(2) / (2.0 * var)
        return torch.exp(log_coeff + quad)

    if isinstance(a, Bernoulli) and isinstance(b, Bernoulli):
        p1 = a.probs.to(dtype=torch.float64).unsqueeze(2)  # (F, Ca, 1, R)
        p2 = b.probs.to(dtype=torch.float64).unsqueeze(1)  # (F, 1, Cb, R)
        return p1 * p2 + (1.0 - p1) * (1.0 - p2)

    if isinstance(a, Categorical) and isinstance(b, Categorical):
        if a.K != b.K:
            raise ShapeError(f"Categorical K mismatch: {a.K} vs {b.K}.")
        p1 = a.probs.to(dtype=torch.float64)  # (F, Ca, R, K)
        p2 = b.probs.to(dtype=torch.float64)
        p1 = p1.unsqueeze(2)  # (F, Ca, 1, R, K)
        p2 = p2.unsqueeze(1)  # (F, 1, Cb, R, K)
        return torch.sum(p1 * p2, dim=-1)  # (F, Ca, Cb, R)

    if isinstance(a, Exponential) and isinstance(b, Exponential):
        # rate > 0, support [0, inf)
        r1 = a.rate.to(dtype=torch.float64).unsqueeze(2)  # (F, Ca, 1, R)
        r2 = b.rate.to(dtype=torch.float64).unsqueeze(1)  # (F, 1, Cb, R)
        return (r1 * r2) / (r1 + r2).clamp_min(1e-30)

    if isinstance(a, Laplace) and isinstance(b, Laplace):
        # See PLAN_SOCS_EXTENSIONS.md for derivation; piecewise integration over x.
        mu1 = a.loc.to(dtype=torch.float64).unsqueeze(2)  # (F, Ca, 1, R)
        mu2 = b.loc.to(dtype=torch.float64).unsqueeze(1)  # (F, 1, Cb, R)
        b1 = a.scale.to(dtype=torch.float64).unsqueeze(2).clamp_min(1e-30)
        b2 = b.scale.to(dtype=torch.float64).unsqueeze(1).clamp_min(1e-30)
        d = torch.abs(mu1 - mu2)

        exp1 = torch.exp(-d / b1)
        exp2 = torch.exp(-d / b2)

        # Common term from tails:
        term_tails = (exp1 + exp2) / (4.0 * (b1 + b2))

        # Middle segment term:
        same = torch.isclose(b1, b2)
        # b1 != b2:
        term_mid = (exp1 - exp2) / (4.0 * (b1 - b2))
        # b1 == b2 == b:
        term_mid_same = torch.exp(-d / b1) * d / (4.0 * b1.pow(2))

        return torch.where(same, term_tails + term_mid_same, term_tails + term_mid)

    if isinstance(a, LogNormal) and isinstance(b, LogNormal):
        # Transform y=log x; integral reduces to Gaussian integral with an extra exp(-y).
        mu1 = a.loc.to(dtype=torch.float64).unsqueeze(2)  # (F, Ca, 1, R)
        mu2 = b.loc.to(dtype=torch.float64).unsqueeze(1)  # (F, 1, Cb, R)
        s1 = a.scale.to(dtype=torch.float64).unsqueeze(2).clamp_min(1e-30)
        s2 = b.scale.to(dtype=torch.float64).unsqueeze(1).clamp_min(1e-30)

        a1 = 1.0 / s1.pow(2)
        a2 = 1.0 / s2.pow(2)
        A = a1 + a2
        D = (a1 * mu1 + a2 * mu2) - 1.0
        E = -0.5 * (a1 * mu1.pow(2) + a2 * mu2.pow(2))

        log_pref = -0.5 * torch.log(2.0 * torch.pi * (s1.pow(2) + s2.pow(2)))
        return torch.exp(log_pref + E + (D.pow(2) / (2.0 * A)))

    if isinstance(a, Poisson) and isinstance(b, Poisson):
        # Inner product over k in {0,1,2,...}:
        #   Σ_k e^{-(λ1+λ2)} (λ1 λ2)^k / (k!)^2 = e^{-(λ1+λ2)} I0(2 sqrt(λ1 λ2)).
        l1 = a.rate.to(dtype=torch.float64).unsqueeze(2).clamp_min(0.0)  # (F, Ca, 1, R)
        l2 = b.rate.to(dtype=torch.float64).unsqueeze(1).clamp_min(0.0)  # (F, 1, Cb, R)
        z = 2.0 * torch.sqrt((l1 * l2).clamp_min(0.0))
        i0 = getattr(torch, "i0", torch.special.i0)  # torch.i0 is available in newer PyTorch
        return torch.exp(-(l1 + l2)) * i0(z)

    if isinstance(a, Gamma) and isinstance(b, Gamma):
        # Gamma(k, rate): f(x) = β^α / Γ(α) x^{α-1} e^{-βx}, x>0.
        # ∫ f1 f2 dx = β1^{α1} β2^{α2} Γ(α1+α2-1) / (Γ(α1)Γ(α2)(β1+β2)^{α1+α2-1})
        a1 = a.concentration.to(dtype=torch.float64).unsqueeze(2).clamp_min(1e-30)  # (F, Ca, 1, R)
        a2 = b.concentration.to(dtype=torch.float64).unsqueeze(1).clamp_min(1e-30)  # (F, 1, Cb, R)
        b1 = a.rate.to(dtype=torch.float64).unsqueeze(2).clamp_min(1e-30)
        b2 = b.rate.to(dtype=torch.float64).unsqueeze(1).clamp_min(1e-30)

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

    if isinstance(a, CLTree) and isinstance(b, CLTree):
        # Tractable only when the two trees share the same structure.
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

        # Probabilities (C,R,...) in float64.
        log_cpt_a = a.log_cpt.to(dtype=torch.float64)
        log_cpt_b = b.log_cpt.to(dtype=torch.float64)

        C1, C2 = a.out_shape.channels, b.out_shape.channels
        R = a.out_shape.repetitions
        K = a.K
        F = a.out_shape.features

        # Root marginals: (C,R,K)
        pa_root = torch.exp(log_cpt_a[root, :, :, :, 0])
        pb_root = torch.exp(log_cpt_b[root, :, :, :, 0])

        # Messages m[node]: (C1,C2,R,K) message from node to its parent as function of parent state.
        msg = torch.ones((F, C1, C2, R, K), dtype=torch.float64, device=log_cpt_a.device)

        post_order = a.post_order.tolist()
        for i in post_order:
            p = parents[i]
            if p == -1:
                continue

            # product of child messages as a function of x_i
            prod_child = torch.ones((C1, C2, R, K), dtype=torch.float64, device=log_cpt_a.device)
            for ch in children[i]:
                prod_child = prod_child * msg[ch]

            pa = torch.exp(log_cpt_a[i])  # (C1,R,K,K) [xi,xp]
            pb = torch.exp(log_cpt_b[i])  # (C2,R,K,K)
            # Combine into (C1,C2,R,xi,xp)
            phi = pa.unsqueeze(1) * pb.unsqueeze(0)
            # m_i(xp) = Σ_xi prod_child(xi) * phi(xi,xp)
            msg_i = torch.einsum("abri,abrio->abro", prod_child, phi)
            msg[i] = msg_i

        # Root product
        prod_root = torch.ones((C1, C2, R, K), dtype=torch.float64, device=log_cpt_a.device)
        for ch in children[root]:
            prod_root = prod_root * msg[ch]

        phi_root = pa_root.unsqueeze(1) * pb_root.unsqueeze(0)  # (C1,C2,R,K)
        z = torch.sum(phi_root * prod_root, dim=-1)  # (C1,C2,R)

        out = torch.ones((F, C1, C2, R), dtype=torch.float64, device=log_cpt_a.device)
        out[0] = z
        return out

    raise UnsupportedOperationError(
        f"Leaf inner product not implemented for {type(a).__name__} × {type(b).__name__}. "
        "Supported: Normal, Bernoulli, Categorical, Exponential, Laplace, LogNormal, Poisson, Gamma, CLTree."
    )


def inner_product_matrix(a: Module, b: Module) -> Tensor:
    """Compute channel-wise inner product matrix between compatible modules.

    Returns:
        Tensor K of shape (features, Ca, Cb, repetitions) in float64, where
        K[f,i,j,r] = ∫ a_{f,i,r}(x) * b_{f,j,r}(x) dx over the module's scope.
    """
    _ensure_same_scope(a, b)
    if a.out_shape.features != b.out_shape.features:
        raise ShapeError(
            f"Feature mismatch: {a.out_shape.features} vs {b.out_shape.features} for {type(a).__name__}."
        )
    if a.out_shape.repetitions != b.out_shape.repetitions:
        raise ShapeError("Repetition mismatch for inner product.")

    # Leaves
    if isinstance(a, LeafModule) and isinstance(b, LeafModule):
        return leaf_inner_product(a, b)

    # Cat is structural: dim=1 concatenates independent factors as separate features,
    # dim=2 concatenates channels for a shared feature axis.
    if isinstance(a, Cat) and isinstance(b, Cat):
        if a.dim != b.dim:
            raise ShapeError("Cat dim mismatch for inner product.")

        if a.dim == 1:
            if len(a.inputs) != len(b.inputs):
                raise ShapeError("Cat arity mismatch for inner product.")
            parts = [
                inner_product_matrix(cast(Module, ai), cast(Module, bi)) for ai, bi in zip(a.inputs, b.inputs)
            ]
            return torch.cat(parts, dim=0)

        if a.dim == 2:
            # Build a block matrix K over concatenated channels.
            F = a.out_shape.features
            R = a.out_shape.repetitions
            Ca = sum(cast(Module, ai).out_shape.channels for ai in a.inputs)
            Cb = sum(cast(Module, bi).out_shape.channels for bi in b.inputs)

            # Precompute all child Ks.
            blocks: list[list[Tensor]] = []
            for ai in a.inputs:
                row: list[Tensor] = []
                for bi in b.inputs:
                    row.append(inner_product_matrix(cast(Module, ai), cast(Module, bi)))
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

            return out

        raise UnsupportedOperationError(f"inner_product does not support Cat(dim={a.dim}).")

    # Product: output feature is 1, channels are shared across features
    if isinstance(a, Product) and isinstance(b, Product):
        # Product wraps a single child module (often a Cat(dim=1) of factors).
        child_k = inner_product_matrix(cast(Module, a.inputs), cast(Module, b.inputs))  # (F, Ca, Cb, R)
        # Multiply across features (independent factors)
        return torch.prod(child_k, dim=0, keepdim=True)  # (1, Ca, Cb, R)

    # Sum / SignedSum: quadratic form in the child inner products
    if isinstance(a, (Sum, SignedSum)) and isinstance(b, (Sum, SignedSum)):
        child_k = inner_product_matrix(cast(Module, a.inputs), cast(Module, b.inputs))  # (F, ICa, ICb, R)
        if child_k.dim() != 4:
            raise ShapeError(f"Expected child K to be 4D, got {tuple(child_k.shape)}.")

        wa = (a.weights if isinstance(a, Sum) else a.weights).to(dtype=torch.float64)  # type: ignore[attr-defined]
        wb = (b.weights if isinstance(b, Sum) else b.weights).to(dtype=torch.float64)  # type: ignore[attr-defined]

        # Shapes:
        #   wa: (F, ICa, OCa, R)
        #   wb: (F, ICb, OCb, R)
        if wa.dim() != 4 or wb.dim() != 4:
            raise ShapeError("Expected weights to be 4D for inner product.")
        if wa.shape[0] != child_k.shape[0] or wb.shape[0] != child_k.shape[0]:
            raise ShapeError("Feature mismatch between weights and child inner products.")
        if wa.shape[3] != child_k.shape[3] or wb.shape[3] != child_k.shape[3]:
            raise ShapeError("Repetition mismatch between weights and child inner products.")

        F, _, _, R = child_k.shape
        OCa = wa.shape[2]
        OCb = wb.shape[2]
        out = torch.empty((F, OCa, OCb, R), dtype=torch.float64, device=child_k.device)

        # For each feature and repetition: out = waᵀ K wb
        for r in range(R):
            for f in range(F):
                kf = child_k[f, :, :, r]  # (ICa, ICb)
                waf = wa[f, :, :, r]  # (ICa, OCa)
                wbf = wb[f, :, :, r]  # (ICb, OCb)
                out[f, :, :, r] = waf.t().matmul(kf).matmul(wbf)

        return out

    raise UnsupportedOperationError(
        f"inner_product_matrix not implemented for {type(a).__name__} × {type(b).__name__}."
    )


def log_self_inner_product_scalar(module: Module) -> Tensor:
    """Compute log ∫ f(x)^2 dx for scalar-output modules.

    Expects the module to output shape (features=1, channels=1, repetitions=1).
    """
    if tuple(module.out_shape) != (1, 1, 1):
        raise ShapeError(f"Expected scalar output (1,1,1) for SOCS component, got {tuple(module.out_shape)}.")

    k = inner_product_matrix(module, module)  # (1, 1, 1, 1)
    z = k[0, 0, 0, 0]
    # Numerical safety: Z should be >= 0 for a self inner product; clamp tiny negatives.
    z = torch.clamp(z, min=0.0)
    return torch.log(z.clamp_min(1e-30))
