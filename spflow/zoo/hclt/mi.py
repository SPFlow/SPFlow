"""Chow-Liu mutual information utilities.

This module implements the performance-critical parts of ChowLiuTrees.jl that we
need for structure learning (binary fast-path). The APIs are intentionally small
and reusable by both CLTree and HCLT learners.
"""

from __future__ import annotations

import torch
from einops import repeat
from torch import Tensor

from spflow.exceptions import InvalidParameterError, ShapeError


def _as_float_tensor(x: Tensor, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    if x.device != device:
        x = x.to(device)
    if x.dtype != dtype:
        x = x.to(dtype=dtype)
    return x


def pairwise_marginal_binary(
    data: Tensor,
    *,
    weights: Tensor | None = None,
    pseudocount: float = 0.0,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Compute pairwise marginals for binary data.

    Matches ChowLiuTrees.jl semantics for binary inputs:
    - `data` is interpreted as 0/1 valued (bool or numeric).
    - `weights` are optional per-sample weights (not renormalized).
    - `pseudocount` adds a *total* mass spread uniformly across the 4 joint
      states (00,01,10,11), and additionally smooths the diagonal (i==j) to
      represent marginals.

    Args:
        data: Tensor of shape (N, F) with values in {0,1} (or bool).
        weights: Optional tensor of shape (N,).
        pseudocount: Total pseudocount mass to add.
        dtype: Floating dtype for computation.

    Returns:
        Tensor of shape (F, F, 4) holding probabilities for [00, 01, 10, 11].
    """
    if data.dim() != 2:
        raise ShapeError(f"data must be 2D (N,F), got shape {tuple(data.shape)}.")
    if pseudocount < 0:
        raise InvalidParameterError("pseudocount must be >= 0.")

    device = data.device
    x = (data != 0) if data.dtype != torch.bool else data
    x = _as_float_tensor(x, dtype=dtype, device=device)
    nx, nf = x.shape
    not_x = 1.0 - x

    if weights is None:
        w_sum = float(nx)
        xw = x
        not_xw = not_x
    else:
        if weights.dim() != 1 or weights.shape[0] != nx:
            raise ShapeError(f"weights must have shape ({nx},), got {tuple(weights.shape)}.")
        w = _as_float_tensor(weights, dtype=dtype, device=device)
        if not torch.isfinite(w).all():
            raise InvalidParameterError("weights must be finite.")
        w_sum = float(w.sum().item())
        xw = x * w[:, None]
        not_xw = not_x * w[:, None]

    base = w_sum + float(pseudocount)
    if base <= 0:
        raise InvalidParameterError("Total mass (sum(weights)+pseudocount) must be > 0.")

    # Counts (F,F)
    c11 = x.t().matmul(xw)
    c10 = x.t().matmul(not_xw)
    c01 = not_x.t().matmul(xw)
    c00 = (w_sum - c11 - c10 - c01).clamp_min(0.0)

    # Stack into (F,F,4) in [00,01,10,11] order.
    pxy = torch.stack([c00, c01, c10, c11], dim=-1)

    # Uniform pseudocount over the 4 joint states.
    joint_add = float(pseudocount) / 4.0
    if joint_add:
        pxy = pxy + joint_add

    # Diagonal represents marginals: keep only {00,11}, and add another joint_add
    # to those entries (mirrors ChowLiuTrees.jl behavior).
    diag = torch.arange(nf, device=device)
    if joint_add:
        pxy[diag, diag, 0] = pxy[diag, diag, 0] + joint_add
        pxy[diag, diag, 3] = pxy[diag, diag, 3] + joint_add
    pxy[diag, diag, 1] = 0.0
    pxy[diag, diag, 2] = 0.0

    return pxy / base


def pairwise_mi_binary(
    data: Tensor,
    *,
    weights: Tensor | None = None,
    pseudocount: float = 0.0,
    dtype: torch.dtype = torch.float64,
    eps: float = 1e-12,
) -> Tensor:
    """Compute mutual information matrix for binary data.

    Returns MI for (i,j). The diagonal MI(i,i) equals the entropy H(X_i).
    Semantics match ChowLiuTrees.jl `pairwise_MI` for binary inputs.
    """
    pxy = pairwise_marginal_binary(data, weights=weights, pseudocount=pseudocount, dtype=dtype)
    nf = pxy.shape[0]

    p0 = pxy[..., 0].diagonal()  # p(x=0)
    p1 = pxy[..., 3].diagonal()  # p(x=1)

    pxpy = torch.empty((nf, nf, 4), dtype=pxy.dtype, device=pxy.device)
    pxpy[..., 0] = p0[:, None] * p0[None, :]
    pxpy[..., 1] = p0[:, None] * p1[None, :]
    pxpy[..., 2] = p1[:, None] * p0[None, :]
    pxpy[..., 3] = p1[:, None] * p1[None, :]

    # xlogx and xlogy with safe handling of zeros.
    def xlogx(p: Tensor) -> Tensor:
        return torch.where(p == 0, torch.zeros_like(p), p * torch.log(p.clamp_min(eps)))

    def xlogy(p: Tensor, q: Tensor) -> Tensor:
        return torch.where(p == 0, torch.zeros_like(p), p * torch.log(q.clamp_min(eps)))

    mi = (xlogx(pxy) - xlogy(pxy, pxpy)).sum(dim=-1)
    return mi


def pairwise_mi_categorical(
    data: Tensor,
    *,
    num_cats: int | None = None,
    weights: Tensor | None = None,
    pseudocount: float = 0.0,
    dtype: torch.dtype = torch.float64,
    chunk_size_pairs: int = 4096,
    eps: float = 1e-12,
) -> Tensor:
    """Compute mutual information matrix for categorical (discrete) data.

    This is designed to be correct and reasonably fast without allocating
    O(F*F*K*K) memory. It works by batching variable pairs and using a single
    `torch.bincount` per batch.

    Semantics follow ChowLiuTrees.jl for categorical inputs:
    - Values are assumed in {0, ..., K-1}.
    - If `weights` provided, pseudocount is scaled by sum(weights)/N.
    - Pseudocount is distributed uniformly over categories.
    """
    if data.dim() != 2:
        raise ShapeError(f"data must be 2D (N,F), got shape {tuple(data.shape)}.")
    if pseudocount < 0:
        raise InvalidParameterError("pseudocount must be >= 0.")
    if chunk_size_pairs < 1:
        raise InvalidParameterError("chunk_size_pairs must be >= 1.")

    n, f = data.shape
    device = data.device

    x = data
    if torch.isnan(x).any():
        raise InvalidParameterError("pairwise_mi_categorical requires complete data (no NaNs).")
    x = x.to(dtype=torch.long)

    if num_cats is None:
        num_cats = int(x.max().item()) + 1 if x.numel() else 0
    if num_cats <= 0:
        raise InvalidParameterError("num_cats must be >= 1.")
    if x.numel() and (x.min().item() < 0 or x.max().item() >= num_cats):
        raise InvalidParameterError("Categorical data must be in {0, ..., num_cats-1}.")

    if weights is None:
        w = torch.ones(n, device=device, dtype=dtype)
        pc = float(pseudocount)
    else:
        if weights.dim() != 1 or weights.shape[0] != n:
            raise ShapeError(f"weights must have shape ({n},), got {tuple(weights.shape)}.")
        w = _as_float_tensor(weights, dtype=dtype, device=device)
        if not torch.isfinite(w).all():
            raise InvalidParameterError("weights must be finite.")
        # Match ChowLiuTrees.jl scaling: keep pseudocount comparable to unweighted case.
        pc = float(pseudocount) * float(w.sum().item()) / float(n)

    Z = float(w.sum().item()) + pc
    if Z <= 0:
        raise InvalidParameterError("Total mass (sum(weights)+pseudocount) must be > 0.")

    # Single-variable marginals with smoothing.
    single_add = pc / float(num_cats)
    counts_i = torch.zeros((f, num_cats), device=device, dtype=dtype)
    for k in range(num_cats):
        counts_i[:, k] = (w[:, None] * (x == k).to(dtype)).sum(dim=0)
    if single_add:
        counts_i = counts_i + single_add
    p_i = counts_i / Z
    log_p_i = torch.log(p_i.clamp_min(eps))

    # MI diag = entropy.
    mi = torch.zeros((f, f), device=device, dtype=dtype)
    mi_diag = -(p_i * log_p_i).sum(dim=1)
    mi.fill_diagonal_(0.0)
    mi[torch.arange(f, device=device), torch.arange(f, device=device)] = mi_diag

    # Prepare all pairs (i<j).
    idx_i, idx_j = torch.triu_indices(f, f, offset=1, device=device)
    num_pairs = int(idx_i.numel())
    if num_pairs == 0:
        return mi

    bins_per_pair = num_cats * num_cats
    pair_add = pc / float(bins_per_pair)

    for start in range(0, num_pairs, chunk_size_pairs):
        end = min(start + chunk_size_pairs, num_pairs)
        ii = idx_i[start:end]  # (B,)
        jj = idx_j[start:end]  # (B,)
        num_pairs_batch = int(ii.numel())

        # codes: (N,B) in [0, K*K)
        codes = x[:, ii] * num_cats + x[:, jj]
        offsets = (torch.arange(num_pairs_batch, device=device, dtype=torch.long) * bins_per_pair).view(
            1, num_pairs_batch
        )
        flat = (codes + offsets).reshape(-1)

        w_flat = repeat(w, "n -> (n pair_batch)", pair_batch=num_pairs_batch)
        counts = torch.bincount(flat, weights=w_flat, minlength=num_pairs_batch * bins_per_pair).reshape(
            num_pairs_batch, num_cats, num_cats
        )
        if pair_add:
            counts = counts + pair_add
        p_ij = counts / Z
        log_p_ij = torch.log(p_ij.clamp_min(eps))

        # MI per pair: sum_{a,b} p_ij(a,b) * (log p_ij - log p_i(a) - log p_j(b))
        term = log_p_ij - log_p_i[ii, :, None] - log_p_i[jj, None, :]
        mi_vals = (p_ij * term).sum(dim=(1, 2))

        mi[ii, jj] = mi_vals
        mi[jj, ii] = mi_vals

    return mi
