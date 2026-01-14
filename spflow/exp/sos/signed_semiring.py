"""Real signed-semirings utilities for numerically stable circuit evaluation.

This module implements a sign-aware log-absolute-value representation:
any real value `x` is represented as a pair `(log|x|, sign(x))`.

This is used to evaluate circuits that may contain negative parameters (e.g.,
SignedSum) while avoiding underflow/overflow when magnitudes are very small/large.
"""

from __future__ import annotations

import torch
from torch import Tensor


def sign_of(x: Tensor) -> Tensor:
    """Return ``sign(x)`` in {-1, 0, +1} as an integer tensor."""
    return torch.sign(x).to(dtype=torch.int8)


def logabs_of(x: Tensor, eps: float = 0.0) -> Tensor:
    """Return ``log(|x|)``, with optional epsilon to avoid ``log(0)``.

    Args:
        x: Input tensor.
        eps: If > 0, computes ``log(|x| + eps)``.
    """
    if eps > 0.0:
        return torch.log(torch.abs(x) + x.new_tensor(eps))
    return torch.log(torch.abs(x))


def signed_logsumexp(
    logabs_terms: Tensor,
    sign_terms: Tensor,
    dim: int,
    keepdim: bool = False,
    eps: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Compute log|Σ_i s_i exp(a_i)| and sign of the sum in a stable way.

    Args:
        logabs_terms: Log-absolute-values `a_i` of the terms.
        sign_terms: Signs `s_i` of the terms in {-1, 0, +1}. Must be broadcastable.
        dim: Dimension to reduce over.
        keepdim: Whether to keep the reduced dimension.
        eps: Additive epsilon to avoid log(0) in edge cases.

    Returns:
        (logabs_sum, sign_sum)
    """
    if logabs_terms.numel() == 0:
        raise ValueError("signed_logsumexp requires at least one term.")

    # m = max(a_i) for stability (treat -inf properly)
    m = torch.max(logabs_terms, dim=dim, keepdim=True).values

    # If all terms are -inf along `dim`, the sum is exactly 0.
    # Avoid exp(nan) from (-inf) - (-inf).
    all_neg_inf = torch.isneginf(m)

    # exp(a_i - m) is in [0, 1] for finite a_i
    scaled = torch.exp(logabs_terms - m)
    if all_neg_inf.any():
        scaled = torch.where(all_neg_inf, torch.zeros_like(scaled), scaled)
    signed_scaled = scaled * sign_terms.to(dtype=scaled.dtype)
    s = torch.sum(signed_scaled, dim=dim, keepdim=True)

    # Handle zeros safely
    sign_s = sign_of(s)
    abs_s = torch.abs(s)
    if eps > 0.0:
        abs_s = abs_s + abs_s.new_tensor(eps)

    logabs_s = torch.log(abs_s)
    out_logabs = m + logabs_s
    if all_neg_inf.any():
        out_logabs = torch.where(all_neg_inf, torch.full_like(out_logabs, float("-inf")), out_logabs)

    if not keepdim:
        out_logabs = out_logabs.squeeze(dim)
        sign_s = sign_s.squeeze(dim)

    return out_logabs, sign_s
