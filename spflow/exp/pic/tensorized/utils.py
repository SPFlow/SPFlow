"""Utility functions for tensorized layer computations.

Provides numerically stable log-space operations for tensorized probabilistic circuits.
"""

from typing import Callable

import torch
from torch import Tensor


def log_func_exp(*x: Tensor, func: Callable[..., Tensor], dim: int, keepdim: bool) -> Tensor:
    """Compute log(func(exp(x))) with numerical stability.

    This implements the log-sum-exp trick generalized to any linear function.
    The function should be linear and behave consistently with `dim` and `keepdim`.

    Args:
        *x: One or more input tensors. All tensors should be broadcastable
            along the relevant dimensions.
        func: A linear function that operates on exp(x). For a single tensor,
            this would typically be a sum-like operation. For two tensors,
            this might be a weighted combination.
        dim: The dimension that is collapsed by the sum-like operation.
        keepdim: Whether to keep `dim` as a size-1 dimension.

    Returns:
        The result of log(func(exp(x))), computed stably.
    """
    if len(x) == 0:
        raise ValueError("At least one tensor must be provided.")

    # Find max for numerical stability (per tensor)
    max_x = [torch.max(xi, dim=dim, keepdim=True)[0] for xi in x]

    # Subtract max and exponentiate
    exp_x = [torch.exp(xi - xi_max) for xi, xi_max in zip(x, max_x)]

    # Apply the linear function
    func_exp_x = func(*exp_x)

    # Sum of maxes (since f is linear: log(f(e^{x-m})) + sum(m) = log(f(e^x)))
    # For multiple inputs: sum all the maxes together
    sum_max_x = max_x[0]
    for m in max_x[1:]:
        sum_max_x = sum_max_x + m

    # Squeeze if not keeping dim
    if not keepdim:
        sum_max_x = sum_max_x.squeeze(dim)

    # Compute log of the result and add back the max
    log_func_exp_x = torch.log(func_exp_x) + sum_max_x

    return log_func_exp_x


def eval_tucker(left: Tensor, right: Tensor, weights: Tensor) -> Tensor:
    """Evaluate Tucker sum-product reduction in log-space.

    Args:
        left: Left child log-probabilities (F, K, *B).
        right: Right child log-probabilities (F, K, *B).
        weights: Tucker weights (F, K, K, O).

    Returns:
        Output log-probabilities (F, O, *B).
    """

    def linear(l: Tensor, r: Tensor) -> Tensor:
        return torch.einsum("fi...,fj...,fijo->fo...", l, r, weights)

    return log_func_exp(left, right, func=linear, dim=1, keepdim=True)


def eval_collapsed_cp(x: Tensor, weights: Tensor, fold_mask: Tensor | None = None) -> Tensor:
    """Evaluate collapsed CP sum-product reduction in log-space.

    Args:
        x: Input child log-probabilities (F, H, K, *B).
        weights: CP weights (F, H, K, O).
        fold_mask: Optional mask (F, H).

    Returns:
        Output log-probabilities (F, O, *B).
    """

    def in_linear(x_prob: Tensor) -> Tensor:
        return torch.einsum("fhko,fhk...->fho...", weights, x_prob)

    x_out = log_func_exp(x, func=in_linear, dim=2, keepdim=True)  # (F, H, O, *B)

    if fold_mask is not None:
        mask = fold_mask.to(dtype=torch.bool, device=x.device).view(
            fold_mask.shape + (1,) * (x_out.ndim - fold_mask.ndim)
        )
        # Use 0 (log 1) as neutral element for product reduction in log-space
        x_out = torch.where(mask, x_out, torch.zeros_like(x_out))

    return x_out.sum(dim=1)  # (F, O, *B)


def eval_mixing(x: Tensor, weights: Tensor, fold_mask: Tensor | None = None) -> Tensor:
    """Evaluate mixing sum layer in log-space.

    Args:
        x: Input child log-probabilities (F, H, K, *B).
        weights: Mixing weights (F, H, K).
        fold_mask: Optional mask (F, H).

    Returns:
        Output log-probabilities (F, K, *B).
    """

    def linear(x_prob: Tensor) -> Tensor:
        if fold_mask is not None:
            mask = fold_mask.to(dtype=x_prob.dtype, device=x_prob.device).view(
                fold_mask.shape + (1,) * (x_prob.ndim - fold_mask.ndim)
            )
            x_prob = x_prob * mask
        return torch.einsum("fhk,fhk...->fk...", weights, x_prob)

    return log_func_exp(x, func=linear, dim=1, keepdim=False)
