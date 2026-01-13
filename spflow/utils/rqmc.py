"""Randomized quasi-Monte Carlo (RQMC) utilities.

This module provides small, dependency-free helpers to generate low-discrepancy
integration points for continuous mixtures, following the paper:

    "Continuous Mixtures of Tractable Probabilistic Models"
    Correia et al., 2023

We currently implement Sobol-based RQMC with a random shift at each call and an
inverse-CDF transform to map from U(0,1)^d to N(0, I).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterError


@dataclass(frozen=True)
class RqmcPoints:
    """RQMC integration points and weights.

    The returned tensors follow the usual numerical integration convention:

    - ``z`` has shape ``(num_points, latent_dim)``
    - ``weights`` has shape ``(num_points,)`` and sums to 1
    """

    z: Tensor
    weights: Tensor


def rqmc_sobol_normal(
    *,
    num_points: int,
    latent_dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    seed: int | None = None,
    eps: float = 1e-6,
) -> RqmcPoints:
    """Generate Sobol-RQMC points for a standard normal prior N(0, I).

    Uses a Sobol low-discrepancy sequence in (0,1)^d, applies a random shift
    modulo 1, and maps to N(0,I) via inverse CDF (icdf).

    Args:
        num_points: Number of integration points N.
        latent_dim: Latent dimension d.
        device: Target device.
        dtype: Target dtype for returned tensors.
        seed: Optional seed used to sample the random shift. If provided, this
            yields deterministic points.
        eps: Clamp epsilon for icdf numerical stability.

    Returns:
        RqmcPoints containing (z, weights).
    """
    if num_points < 1:
        raise InvalidParameterError("num_points must be >= 1.")
    if latent_dim < 1:
        raise InvalidParameterError("latent_dim must be >= 1.")
    if eps <= 0 or eps >= 0.1:
        raise InvalidParameterError("eps must be in (0, 0.1).")

    device = torch.device("cpu") if device is None else device
    dtype = torch.get_default_dtype() if dtype is None else dtype

    # SobolEngine produces a deterministic low-discrepancy sequence.
    engine = torch.quasirandom.SobolEngine(dimension=latent_dim, scramble=False, seed=0)
    u = engine.draw(num_points).to(device=device, dtype=dtype)  # (N,d) in [0,1)

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

    # Random shift modulo 1 (RQMC).
    shift = torch.rand((1, latent_dim), generator=gen, device=device, dtype=dtype)
    u = torch.remainder(u + shift, 1.0)
    u = u.clamp(min=eps, max=1.0 - eps)

    # Inverse-CDF transform to standard normal.
    normal = torch.distributions.Normal(
        loc=torch.zeros((), device=device, dtype=dtype),
        scale=torch.ones((), device=device, dtype=dtype),
    )
    z = normal.icdf(u)

    weights = torch.full((num_points,), 1.0 / float(num_points), device=device, dtype=dtype)
    return RqmcPoints(z=z, weights=weights)
