"""Cache utilities for inference operations."""
from __future__ import annotations

from typing import TypedDict, TYPE_CHECKING

import torch

if TYPE_CHECKING:  # Avoid circular imports
    from spflow.modules.base import Module


class Cache(TypedDict, total=False):
    """Type definition for cache dictionary used during inference.

    The cache stores intermediate results to avoid redundant computations
    in DAG traversal, enable efficient EM training, and support conditional sampling.

    Keys:
        log_likelihood: Maps module instances to their cached log-likelihood tensors.
    """

    log_likelihood: dict["Module", "torch.Tensor"]


def init_cache(cache: Cache | None) -> Cache:
    """Initialize cache dictionary if None.

    Args:
        cache: Existing cache or None.

    Returns:
        Initialized cache dictionary.
    """
    if cache is None:
        cache = {}
    return cache
