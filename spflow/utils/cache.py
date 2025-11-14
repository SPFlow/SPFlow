"""Cache utilities for efficient inference in SPFlow.

Provides caching to optimize inference, learning, and sampling by avoiding
redundant computations in DAG traversals. Uses TypedDict for type safety.
"""

from __future__ import annotations

from typing import TypedDict, TYPE_CHECKING

import torch

if TYPE_CHECKING:  # Avoid circular imports
    from spflow.modules.base import Module


class Cache(TypedDict, total=False):
    """Cache dictionary for storing intermediate inference results.

    Keys:
        log_likelihood: Maps module instances to cached log-likelihood tensors.
    """

    log_likelihood: dict["Module", "torch.Tensor"]


def init_cache(cache: Cache | None) -> Cache:
    """Initialize cache dictionary if None.

    Args:
        cache: Existing cache dictionary to use, or None to create a new one.

    Returns:
        Cache: The initialized cache dictionary.
    """
    if cache is None:
        cache = {}
    return cache
