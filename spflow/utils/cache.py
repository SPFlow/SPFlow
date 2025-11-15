"""Cache utilities for efficient inference in SPFlow.

Provides thread-safe caching to optimize inference, learning, and sampling by
avoiding redundant computations in DAG traversals. Uses WeakKeyDictionary to
allow garbage collection of cached modules.
"""

from __future__ import annotations

import functools
import threading
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, TypeVar
from weakref import WeakKeyDictionary

if TYPE_CHECKING:  # Avoid circular imports
    from spflow.modules.base import Module

T = TypeVar("T")


class Cache:
    """Thread-safe cache with per-method-type locking and weak key references.

    Uses WeakKeyDictionary to store cached values keyed by module instances,
    allowing garbage collection when modules are no longer referenced elsewhere.
    """

    def __init__(self):
        """Initialize cache with per-method locks and storage."""
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._cache: dict[str, WeakKeyDictionary[Module, Any]] = {}

    def get(self, method_name: str, module: Module) -> Any | None:
        """Retrieve cached value for a module.

        Args:
            method_name: Name of the cached method (e.g., "log_likelihood").
            module: Module instance to use as cache key.

        Returns:
            Cached value if present, None otherwise.
        """
        if method_name not in self._cache:
            return None
        return self._cache[method_name].get(module)

    def set(self, method_name: str, module: Module, value: Any) -> None:
        """Store a value in cache for a module.

        Args:
            method_name: Name of the cached method (e.g., "log_likelihood").
            module: Module instance to use as cache key.
            value: Value to cache.
        """
        with self._locks[method_name]:
            if method_name not in self._cache:
                self._cache[method_name] = WeakKeyDictionary()
            self._cache[method_name][module] = value

    def __getitem__(self, method_name: str) -> WeakKeyDictionary[Module, Any]:
        """Get the cache dictionary for a method type (for backward compatibility).

        Args:
            method_name: Name of the cached method (e.g., "log_likelihood").

        Returns:
            WeakKeyDictionary for this method type.
        """
        if method_name not in self._cache:
            with self._locks[method_name]:
                if method_name not in self._cache:
                    self._cache[method_name] = WeakKeyDictionary()
        return self._cache[method_name]

    def __contains__(self, method_name: str) -> bool:
        """Check if a method type has any cached values.

        Args:
            method_name: Name of the cached method.

        Returns:
            True if the method type has cached entries.
        """
        return method_name in self._cache and len(self._cache[method_name]) > 0


def cached(method_name: str) -> Callable[[Callable], Callable]:
    """Decorator for automatically caching method results.

    Caches the result of a method in a thread-safe manner using the module
    instance as the cache key. The decorated method must have a `cache`
    parameter (can be None).

    Example:
        ```python
        @cached("log_likelihood")
        def log_likelihood(self, data, cache=None):
            # Computation here
            return result
        ```

    Args:
        method_name: Name to use as cache type key (e.g., "log_likelihood").

    Returns:
        Decorator function.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(
                self: Module, *args, cache: Cache | None = None, **kwargs
        ) -> T:
            # Initialize cache if not provided
            if cache is None:
                cache = Cache()

            # Check cache first
            cached_value = cache.get(method_name, self)
            if cached_value is not None:
                return cached_value

            # Compute result
            result = func(self, *args, cache=cache, **kwargs)

            # Store in cache
            cache.set(method_name, self, result)

            return result

        return wrapper

    return decorator
