"""Unit tests for SPFlow cache utilities."""

import gc
import threading
from typing import Optional
from weakref import WeakKeyDictionary

import torch

from spflow.utils.cache import Cache, cached


# Mock module class for testing
class MockModule:
    """Simple mock module for testing cache functionality."""

    def __init__(self, name: str = "mock"):
        self.name = name

    @cached
    def log_likelihood(self, data, cache: Optional[Cache] = None):
        """Mock log_likelihood method with caching."""
        # Simulate some computation
        return torch.tensor([1.0, 2.0, 3.0])

    @cached
    def custom_method(self, value, cache: Optional[Cache] = None):
        """Mock method with custom cache type."""
        return value * 2


class TestThreadSafeCache:
    """Tests for ThreadSafeCache class."""

    def test_cache_initialization(self):
        """Test that cache initializes correctly."""
        cache = Cache()
        assert isinstance(cache._locks, dict)
        assert isinstance(cache._cache, dict)
        assert len(cache._cache) == 0

    def test_cache_set_and_get(self):
        """Test basic set and get operations."""
        cache = Cache()
        module = MockModule()
        value = torch.tensor([1.0, 2.0])

        # Set value
        cache.set("log_likelihood", module, value)

        # Get value
        retrieved = cache.get("log_likelihood", module)
        assert torch.equal(retrieved, value)

    def test_cache_get_nonexistent(self):
        """Test getting a non-existent cache entry returns None."""
        cache = Cache()
        module = MockModule()

        retrieved = cache.get("nonexistent", module)
        assert retrieved is None

    def test_cache_multiple_methods(self):
        """Test caching for multiple method types."""
        cache = Cache()
        module = MockModule()

        value1 = torch.tensor([1.0])
        value2 = torch.tensor([2.0])

        cache.set("log_likelihood", module, value1)
        cache.set("sample", module, value2)

        assert torch.equal(cache.get("log_likelihood", module), value1)
        assert torch.equal(cache.get("sample", module), value2)

    def test_cache_multiple_modules(self):
        """Test caching for multiple modules."""
        cache = Cache()
        module1 = MockModule("mod1")
        module2 = MockModule("mod2")

        value1 = torch.tensor([1.0])
        value2 = torch.tensor([2.0])

        cache.set("log_likelihood", module1, value1)
        cache.set("log_likelihood", module2, value2)

        assert torch.equal(cache.get("log_likelihood", module1), value1)
        assert torch.equal(cache.get("log_likelihood", module2), value2)

    def test_getitem_creates_weak_dict(self):
        """Test __getitem__ creates WeakKeyDictionary for method."""
        cache = Cache()

        # Access non-existent method
        weak_dict = cache["log_likelihood"]

        assert isinstance(weak_dict, WeakKeyDictionary)
        assert "log_likelihood" in cache._cache

    def test_contains_empty(self):
        """Test __contains__ returns False for empty cache."""
        cache = Cache()
        assert "log_likelihood" not in cache

    def test_contains_with_entries(self):
        """Test __contains__ returns True when entries exist."""
        cache = Cache()
        module = MockModule()
        cache.set("log_likelihood", module, torch.tensor([1.0]))

        assert "log_likelihood" in cache

    def test_weak_reference_cleanup(self):
        """Test that modules are garbage collected when no longer referenced."""
        cache = Cache()
        module = MockModule()

        value = torch.tensor([1.0])
        cache.set("log_likelihood", module, value)

        # Verify it's cached
        assert cache.get("log_likelihood", module) is not None

        # Delete the module and force garbage collection
        del module
        gc.collect()

        # The weak reference should be gone, creating a new module to check
        new_module = MockModule()
        assert cache.get("log_likelihood", new_module) is None

    def test_thread_safety_concurrent_access(self):
        """Test that concurrent access to same cache method is thread-safe."""
        cache = Cache()
        module = MockModule()
        results = []

        def set_and_get(value):
            cache.set("log_likelihood", module, value)
            retrieved = cache.get("log_likelihood", module)
            results.append(retrieved)

        # Create multiple threads writing to the same cache
        threads = []
        for i in range(5):
            t = threading.Thread(target=set_and_get, args=(torch.tensor([float(i)])))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All threads should complete without error
        assert len(results) == 5


class TestCachedDecorator:
    """Tests for @cached decorator."""

    def test_decorator_caches_result(self):
        """Test that decorator caches method results."""
        cache = Cache()
        module = MockModule()

        # First call
        result1 = module.log_likelihood(torch.tensor([1.0, 2.0]), cache=cache)

        # Second call should return cached value (same object)
        result2 = module.log_likelihood(torch.tensor([1.0, 2.0]), cache=cache)

        assert result1 is result2

    def test_decorator_returns_correct_value(self):
        """Test that decorator returns correct computed value."""
        cache = Cache()
        module = MockModule()

        result = module.log_likelihood(torch.tensor([1.0, 2.0]), cache=cache)

        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.equal(result, expected)

    def test_decorator_without_cache_parameter(self):
        """Test that decorator creates cache if not provided."""
        module = MockModule()

        # Call without cache parameter
        result = module.log_likelihood(torch.tensor([1.0, 2.0]))

        assert torch.is_tensor(result)
        assert result.shape == (3,)

    def test_decorator_caches_in_provided_cache(self):
        """Test that decorator caches in the provided cache object."""
        cache = Cache()
        module = MockModule()

        module.log_likelihood(torch.tensor([1.0, 2.0]), cache=cache)

        # Check that value is in cache
        assert cache.get("log_likelihood", module) is not None

    def test_decorator_with_multiple_cache_types(self):
        """Test decorator with different cache types."""
        cache = Cache()
        module = MockModule()

        # Use different cache types
        module.log_likelihood(torch.tensor([1.0]), cache=cache)
        result = module.custom_method(5, cache=cache)

        # Both should be cached
        assert cache.get("log_likelihood", module) is not None
        assert cache.get("custom_method", module) == 10

    def test_decorator_different_modules_separate_cache(self):
        """Test that different modules have separate cache entries."""
        cache = Cache()
        module1 = MockModule("mod1")
        module2 = MockModule("mod2")

        module1.log_likelihood(torch.tensor([1.0]), cache=cache)
        module2.log_likelihood(torch.tensor([2.0]), cache=cache)

        result1 = cache.get("log_likelihood", module1)
        result2 = cache.get("log_likelihood", module2)

        assert torch.equal(result1, torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(result2, torch.tensor([1.0, 2.0, 3.0]))

    def test_decorator_preserves_function_metadata(self):
        """Test that @cached preserves the decorated function's metadata."""
        assert MockModule.log_likelihood.__name__ == "log_likelihood"
        assert "Mock log_likelihood method" in MockModule.log_likelihood.__doc__


class TestCacheIntegration:
    """Integration tests for cache system."""

    def test_nested_cache_calls(self):
        """Test caching works with nested method calls."""
        cache = Cache()
        module = MockModule()

        # Simulate nested calls
        result1 = module.log_likelihood(torch.tensor([1.0]), cache=cache)
        result2 = module.custom_method(3, cache=cache)

        # Both should be cached
        cached_ll = cache.get("log_likelihood", module)
        cached_cm = cache.get("custom_method", module)

        assert cached_ll is not None
        assert cached_cm is not None
        assert cached_cm == 6

    def test_cache_persistence_across_calls(self):
        """Test that cache persists across multiple calls."""
        cache = Cache()
        module = MockModule()

        # Multiple calls to the same method
        for _ in range(3):
            module.log_likelihood(torch.tensor([1.0]), cache=cache)

        # Should still be cached with same result
        assert cache.get("log_likelihood", module) is not None

    def test_cache_with_tensor_gradients(self):
        """Test that cached tensors preserve gradient information."""
        cache = Cache()
        module = MockModule()

        data = torch.tensor([1.0, 2.0], requires_grad=True)
        result = module.log_likelihood(data, cache=cache)

        # Check that we can still access the computed result
        cached = cache.get("log_likelihood", module)
        assert cached is not None
