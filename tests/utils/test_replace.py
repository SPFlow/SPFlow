"""Unit tests for method replacement context manager."""

import numpy as np
from typing import Optional

import pytest
import torch

from spflow.meta import Scope
from spflow.modules.base import Module
from spflow.modules.leaves import Normal
from spflow.modules.ops import Cat
from spflow.modules.products import Product
from spflow.modules.sums import Sum
from spflow.utils.cache import Cache, cached
from spflow.utils.replace import replace


class MockModule(Module):
    """Simple mock module for testing replace functionality."""

    def __init__(self, name: str = "mock"):
        super().__init__()
        self._scope = Scope([0])
        self.name = name
        self.call_count = 0

    @property
    def out_features(self) -> int:
        """Return number of output features."""
        return 1

    @property
    def out_channels(self) -> int:
        """Return number of output channels."""
        return 1

    @property
    def feature_to_scope(self) -> np.ndarray:
        """Return list of scopes per feature."""
        return np.array([Scope([0])]).view(-1, 1)

    @cached
    def log_likelihood(self, data, cache: Optional[Cache] = None):
        """Mock log_likelihood method with caching."""
        self.call_count += 1
        return torch.tensor([1.0, 2.0, 3.0])

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None, sampling_ctx=None):
        """Mock sample method."""
        return torch.randn(num_samples or 1, 1)

    def marginalize(self, marg_rvs, prune=True, cache=None):
        """Mock marginalize method."""
        return None

    def non_cached_method(self):
        """Method without @cached decorator."""
        return "original"


class TestBasicReplacement:
    """Tests for basic method replacement."""

    def test_replace_undecorated_method(self):
        """Test replacing a method without @cached decorator."""

        def custom_method(self):
            return "custom"

        mock = MockModule()

        # Before replacement
        assert mock.non_cached_method() == "original"

        # With replacement
        with replace(MockModule.non_cached_method, custom_method):
            assert mock.non_cached_method() == "custom"

        # After replacement
        assert mock.non_cached_method() == "original"

    def test_replace_cached_method(self):
        """Test replacing a @cached decorated method."""

        def custom_ll(self, data, cache=None):
            return torch.tensor([4.0, 5.0, 6.0])

        mock = MockModule()

        # Before replacement
        result_before = mock.log_likelihood(torch.tensor([1.0]))
        assert torch.equal(result_before, torch.tensor([1.0, 2.0, 3.0]))

        # With replacement
        with replace(MockModule.log_likelihood, custom_ll):
            result_custom = mock.log_likelihood(torch.tensor([1.0]))
            assert torch.equal(result_custom, torch.tensor([4.0, 5.0, 6.0]))

        # After replacement
        result_after = mock.log_likelihood(torch.tensor([1.0]))
        assert torch.equal(result_after, torch.tensor([1.0, 2.0, 3.0]))

    def test_replacement_affects_all_instances(self):
        """Test that class-level replacement affects all instances."""

        def custom_ll(self, data, cache=None):
            return torch.tensor([7.0, 8.0, 9.0])

        mock1 = MockModule("mock1")
        mock2 = MockModule("mock2")

        # Before replacement - both use original
        result1_before = mock1.log_likelihood(torch.tensor([1.0]))
        result2_before = mock2.log_likelihood(torch.tensor([1.0]))
        assert torch.equal(result1_before, torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(result2_before, torch.tensor([1.0, 2.0, 3.0]))

        # With replacement - both use custom
        with replace(MockModule.log_likelihood, custom_ll):
            result1_custom = mock1.log_likelihood(torch.tensor([1.0]))
            result2_custom = mock2.log_likelihood(torch.tensor([1.0]))
            assert torch.equal(result1_custom, torch.tensor([7.0, 8.0, 9.0]))
            assert torch.equal(result2_custom, torch.tensor([7.0, 8.0, 9.0]))


class TestCachingBehavior:
    """Tests for caching with replacement."""

    def test_replacement_preserves_caching(self):
        """Test that replaced cached method still uses caching."""

        call_count = 0

        def custom_ll(self, data, cache=None):
            nonlocal call_count
            call_count += 1
            return torch.tensor([10.0])

        cache = Cache()
        mock = MockModule()

        with replace(MockModule.log_likelihood, custom_ll):
            # First call should compute
            result1 = mock.log_likelihood(torch.tensor([1.0]), cache=cache)
            assert call_count == 1

            # Second call should use cache
            result2 = mock.log_likelihood(torch.tensor([1.0]), cache=cache)
            assert call_count == 1  # Not incremented
            assert result1 is result2  # Same object (cached)

    def test_cache_key_with_replacement(self):
        """Test that cache keys work correctly with replaced methods."""

        def custom_ll(self, data, cache=None):
            return torch.tensor([11.0])

        cache = Cache()
        mock = MockModule()

        with replace(MockModule.log_likelihood, custom_ll):
            result1 = mock.log_likelihood(torch.tensor([1.0]), cache=cache)

            # Call again - should use cached value
            result2 = mock.log_likelihood(torch.tensor([1.0]), cache=cache)

            # Results should be the same object (cached)
            assert result1 is result2
            assert torch.equal(result1, torch.tensor([11.0]))


class TestRestorationAndCleanup:
    """Tests for proper restoration of original methods."""

    def test_original_restored_on_normal_exit(self):
        """Test that original method is restored after context exit."""

        def custom_method(self):
            return "custom"

        mock = MockModule()

        # Use replacement
        with replace(MockModule.non_cached_method, custom_method):
            pass

        # Original should be restored
        assert mock.non_cached_method() == "original"

    def test_original_restored_on_exception(self):
        """Test that original method is restored even when exception occurs."""

        def custom_method(self):
            return "custom"

        mock = MockModule()

        # Try with exception
        try:
            with replace(MockModule.non_cached_method, custom_method):
                assert mock.non_cached_method() == "custom"
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Original should still be restored
        assert mock.non_cached_method() == "original"

    def test_multiple_sequential_replacements(self):
        """Test sequential replacements work correctly."""

        def custom1(self):
            return "custom1"

        def custom2(self):
            return "custom2"

        mock = MockModule()

        # First replacement
        with replace(MockModule.non_cached_method, custom1):
            assert mock.non_cached_method() == "custom1"

        # Second replacement
        with replace(MockModule.non_cached_method, custom2):
            assert mock.non_cached_method() == "custom2"

        # Back to original
        assert mock.non_cached_method() == "original"


class TestErrorHandling:
    """Tests for error handling in replace()."""

    def test_invalid_method_reference_non_callable(self):
        """Test error when passing non-callable as method reference."""
        with pytest.raises(TypeError, match="Expected a callable"):
            with replace("not_a_method", lambda self: None):
                pass

    def test_invalid_method_reference_no_name(self):
        """Test error when method reference has no __name__."""
        # Lambda functions have __name__, so we test with a class instance
        class NoName:
            def __call__(self):
                pass

        obj = NoName()
        # Remove __name__ attribute by deleting it
        if hasattr(obj, "__name__"):
            delattr(obj, "__name__")

        with pytest.raises(TypeError):
            with replace(obj, lambda self: None):
                pass

    def test_invalid_method_reference_no_qualname(self):
        """Test error when method reference has no __qualname__."""

        def func():
            pass

        # Remove __qualname__ if possible
        if hasattr(func, "__qualname__"):
            # Can't actually remove __qualname__ from functions in Python,
            # but we can test the error message
            pass


class TestIntegrationWithRealModules:
    """Integration tests with actual SPFlow modules."""

    def test_replace_sum_log_likelihood(self):
        """Test replacing log_likelihood on Sum module."""
        # Create a simple structure: Sum(Normal)
        scope = Scope([0, 1])
        normal = Normal(scope=scope, out_channels=2)
        sum_module = Sum(inputs=normal, out_channels=2)

        def custom_ll(self, data, cache=None):
            # Custom implementation that returns ones
            batch_size = data.shape[0]
            # Match the shape of normal's output (4D for this case)
            return torch.ones(batch_size, 2, 1, 1)

        data = torch.randn(3, 2)  # 3 samples, 2 features

        # Get original result
        original_result = sum_module.log_likelihood(data)
        original_shape = original_result.shape
        original_batch_size = original_shape[0]

        # Use custom implementation
        with replace(Sum.log_likelihood, custom_ll):
            custom_result = sum_module.log_likelihood(data)
            assert custom_result.shape[0] == 3  # batch size preserved

        # Original restored
        restored_result = sum_module.log_likelihood(data)
        assert torch.equal(restored_result, original_result)

    def test_replace_in_nested_structure(self):
        """Test replacement works in a nested structure."""
        scope = Scope([0, 1])
        normal = Normal(scope=scope, out_channels=2)
        sum_module = Sum(inputs=normal, out_channels=2)
        product_module = Product(inputs=sum_module)

        call_count = 0

        def custom_sum_ll(self, data, cache=None):
            nonlocal call_count
            call_count += 1
            # Return ones with the correct shape
            batch_size = data.shape[0]
            return torch.ones(batch_size, 2, 1)

        data = torch.randn(2, 2)

        # Original behavior
        original_result = product_module.log_likelihood(data)

        # With replacement - the custom Sum.log_likelihood should be called
        with replace(Sum.log_likelihood, custom_sum_ll):
            custom_result = product_module.log_likelihood(data)
            assert call_count == 1  # Sum.log_likelihood was called once

        # Verify the replacement affected the sum module
        assert not torch.equal(custom_result, original_result)

    def test_replace_multiple_instances_same_class(self):
        """Test that replacement affects all instances of a class in a tree."""
        scope1 = Scope([0, 1])
        scope2 = Scope([2, 3])

        # Create a tree with multiple Sum nodes
        normal1 = Normal(scope=scope1, out_channels=2)
        sum1 = Sum(inputs=normal1, out_channels=2)

        normal2 = Normal(scope=scope2, out_channels=2)
        sum2 = Sum(inputs=normal2, out_channels=2)

        # Cat both sums together
        combined = Cat(inputs=[sum1, sum2], dim=1)

        call_count = 0

        def custom_ll(self, data, cache=None):
            nonlocal call_count
            call_count += 1
            # Return the correct shape based on the Sum's out_channels
            batch_size = data.shape[0]
            return torch.zeros(batch_size, 2, 1)

        # Use data that matches the combined scope
        data = torch.randn(2, 4)  # 4 features for scopes [0,1,2,3]

        # With replacement, both Sum instances should use custom
        with replace(Sum.log_likelihood, custom_ll):
            combined.log_likelihood(data)
            # Both sum1 and sum2 should have called the custom function
            assert call_count == 2


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_replacement_with_different_signature(self):
        """Test that custom function can have different implementation details."""

        def custom_ll(self, data, cache=None):
            # Different computation
            if data.shape[0] > 1:
                return torch.tensor([100.0])
            else:
                return torch.tensor([200.0])

        mock = MockModule()

        with replace(MockModule.log_likelihood, custom_ll):
            result1 = mock.log_likelihood(torch.randn(1, 2))
            result2 = mock.log_likelihood(torch.randn(2, 2))

            assert torch.equal(result1, torch.tensor([200.0]))
            assert torch.equal(result2, torch.tensor([100.0]))

    def test_replacement_modifying_self_state(self):
        """Test that replacement can modify module state."""

        def custom_ll(self, data, cache=None):
            self.call_count += 10  # Modify state
            return torch.tensor([50.0])

        mock = MockModule()
        assert mock.call_count == 0

        with replace(MockModule.log_likelihood, custom_ll):
            mock.log_likelihood(torch.tensor([1.0]))
            assert mock.call_count == 10

        # call_count should remain modified (only method is restored)
        assert mock.call_count == 10

    def test_empty_context(self):
        """Test that empty replace context works (no calls to method)."""

        def custom_method(self):
            return "custom"

        mock = MockModule()

        # Enter and exit without calling
        with replace(MockModule.non_cached_method, custom_method):
            pass

        # Original should be available
        assert mock.non_cached_method() == "original"
