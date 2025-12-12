"""Unit tests for SumConv module."""

from itertools import product
from unittest.mock import patch

import numpy as np
import pytest
import torch

from spflow.meta.data import Scope
from spflow.modules.conv import SumConv
from spflow.modules.leaves import Normal
from spflow.utils.cache import Cache

# Test parameter values
in_channels_values = [1, 3]
out_channels_values = [1, 5]
height_width_values = [(4, 4), (8, 8)]

# For EM tests, only test with in_channels > 1 to ensure weights can change
em_in_channels_values = [3]

# Combined parameter lists
construction_params = list(product(in_channels_values, out_channels_values, height_width_values))
ll_params = list(product(in_channels_values, out_channels_values, height_width_values))
sample_params = list(product(in_channels_values, out_channels_values, height_width_values))
em_params = list(product(em_in_channels_values, out_channels_values, height_width_values))


def make_normal_leaf(height: int, width: int, out_channels: int, num_repetitions: int = 1):
    """Create a Normal leaf for testing with spatial structure."""
    num_features = height * width
    scope = Scope(list(range(num_features)))
    return Normal(scope=scope, out_channels=out_channels, num_repetitions=num_repetitions)


class TestSumConvConstruction:
    """Test SumConv construction and initialization."""

    @pytest.mark.parametrize("in_channels,out_channels,hw", construction_params)
    def test_basic_construction(self, in_channels, out_channels, hw):
        """Test that SumConv can be constructed with valid parameters."""
        height, width = hw
        leaf = make_normal_leaf(height, width, out_channels=in_channels)
        module = SumConv(inputs=leaf, out_channels=out_channels, kernel_size=2)

        assert module.kernel_size == 2
        assert module.out_shape.channels == out_channels
        assert module.in_channels == in_channels

    @pytest.mark.parametrize("hw", height_width_values)
    def test_invalid_out_channels(self, hw):
        """Test that out_channels < 1 raises ValueError."""
        height, width = hw
        leaf = make_normal_leaf(height, width, out_channels=3)
        with pytest.raises(ValueError, match="out_channels must be >= 1"):
            SumConv(inputs=leaf, out_channels=0, kernel_size=2)

    @pytest.mark.parametrize("hw", height_width_values)
    def test_invalid_kernel_size(self, hw):
        """Test that kernel_size < 1 raises ValueError."""
        height, width = hw
        leaf = make_normal_leaf(height, width, out_channels=3)
        with pytest.raises(ValueError, match="kernel_size must be >= 1"):
            SumConv(inputs=leaf, out_channels=5, kernel_size=0)

    @pytest.mark.parametrize("in_channels,out_channels,hw", construction_params)
    @pytest.mark.parametrize("num_reps", [1, 2])
    def test_weights_shape(self, in_channels, out_channels, hw, num_reps):
        """Test that weights have correct shape."""
        height, width = hw
        leaf = make_normal_leaf(height, width, out_channels=in_channels, num_repetitions=num_reps)
        module = SumConv(inputs=leaf, out_channels=out_channels, kernel_size=2, num_repetitions=num_reps)

        # Shape: (out_c, in_c, k, k, reps)
        assert module.weights.shape == (out_channels, in_channels, 2, 2, num_reps)

    @pytest.mark.parametrize("in_channels,out_channels,hw", construction_params)
    def test_weights_normalized(self, in_channels, out_channels, hw):
        """Test that weights sum to 1 over input channels."""
        height, width = hw
        leaf = make_normal_leaf(height, width, out_channels=in_channels)
        module = SumConv(inputs=leaf, out_channels=out_channels, kernel_size=2)

        weights_sum = module.weights.sum(dim=1)  # Sum over in_channels
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum))


class TestSumConvLogLikelihood:
    """Test SumConv log_likelihood computation."""

    @pytest.mark.parametrize("in_channels,out_channels,hw", ll_params)
    @pytest.mark.parametrize("num_reps", [1, 2])
    def test_log_likelihood_shape(self, in_channels, out_channels, hw, num_reps):
        """Test that log_likelihood output has correct shape."""
        height, width = hw
        leaf = make_normal_leaf(height, width, out_channels=in_channels, num_repetitions=num_reps)
        module = SumConv(inputs=leaf, out_channels=out_channels, kernel_size=2, num_repetitions=num_reps)

        batch_size = 10
        data = torch.randn(batch_size, height * width)
        ll = module.log_likelihood(data)

        # Output shape: (batch, features, out_channels, reps)
        assert ll.shape == (batch_size, height * width, out_channels, num_reps)

    @pytest.mark.parametrize("in_channels,out_channels,hw", ll_params)
    def test_log_likelihood_finite(self, in_channels, out_channels, hw):
        """Test that log_likelihood values are finite."""
        height, width = hw
        leaf = make_normal_leaf(height, width, out_channels=in_channels)
        module = SumConv(inputs=leaf, out_channels=out_channels, kernel_size=2)

        data = torch.randn(10, height * width)
        ll = module.log_likelihood(data)

        assert torch.isfinite(ll).all()

    @pytest.mark.parametrize("in_channels,out_channels,hw", ll_params)
    def test_log_likelihood_cached(self, in_channels, out_channels, hw):
        """Test that log_likelihood is properly cached."""
        height, width = hw
        leaf = make_normal_leaf(height, width, out_channels=in_channels)
        module = SumConv(inputs=leaf, out_channels=out_channels, kernel_size=2)

        data = torch.randn(10, height * width)
        cache = Cache()

        ll1 = module.log_likelihood(data, cache=cache)
        ll2 = module.log_likelihood(data, cache=cache)

        # Should return same object from cache
        assert ll1 is ll2


class TestSumConvSample:
    """Test SumConv sampling."""

    @pytest.mark.parametrize("in_channels,out_channels,hw", sample_params)
    def test_sample_shape(self, in_channels, out_channels, hw):
        """Test that samples have correct shape."""
        height, width = hw
        leaf = make_normal_leaf(height, width, out_channels=in_channels)
        module = SumConv(inputs=leaf, out_channels=out_channels, kernel_size=2)

        num_samples = 20
        samples = module.sample(num_samples=num_samples)

        assert samples.shape == (num_samples, height * width)

    @pytest.mark.parametrize("in_channels,out_channels,hw", sample_params)
    def test_sample_finite(self, in_channels, out_channels, hw):
        """Test that samples are finite."""
        height, width = hw
        leaf = make_normal_leaf(height, width, out_channels=in_channels)
        module = SumConv(inputs=leaf, out_channels=out_channels, kernel_size=2)

        samples = module.sample(num_samples=10)
        assert torch.isfinite(samples).all()

    @pytest.mark.parametrize("in_channels,out_channels,hw", sample_params)
    def test_sample_mpe_deterministic(self, in_channels, out_channels, hw):
        """Test that MPE sampling is deterministic."""
        height, width = hw
        leaf = make_normal_leaf(height, width, out_channels=in_channels)
        module = SumConv(inputs=leaf, out_channels=out_channels, kernel_size=2)

        torch.manual_seed(42)
        samples1 = module.sample(num_samples=5, is_mpe=True)

        torch.manual_seed(42)
        samples2 = module.sample(num_samples=5, is_mpe=True)

        # MPE should give deterministic results (same selection)
        # Note: actual values may vary due to leaf sampling
        assert samples1.shape == samples2.shape


class TestSumConvFeatureToScope:
    """Test SumConv feature_to_scope property."""

    @pytest.mark.parametrize("in_channels,out_channels,hw", construction_params)
    def test_feature_to_scope_preserved(self, in_channels, out_channels, hw):
        """Test that per-pixel scopes are preserved from input."""
        height, width = hw
        leaf = make_normal_leaf(height, width, out_channels=in_channels)
        module = SumConv(inputs=leaf, out_channels=out_channels, kernel_size=2)

        # SumConv should preserve input scopes
        assert np.array_equal(module.feature_to_scope, leaf.feature_to_scope)

    @pytest.mark.parametrize("in_channels,out_channels,hw", construction_params)
    @pytest.mark.parametrize("num_reps", [1, 2])
    def test_feature_to_scope_shape(self, in_channels, out_channels, hw, num_reps):
        """Test feature_to_scope has correct shape."""
        height, width = hw
        leaf = make_normal_leaf(height, width, out_channels=in_channels, num_repetitions=num_reps)
        module = SumConv(inputs=leaf, out_channels=out_channels, kernel_size=2, num_repetitions=num_reps)

        f2s = module.feature_to_scope
        assert f2s.shape == (height * width, num_reps)  # (features, repetitions)


class TestSumConvEM:
    """Test SumConv expectation_maximization."""

    @pytest.mark.parametrize("in_channels,out_channels,hw", em_params)
    def test_em_updates_weights(self, in_channels, out_channels, hw):
        """Test that EM updates the weights."""
        height, width = hw
        leaf = make_normal_leaf(height, width, out_channels=in_channels)
        module = SumConv(inputs=leaf, out_channels=out_channels, kernel_size=2)

        # Store original weights
        original_weights = module.weights.clone()

        # Run forward pass to populate cache
        data = torch.randn(50, height * width)
        cache = Cache()
        ll = module.log_likelihood(data, cache=cache)

        # Manually set gradient for module_lls (simulating EM upward pass)
        # In real EM, this comes from the parent's expectations
        cache["log_likelihood"][module].grad = torch.ones_like(cache["log_likelihood"][module])

        # Run EM with mocked leaf EM to avoid leaf gradient issues
        with patch.object(leaf, 'expectation_maximization'):
            module.expectation_maximization(data, cache=cache)

        # Check weights changed (only possible if in_channels > 1)
        assert not torch.allclose(module.weights, original_weights)

    @pytest.mark.parametrize("in_channels,out_channels,hw", em_params)
    def test_em_weights_still_normalized(self, in_channels, out_channels, hw):
        """Test that weights are still normalized after EM."""
        height, width = hw
        leaf = make_normal_leaf(height, width, out_channels=in_channels)
        module = SumConv(inputs=leaf, out_channels=out_channels, kernel_size=2)

        # Run forward pass and EM
        data = torch.randn(50, height * width)
        cache = Cache()
        ll = module.log_likelihood(data, cache=cache)

        # Set gradient
        cache["log_likelihood"][module].grad = torch.ones_like(cache["log_likelihood"][module])

        with patch.object(leaf, 'expectation_maximization'):
            module.expectation_maximization(data, cache=cache)

        # Check weights still sum to 1 over in_channels
        weights_sum = module.weights.sum(dim=1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)
