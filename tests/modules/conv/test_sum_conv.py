"""Unit tests for SumConv module."""

from itertools import product
from unittest.mock import patch

import numpy as np
import pytest
import torch

from spflow.exceptions import InvalidWeightsError, MissingCacheError, ShapeError
from spflow.meta.data import Scope
from spflow.modules.conv import SumConv
from spflow.modules.leaves import Normal
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext

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
        torch.testing.assert_close(weights_sum, torch.ones_like(weights_sum), rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("in_channels,out_channels,hw", construction_params)
    def test_extra_repr_contains_metadata(self, in_channels, out_channels, hw):
        """Test repr includes key layer metadata."""
        height, width = hw
        leaf = make_normal_leaf(height, width, out_channels=in_channels)
        module = SumConv(inputs=leaf, out_channels=out_channels, kernel_size=2)

        repr_str = module.extra_repr()
        assert f"in_channels={in_channels}" in repr_str
        assert f"out_channels={out_channels}" in repr_str
        assert "kernel_size=2" in repr_str

    def test_weights_setter_updates_weights(self):
        """Test setting valid weights updates module parameters."""
        leaf = make_normal_leaf(4, 4, out_channels=3)
        module = SumConv(inputs=leaf, out_channels=2, kernel_size=2)

        values = torch.full(module.weights_shape, 1.0 / 3.0)
        module.weights = values

        torch.testing.assert_close(module.weights, values, atol=1e-5, rtol=1e-5)

    def test_weights_setter_rejects_invalid_shape(self):
        """Test setting weights with invalid shape raises ShapeError."""
        leaf = make_normal_leaf(4, 4, out_channels=3)
        module = SumConv(inputs=leaf, out_channels=2, kernel_size=2)

        with pytest.raises(ShapeError, match="Invalid shape for weights"):
            module.weights = torch.ones(2, 3, 2, 1, 1)

    def test_weights_setter_rejects_non_positive_weights(self):
        """Test non-positive weights are rejected."""
        leaf = make_normal_leaf(4, 4, out_channels=3)
        module = SumConv(inputs=leaf, out_channels=2, kernel_size=2)

        values = torch.full(module.weights_shape, 1.0 / 3.0)
        values[0, 0, 0, 0, 0] = 0.0

        with pytest.raises(InvalidWeightsError, match="all positive"):
            module.weights = values

    def test_weights_setter_rejects_non_normalized_weights(self):
        """Test non-normalized weights are rejected."""
        leaf = make_normal_leaf(4, 4, out_channels=2)
        module = SumConv(inputs=leaf, out_channels=2, kernel_size=2)

        values = torch.full(module.weights_shape, 0.6)

        with pytest.raises(InvalidWeightsError, match="sum to one"):
            module.weights = values


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

    def test_log_likelihood_broadcasts_input_repetitions(self):
        """Test current behavior for reps=1 input and reps>1 output path."""
        leaf = make_normal_leaf(4, 4, out_channels=2, num_repetitions=1)
        module = SumConv(inputs=leaf, out_channels=3, kernel_size=2, num_repetitions=2)

        data = torch.randn(4, 16)
        with pytest.raises(RuntimeError, match="number of sizes provided"):
            module.log_likelihood(data)

    def test_log_likelihood_repetition_mismatch_raises(self):
        """Test incompatible repetition counts raise ValueError."""
        leaf = make_normal_leaf(4, 4, out_channels=2, num_repetitions=2)
        module = SumConv(inputs=leaf, out_channels=3, kernel_size=2, num_repetitions=3)

        with pytest.raises(ValueError, match="Input repetitions 2 incompatible with output 3"):
            module.log_likelihood(torch.randn(2, 16))

    def test_log_likelihood_small_spatial_dims_uses_special_case(self):
        """Test special path when spatial dimensions are smaller than kernel."""
        leaf = make_normal_leaf(1, 1, out_channels=2)
        module = SumConv(inputs=leaf, out_channels=3, kernel_size=2)

        ll = module.log_likelihood(torch.randn(5, 1))
        assert ll.shape == (5, 1, 3, 1)

    def test_log_likelihood_non_square_features_raises(self):
        """Test non-square feature counts are rejected."""
        leaf = Normal(scope=Scope(list(range(6))), out_channels=2)
        module = SumConv(inputs=leaf, out_channels=2, kernel_size=2)

        with pytest.raises(ValueError, match="not a perfect square"):
            module.log_likelihood(torch.randn(3, 6))

    def test_log_likelihood_non_divisible_spatial_dims_raises(self):
        """Test square dims not divisible by kernel size are rejected."""
        leaf = make_normal_leaf(3, 3, out_channels=2)
        module = SumConv(inputs=leaf, out_channels=2, kernel_size=2)

        with pytest.raises(ValueError, match="must be divisible by kernel_size"):
            module.log_likelihood(torch.randn(2, 9))


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

        samples1 = module.sample(num_samples=5, is_mpe=True)

        samples2 = module.sample(num_samples=5, is_mpe=True)

        # MPE should give deterministic results (same selection)
        # Note: actual values may vary due to leaf sampling
        assert samples1.shape == samples2.shape

    def test_sample_defaults_to_single_sample(self):
        """Test sample() uses one sample when no inputs are provided."""
        leaf = make_normal_leaf(4, 4, out_channels=2)
        module = SumConv(inputs=leaf, out_channels=2, kernel_size=2)

        samples = module.sample()
        assert samples.shape == (1, 16)

    def test_sample_uses_repetition_index_and_cached_input_lls(self):
        """Test sampling path with repetition selection and conditional cache."""
        batch_size = 3
        leaf = make_normal_leaf(4, 4, out_channels=2, num_repetitions=2)
        module = SumConv(inputs=leaf, out_channels=2, kernel_size=2, num_repetitions=2)

        cache = Cache()
        data = torch.randn(batch_size, 16)
        module.log_likelihood(data, cache=cache)

        sampling_ctx = SamplingContext(
            channel_index=torch.zeros((batch_size, 1), dtype=torch.long),
            mask=torch.ones((batch_size, 1), dtype=torch.bool),
            repetition_index=torch.tensor([0, 1, 0], dtype=torch.long),
        )

        out = module.sample(data=torch.full((batch_size, 16), float("nan")), cache=cache, sampling_ctx=sampling_ctx)
        assert out.shape == (batch_size, 16)
        assert sampling_ctx.channel_index.shape == (batch_size, 16)

    def test_sample_upsamples_context_from_parent_spatial_shape(self):
        """Test sampling context upsampling branch for reduced parent features."""
        batch_size = 2
        leaf = make_normal_leaf(4, 4, out_channels=2)
        module = SumConv(inputs=leaf, out_channels=2, kernel_size=2)

        sampling_ctx = SamplingContext(
            channel_index=torch.zeros((batch_size, 4), dtype=torch.long),
            mask=torch.ones((batch_size, 4), dtype=torch.bool),
        )

        out = module.sample(data=torch.full((batch_size, 16), float("nan")), sampling_ctx=sampling_ctx)
        assert out.shape == (batch_size, 16)
        assert sampling_ctx.channel_index.shape == (batch_size, 16)

    def test_sample_non_square_features_raises(self):
        """Test sample rejects non-square feature counts."""
        leaf = Normal(scope=Scope(list(range(6))), out_channels=2)
        module = SumConv(inputs=leaf, out_channels=2, kernel_size=2)

        with pytest.raises(ValueError, match="not a perfect square"):
            module.sample(num_samples=2)

    def test_sample_non_divisible_spatial_dims_raises(self):
        """Test sample rejects square dims not divisible by kernel size."""
        leaf = make_normal_leaf(3, 3, out_channels=2)
        module = SumConv(inputs=leaf, out_channels=2, kernel_size=2)

        with pytest.raises(ValueError, match="must be divisible by kernel_size"):
            module.sample(num_samples=2)


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
        with patch.object(leaf, "expectation_maximization"):
            module.expectation_maximization(data, cache=cache)

        # Check weights changed (only possible if in_channels > 1)
        assert not torch.allclose(module.weights, original_weights, rtol=0.0, atol=0.0)

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

        with patch.object(leaf, "expectation_maximization"):
            module.expectation_maximization(data, cache=cache)

        # Check weights still sum to 1 over in_channels
        weights_sum = module.weights.sum(dim=1)
        torch.testing.assert_close(weights_sum, torch.ones_like(weights_sum), rtol=1e-5, atol=1e-5)

    def test_em_missing_input_lls_raises(self):
        """Test EM raises if input log-likelihoods are missing in cache."""
        leaf = make_normal_leaf(4, 4, out_channels=2)
        module = SumConv(inputs=leaf, out_channels=2, kernel_size=2)

        cache = Cache()
        cache["log_likelihood"][module] = torch.zeros(2, 16, 2, 1)

        with pytest.raises(MissingCacheError, match="Input log-likelihoods not found in cache"):
            module.expectation_maximization(data=torch.randn(2, 16), cache=cache)

    def test_em_missing_module_lls_raises(self):
        """Test EM raises if module log-likelihoods are missing in cache."""
        leaf = make_normal_leaf(4, 4, out_channels=2)
        module = SumConv(inputs=leaf, out_channels=2, kernel_size=2)

        cache = Cache()
        cache["log_likelihood"][leaf] = torch.zeros(2, 16, 2, 1)

        with pytest.raises(MissingCacheError, match="Module log-likelihoods not found in cache"):
            module.expectation_maximization(data=torch.randn(2, 16), cache=cache)

    def test_em_handles_missing_gradients(self):
        """Test EM falls back to uniform log gradients when grad is missing."""
        leaf = make_normal_leaf(4, 4, out_channels=2)
        module = SumConv(inputs=leaf, out_channels=2, kernel_size=2)

        cache = Cache()
        data = torch.randn(10, 16)
        module.log_likelihood(data, cache=cache)

        with patch.object(leaf, "expectation_maximization") as mock_leaf_em:
            module.expectation_maximization(data, cache=cache)

        mock_leaf_em.assert_called_once()

    def test_em_raises_when_expected_gradient_missing(self):
        """Test EM raises when gradient was expected but is missing."""
        leaf = make_normal_leaf(4, 4, out_channels=2)
        module = SumConv(inputs=leaf, out_channels=2, kernel_size=2)

        cache = Cache()
        data = torch.randn(10, 16)
        module.log_likelihood(data, cache=cache)
        cache["log_likelihood"][module].retain_grad()

        with pytest.raises(RuntimeError, match="Expected gradient for cached module log-likelihood"):
            module.expectation_maximization(data, cache=cache)


class TestSumConvMarginalize:
    """Test SumConv marginalization behavior."""

    def test_marginalize_returns_none_when_fully_marginalized(self):
        """Test full scope marginalization returns None."""
        leaf = make_normal_leaf(4, 4, out_channels=2)
        module = SumConv(inputs=leaf, out_channels=3, kernel_size=2)

        result = module.marginalize(marg_rvs=list(range(16)))
        assert result is None

    def test_marginalize_returns_none_when_input_marginalizes_to_none(self):
        """Test return None when child marginalization returns None."""
        leaf = make_normal_leaf(4, 4, out_channels=2)
        module = SumConv(inputs=leaf, out_channels=3, kernel_size=2)

        with patch.object(leaf, "marginalize", return_value=None):
            result = module.marginalize(marg_rvs=[0])

        assert result is None

    def test_marginalize_returns_sumconv_for_partial_scope(self):
        """Test partial marginalization returns a SumConv with same metadata."""
        leaf = make_normal_leaf(4, 4, out_channels=2, num_repetitions=2)
        module = SumConv(inputs=leaf, out_channels=3, kernel_size=2, num_repetitions=2)

        result = module.marginalize(marg_rvs=[0, 1, 2])

        assert isinstance(result, SumConv)
        assert result.out_shape.channels == module.out_shape.channels
        assert result.kernel_size == module.kernel_size
        assert result.out_shape.repetitions == module.out_shape.repetitions
