"""Tests for RepetitionMixingLayer expectation_maximization method."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from spflow.learn import expectation_maximization
from spflow.meta import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.sums.repetition_mixing_layer import RepetitionMixingLayer
from spflow.utils.cache import Cache, cached


class CachingDummyInput(Module):
    """Dummy input module that properly caches log_likelihood for EM testing."""

    def __init__(self, out_channels: int = 2, num_repetitions: int = 2, out_features: int = 1):
        super().__init__()
        self.scope = Scope(list(range(out_features)))
        self._feature_to_scope = np.array(
            [[Scope(i)] for i in range(out_features)], dtype=object
        )
        self._in_shape = ModuleShape(out_features, 1, 1)
        self._out_shape = ModuleShape(out_features, out_channels, num_repetitions)

    @property
    def feature_to_scope(self):
        return self._feature_to_scope

    @cached
    def log_likelihood(self, data, cache=None):
        batch = data.shape[0]
        # Return small negative values that require gradients for EM
        result = torch.randn(
            batch,
            self.out_shape.features,
            self.out_shape.channels,
            self.out_shape.repetitions,
            device=data.device,
            requires_grad=True,
        )
        return result

    def sample(self, *args, **kwargs):
        data = kwargs.get("data")
        if data is None:
            data = torch.full((1, len(self.scope.query)), torch.nan, device=self.device)
        data[:, self.scope.query] = 0.0
        return data

    def expectation_maximization(self, data, cache=None):
        return None

    def maximum_likelihood_estimation(self, data, weights=None, cache=None):
        return None

    def marginalize(self, marg_rvs, prune=True, cache=None):
        return self


# Test parameters
in_channels_values = [2, 4]
num_repetitions_values = [2, 3]
out_features_values = [1, 2]


class TestRepetitionMixingLayerEMBasic:
    """Test RepetitionMixingLayer expectation_maximization basic functionality."""

    @pytest.mark.parametrize("in_channels", in_channels_values)
    @pytest.mark.parametrize("num_reps", num_repetitions_values)
    @pytest.mark.parametrize("out_features", out_features_values)
    def test_em_updates_weights(self, in_channels: int, num_reps: int, out_features: int):
        """Test that EM updates the weights."""
        inputs = CachingDummyInput(
            out_channels=in_channels, num_repetitions=num_reps, out_features=out_features
        )
        layer = RepetitionMixingLayer(
            inputs=inputs, out_channels=in_channels, num_repetitions=num_reps
        )

        # Store original weights
        original_weights = layer.weights.clone()

        # Run forward pass to populate cache
        data = torch.randn(50, out_features)
        cache = Cache()
        ll = layer.log_likelihood(data, cache=cache)

        # Set gradient for module_lls (simulating EM upward pass)
        cache["log_likelihood"][layer].grad = torch.ones_like(cache["log_likelihood"][layer])

        # Run EM with mocked leaf EM to avoid propagation issues
        with patch.object(inputs, "expectation_maximization"):
            layer.expectation_maximization(data, cache=cache)

        # Check weights changed (always possible since sum_dim sums over repetitions)
        assert not torch.allclose(layer.weights, original_weights)

    @pytest.mark.parametrize("in_channels", in_channels_values)
    @pytest.mark.parametrize("num_reps", num_repetitions_values)
    def test_em_weights_still_normalized(self, in_channels: int, num_reps: int):
        """Test that weights are still normalized after EM."""
        inputs = CachingDummyInput(out_channels=in_channels, num_repetitions=num_reps)
        layer = RepetitionMixingLayer(
            inputs=inputs, out_channels=in_channels, num_repetitions=num_reps
        )

        # Run forward pass and EM
        data = torch.randn(50, 1)
        cache = Cache()
        ll = layer.log_likelihood(data, cache=cache)

        # Set gradient
        cache["log_likelihood"][layer].grad = torch.ones_like(cache["log_likelihood"][layer])

        with patch.object(inputs, "expectation_maximization"):
            layer.expectation_maximization(data, cache=cache)

        # Check weights still sum to 1 over sum_dim (repetitions)
        weights_sum = layer.weights.sum(dim=layer.sum_dim)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)


class TestRepetitionMixingLayerEMErrors:
    """Test RepetitionMixingLayer expectation_maximization error handling."""

    def test_em_raises_without_input_lls_in_cache(self):
        """Test that EM raises ValueError when input log-likelihoods not in cache."""
        inputs = CachingDummyInput(out_channels=2, num_repetitions=2)
        layer = RepetitionMixingLayer(inputs=inputs, out_channels=2, num_repetitions=2)

        data = torch.randn(10, 1)
        cache = Cache()

        # Add module lls but not input lls
        cache["log_likelihood"][layer] = torch.zeros(10, 1, 2, 1)

        with pytest.raises(ValueError, match="Input log-likelihoods not found in cache"):
            layer.expectation_maximization(data, cache=cache)

    def test_em_raises_without_module_lls_in_cache(self):
        """Test that EM raises ValueError when module log-likelihoods not in cache."""
        inputs = CachingDummyInput(out_channels=2, num_repetitions=2)
        layer = RepetitionMixingLayer(inputs=inputs, out_channels=2, num_repetitions=2)

        data = torch.randn(10, 1)
        cache = Cache()

        # Add input lls but not module lls
        cache["log_likelihood"][inputs] = torch.zeros(10, 1, 2, 2)

        with pytest.raises(ValueError, match="Module log-likelihoods not found in cache"):
            layer.expectation_maximization(data, cache=cache)

    def test_em_creates_cache_if_none(self):
        """Test that EM creates a cache if none is provided."""
        inputs = CachingDummyInput(out_channels=2, num_repetitions=2)
        layer = RepetitionMixingLayer(inputs=inputs, out_channels=2, num_repetitions=2)

        data = torch.randn(10, 1)

        # Should raise ValueError because cache doesn't have required lls
        with pytest.raises(ValueError, match="Input log-likelihoods not found in cache"):
            layer.expectation_maximization(data, cache=None)


class TestRepetitionMixingLayerEMPropagation:
    """Test RepetitionMixingLayer expectation_maximization propagates to children."""

    def test_em_calls_input_em(self):
        """Test that EM calls expectation_maximization on input module."""
        inputs = CachingDummyInput(out_channels=2, num_repetitions=2)
        layer = RepetitionMixingLayer(inputs=inputs, out_channels=2, num_repetitions=2)

        data = torch.randn(10, 1)
        cache = Cache()
        ll = layer.log_likelihood(data, cache=cache)

        # Set gradient
        cache["log_likelihood"][layer].grad = torch.ones_like(cache["log_likelihood"][layer])

        with patch.object(inputs, "expectation_maximization") as mock_em:
            layer.expectation_maximization(data, cache=cache)
            mock_em.assert_called_once_with(data, cache=cache)


class TestRepetitionMixingLayerEMIntegration:
    """Integration tests for RepetitionMixingLayer EM using the full EM algorithm."""

    @pytest.mark.parametrize("in_channels", in_channels_values)
    @pytest.mark.parametrize("num_reps", num_repetitions_values)
    def test_em_via_global_function(self, in_channels: int, num_reps: int):
        """Test EM through the global expectation_maximization function."""
        inputs = CachingDummyInput(out_channels=in_channels, num_repetitions=num_reps)
        layer = RepetitionMixingLayer(
            inputs=inputs, out_channels=in_channels, num_repetitions=num_reps
        )

        original_weights = layer.weights.clone()

        data = torch.randn(50, 1)
        ll_history = expectation_maximization(layer, data, max_steps=3)

        # Check that EM ran for expected steps
        assert len(ll_history) >= 1

        # Check that the log-likelihoods are finite
        assert torch.isfinite(ll_history).all()
