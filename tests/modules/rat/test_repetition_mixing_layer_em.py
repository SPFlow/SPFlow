"""Tests for RepetitionMixingLayer expectation_maximization method."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from spflow.exceptions import MissingCacheError
from spflow.learn import expectation_maximization
from spflow.meta import Scope
from spflow.modules.sums.repetition_mixing_layer import RepetitionMixingLayer
from spflow.utils.cache import Cache, cached
from tests.utils.leaves import CachingDummyInput


# Tiny sweep is enough to cover shape-dependent EM branches.
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
        layer = RepetitionMixingLayer(inputs=inputs, out_channels=in_channels, num_repetitions=num_reps)

        original_weights = layer.weights.clone()

        data = torch.randn(50, out_features)
        cache = Cache()
        ll = layer.log_likelihood(data, cache=cache)

        # Simulate parent responsibilities so this test isolates local EM math.
        cache["log_likelihood"][layer].grad = torch.ones_like(cache["log_likelihood"][layer])

        # Mock child updates to keep the assertion focused on layer weights.
        with patch.object(inputs, "_expectation_maximization_step"):
            layer._expectation_maximization_step(data, cache=cache)

        assert not torch.allclose(layer.weights, original_weights, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("in_channels", in_channels_values)
    @pytest.mark.parametrize("num_reps", num_repetitions_values)
    def test_em_weights_still_normalized(self, in_channels: int, num_reps: int):
        """Test that weights are still normalized after EM."""
        inputs = CachingDummyInput(out_channels=in_channels, num_repetitions=num_reps)
        layer = RepetitionMixingLayer(inputs=inputs, out_channels=in_channels, num_repetitions=num_reps)

        data = torch.randn(50, 1)
        cache = Cache()
        ll = layer.log_likelihood(data, cache=cache)

        cache["log_likelihood"][layer].grad = torch.ones_like(cache["log_likelihood"][layer])

        with patch.object(inputs, "_expectation_maximization_step"):
            layer._expectation_maximization_step(data, cache=cache)

        # EM must keep normalized mixing weights over the repetition axis.
        weights_sum = layer.weights.sum(dim=layer.sum_dim)
        torch.testing.assert_close(weights_sum, torch.ones_like(weights_sum), rtol=1e-5, atol=1e-5)


class TestRepetitionMixingLayerEMErrors:
    """Test RepetitionMixingLayer expectation_maximization error handling."""

    def test_em_raises_without_input_lls_in_cache(self):
        """Test that EM raises ValueError when input log-likelihoods not in cache."""
        inputs = CachingDummyInput(out_channels=2, num_repetitions=2)
        layer = RepetitionMixingLayer(inputs=inputs, out_channels=2, num_repetitions=2)

        data = torch.randn(10, 1)
        cache = Cache()

        # Missing child likelihoods should fail fast instead of silently skipping EM.
        cache["log_likelihood"][layer] = torch.zeros(10, 1, 2, 1)

        with pytest.raises(ValueError):
            layer._expectation_maximization_step(data, cache=cache)

    def test_em_raises_without_module_lls_in_cache(self):
        """Test that EM raises ValueError when module log-likelihoods not in cache."""
        inputs = CachingDummyInput(out_channels=2, num_repetitions=2)
        layer = RepetitionMixingLayer(inputs=inputs, out_channels=2, num_repetitions=2)

        data = torch.randn(10, 1)
        cache = Cache()

        # Missing parent likelihoods should fail fast for the same reason.
        cache["log_likelihood"][inputs] = torch.zeros(10, 1, 2, 2)

        with pytest.raises(ValueError):
            layer._expectation_maximization_step(data, cache=cache)

    def test_em_requires_cached_lls(self):
        """Test that EM fails when cache has no required log-likelihood tensors."""
        inputs = CachingDummyInput(out_channels=2, num_repetitions=2)
        layer = RepetitionMixingLayer(inputs=inputs, out_channels=2, num_repetitions=2)

        data = torch.randn(10, 1)
        cache = Cache()

        with pytest.raises(MissingCacheError):
            layer._expectation_maximization_step(data, cache=cache)


class TestRepetitionMixingLayerEMPropagation:
    """Test RepetitionMixingLayer expectation_maximization propagates to children."""

    def test_em_calls_input_em(self):
        """Test that EM calls expectation_maximization on input module."""
        inputs = CachingDummyInput(out_channels=2, num_repetitions=2)
        layer = RepetitionMixingLayer(inputs=inputs, out_channels=2, num_repetitions=2)

        data = torch.randn(10, 1)
        cache = Cache()
        ll = layer.log_likelihood(data, cache=cache)

        cache["log_likelihood"][layer].grad = torch.ones_like(cache["log_likelihood"][layer])

        with patch.object(inputs, "_expectation_maximization_step") as mock_em:
            layer._expectation_maximization_step(data, cache=cache)
            mock_em.assert_called_once_with(data, bias_correction=True, cache=cache)


class TestRepetitionMixingLayerEMIntegration:
    """Integration tests for RepetitionMixingLayer EM using the full EM algorithm."""

    @pytest.mark.parametrize("in_channels", in_channels_values)
    @pytest.mark.parametrize("num_reps", num_repetitions_values)
    def test_em_via_global_function(self, in_channels: int, num_reps: int):
        """Test EM through the global expectation_maximization function."""
        inputs = CachingDummyInput(out_channels=in_channels, num_repetitions=num_reps)
        layer = RepetitionMixingLayer(inputs=inputs, out_channels=in_channels, num_repetitions=num_reps)

        original_weights = layer.weights.clone()

        data = torch.randn(50, 1)
        ll_history = expectation_maximization(layer, data, max_steps=3)

        assert len(ll_history) >= 1

        # Finite history confirms the full EM loop stayed numerically stable.
        assert torch.isfinite(ll_history).all()
