"""Tests for WeightedSum module."""

import pytest
import torch

from spflow.meta.data.scope import Scope
from spflow.modules.leaves.normal import Normal
from spflow.exp.pic.weighted_sum import WeightedSum


class TestWeightedSumInit:
    """Tests for WeightedSum initialization."""

    def test_basic_init(self):
        """Test basic initialization with a single input."""
        leaf = Normal(scope=Scope([0]), out_channels=3)
        weights = torch.ones(1, 3, 2, 1)  # (F, IC, OC, R)

        ws = WeightedSum(inputs=leaf, weights=weights)

        assert ws.out_shape.channels == 2
        assert ws.out_shape.features == 1
        assert ws.out_shape.repetitions == 1

    def test_init_1d_weights(self):
        """Test initialization with 1D weights (broadcasts to 4D)."""
        leaf = Normal(scope=Scope([0]), out_channels=4)
        weights = torch.ones(4)  # Will become (1, 4, 1, 1)

        ws = WeightedSum(inputs=leaf, weights=weights)

        assert ws._weights.shape == (1, 4, 1, 1)

    def test_init_2d_weights(self):
        """Test initialization with 2D weights."""
        leaf = Normal(scope=Scope([0]), out_channels=3)
        weights = torch.ones(3, 2)  # Will become (1, 3, 2, 1)

        ws = WeightedSum(inputs=leaf, weights=weights)

        assert ws._weights.shape == (1, 3, 2, 1)

    def test_init_multiple_inputs(self):
        """Test initialization with multiple inputs (will be concatenated)."""
        leaf1 = Normal(scope=Scope([0]), out_channels=2)
        leaf2 = Normal(scope=Scope([0]), out_channels=2)
        weights = torch.ones(1, 4, 1, 1)  # 4 = 2 + 2 (concatenated)

        ws = WeightedSum(inputs=[leaf1, leaf2], weights=weights)

        assert ws.in_shape.channels == 4  # Concatenated

    def test_init_empty_inputs_raises(self):
        """Test that empty inputs raises ValueError."""
        with pytest.raises(ValueError, match="requires at least one input"):
            WeightedSum(inputs=[], weights=torch.ones(1))

    def test_init_invalid_weight_dim_raises(self):
        """Test that 5D+ weights raises ShapeError."""
        from spflow.exceptions import ShapeError

        leaf = Normal(scope=Scope([0]), out_channels=2)
        weights = torch.ones(1, 1, 1, 1, 1)  # 5D

        with pytest.raises(ShapeError, match="must be 1D, 2D, 3D, or 4D"):
            WeightedSum(inputs=leaf, weights=weights)


class TestWeightedSumNoNormalization:
    """Tests verifying weights are NOT normalized."""

    def test_weights_preserved_exactly(self):
        """Test that weights are stored exactly as provided."""
        leaf = Normal(scope=Scope([0]), out_channels=3)
        weights = torch.tensor([0.1, 0.2, 0.9]).view(1, 3, 1, 1)  # (F, IC, OC, R)

        ws = WeightedSum(inputs=leaf, weights=weights)

        # Weights should NOT be normalized to sum to 1 per channel
        assert torch.allclose(ws.weights, weights)
        # These weights don't sum to 1 and that should be fine
        assert not torch.allclose(ws.weights.sum(dim=1), torch.ones(1, 1, 1))

    def test_unnormalized_weights_allowed(self):
        """Test that quadrature-style weights (not summing to 1) work."""
        leaf = Normal(scope=Scope([0]), out_channels=5)
        # Quadrature weights example (e.g., Gauss-Legendre)
        quadrature_weights = torch.tensor([0.2369, 0.4786, 0.5688, 0.4786, 0.2369])
        ws_weights = quadrature_weights.view(1, 5, 1, 1)

        ws = WeightedSum(inputs=leaf, weights=ws_weights)

        # These sum to ~2.0, not 1.0, and should be preserved
        assert torch.allclose(ws.weights.sum(), torch.tensor(2.0), atol=0.01)

    def test_log_weights_is_log_of_raw(self):
        """Test that log_weights returns log of raw weights, not log-softmax."""
        leaf = Normal(scope=Scope([0]), out_channels=3)
        weights = torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1)

        ws = WeightedSum(inputs=leaf, weights=weights)

        expected_log_weights = torch.log(weights)
        assert torch.allclose(ws.log_weights, expected_log_weights)

    def test_log_weights_allows_structural_zeros(self):
        """Test that structural zeros produce -inf log-weights (needed for sparse mixing matrices)."""
        leaf = Normal(scope=Scope([0]), out_channels=3)
        weights = torch.tensor([1.0, 0.0, 2.0]).view(1, 3, 1, 1)  # (F, IC, OC, R)

        ws = WeightedSum(inputs=leaf, weights=weights)

        assert torch.isneginf(ws.log_weights).any()


class TestWeightedSumLogLikelihood:
    """Tests for log-likelihood computation."""

    def test_log_likelihood_shape(self):
        """Test log-likelihood output shape."""
        leaf = Normal(scope=Scope([0, 1]), out_channels=3)
        weights = torch.ones(2, 3, 2, 1)  # 2 features, 3 in_channels, 2 out_channels

        ws = WeightedSum(inputs=leaf, weights=weights)

        data = torch.randn(10, 2)  # batch=10, features=2
        ll = ws.log_likelihood(data)

        assert ll.shape == (10, 2, 2, 1)  # (batch, features, out_channels, repetitions)

    def test_log_likelihood_uses_raw_weights(self):
        """Test that log-likelihood uses unnormalized weights."""
        # Create simple setup
        leaf = Normal(scope=Scope([0]), out_channels=1)
        # Weight of 2.0 (not normalized)
        weights = torch.tensor([[[[2.0]]]])

        ws = WeightedSum(inputs=leaf, weights=weights)

        data = torch.tensor([[0.0]])
        ll = ws.log_likelihood(data)

        # log(2 * p(x)) = log(2) + log(p(x))
        leaf_ll = leaf.log_likelihood(data)
        expected = torch.logsumexp(leaf_ll.unsqueeze(3) + torch.log(weights), dim=2)

        assert torch.allclose(ll, expected, atol=1e-5)


class TestWeightedSumSetWeights:
    """Tests for setting weights."""

    def test_set_weights(self):
        """Test setting weights directly."""
        leaf = Normal(scope=Scope([0]), out_channels=3)
        weights = torch.ones(1, 3, 1, 1)

        ws = WeightedSum(inputs=leaf, weights=weights)

        new_weights = torch.full((1, 3, 1, 1), 0.5)
        ws.weights = new_weights

        assert torch.allclose(ws.weights, new_weights)

    def test_set_weights_wrong_shape_raises(self):
        """Test that setting weights with wrong shape raises error."""
        from spflow.exceptions import ShapeError

        leaf = Normal(scope=Scope([0]), out_channels=3)
        weights = torch.ones(1, 3, 1, 1)

        ws = WeightedSum(inputs=leaf, weights=weights)

        with pytest.raises(ShapeError, match="Invalid shape"):
            ws.weights = torch.ones(1, 2, 1, 1)  # Wrong shape
