"""Unit tests for SPFlow inference utilities."""

import pytest
import torch

from spflow.utils.inference import log_posterior


class TestLogPosterior:
    """Tests for log_posterior function."""

    def test_basic_log_posterior(self):
        """Test basic log posterior computation with simple values."""
        # log_likelihood: [batch_size, num_classes]
        log_likelihood = torch.log(torch.tensor([[0.6, 0.4], [0.3, 0.7]]))
        # log_prior: [num_classes]
        log_prior = torch.log(torch.tensor([0.5, 0.5]))

        result = log_posterior(log_likelihood, log_prior)

        # Result should have same shape as log_likelihood
        assert result.shape == log_likelihood.shape

        # With uniform prior, posterior should equal likelihood
        # (after normalization, which logsumexp handles)
        probs = torch.exp(result)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_uniform_prior(self):
        """Test that uniform prior maintains likelihood proportions."""
        log_likelihood = torch.log(torch.tensor([[0.8, 0.2]]))
        log_prior = torch.log(torch.tensor([0.5, 0.5]))

        result = log_posterior(log_likelihood, log_prior)
        probs = torch.exp(result)

        # With uniform prior, posterior should equal normalized likelihood
        expected = torch.tensor([[0.8, 0.2]])
        assert torch.allclose(probs, expected, atol=1e-5)

    def test_non_uniform_prior(self):
        """Test posterior computation with non-uniform prior."""
        # Equal likelihoods
        log_likelihood = torch.log(torch.tensor([[0.5, 0.5]]))
        # Prior favoring class 0
        log_prior = torch.log(torch.tensor([0.9, 0.1]))

        result = log_posterior(log_likelihood, log_prior)
        probs = torch.exp(result)

        # Posterior should reflect prior when likelihoods are equal
        assert probs[0, 0] > probs[0, 1]
        assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-5)

    def test_multiple_samples(self):
        """Test with multiple samples in batch."""
        batch_size = 5
        num_classes = 3

        log_likelihood = torch.randn(batch_size, num_classes)
        log_prior = torch.log_softmax(torch.randn(num_classes), dim=0)

        result = log_posterior(log_likelihood, log_prior)

        assert result.shape == (batch_size, num_classes)
        # Each row should be a valid log probability (sums to 1 in probability space)
        probs = torch.exp(result)
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)

    def test_output_is_normalized(self):
        """Test that output probabilities sum to 1."""
        log_likelihood = torch.randn(10, 4)
        log_prior = torch.randn(4)

        result = log_posterior(log_likelihood, log_prior)
        probs = torch.exp(result)

        # Each sample's posterior should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(10), atol=1e-5)

    def test_gradient_flow(self):
        """Test that gradients flow through the computation."""
        log_likelihood = torch.randn(3, 2, requires_grad=True)
        log_prior = torch.randn(2, requires_grad=True)

        result = log_posterior(log_likelihood, log_prior)
        loss = result.sum()
        loss.backward()

        assert log_likelihood.grad is not None
        assert log_prior.grad is not None

    def test_single_class(self):
        """Test with single class (edge case)."""
        log_likelihood = torch.tensor([[0.0], [0.0]])
        log_prior = torch.tensor([0.0])

        result = log_posterior(log_likelihood, log_prior)

        # With single class, posterior is always 1.0 (log = 0.0)
        assert torch.allclose(result, torch.zeros(2, 1), atol=1e-5)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very negative log likelihoods
        log_likelihood = torch.tensor([[-1000.0, -1001.0], [-1001.0, -1000.0]])
        log_prior = torch.tensor([0.0, 0.0])

        result = log_posterior(log_likelihood, log_prior)

        # Should not produce NaN or Inf
        assert torch.isfinite(result).all()

        # Probabilities should still sum to 1 (use larger tolerance for extreme values)
        probs = torch.exp(result)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-3)

    @pytest.mark.parametrize(
        "batch_size,num_classes",
        [(1, 2), (10, 3), (100, 10), (1, 1)],
    )
    def test_various_shapes(self, batch_size: int, num_classes: int):
        """Test with various input shapes."""
        log_likelihood = torch.randn(batch_size, num_classes)
        log_prior = torch.randn(num_classes)

        result = log_posterior(log_likelihood, log_prior)

        assert result.shape == (batch_size, num_classes)
        probs = torch.exp(result)
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)
