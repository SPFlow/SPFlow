"""Unit tests for range/interval inference utilities."""

import pytest
import torch

from spflow.meta.data import IntervalEvidence
from spflow.modules.leaves.normal import Normal
from spflow.modules.leaves.uniform import Uniform
from spflow.modules.sums.sum import Sum
from spflow.modules.products.product import Product
from spflow.utils.range_inference import log_likelihood_interval


class TestUniformIntervalProb:
    """Tests for Uniform distribution interval probability."""

    def test_uniform_interval_prob_matches_analytic(self):
        """Verify P(a <= X <= b) = (b-a)/(high-low) for Uniform."""
        # Uniform(0, 1)
        leaf = Uniform(scope=0, low=torch.tensor([0.0]), high=torch.tensor([1.0]))

        low = torch.tensor([[0.2]])
        high = torch.tensor([[0.8]])

        log_prob = log_likelihood_interval(leaf, low, high)
        prob = torch.exp(log_prob)

        # Expected: (0.8 - 0.2) / (1.0 - 0.0) = 0.6
        expected = torch.tensor([[[[0.6]]]])
        torch.testing.assert_close(prob, expected, rtol=1e-5, atol=1e-5)

    def test_uniform_full_interval(self):
        """Full interval should give probability 1."""
        leaf = Uniform(scope=0, low=torch.tensor([0.0]), high=torch.tensor([1.0]))

        low = torch.tensor([[0.0]])
        high = torch.tensor([[1.0]])

        log_prob = log_likelihood_interval(leaf, low, high)
        prob = torch.exp(log_prob)

        expected = torch.tensor([[[[1.0]]]])
        torch.testing.assert_close(prob, expected, rtol=1e-5, atol=1e-5)

    def test_uniform_interval_outside_support(self):
        """Interval completely outside support should give probability 0."""
        leaf = Uniform(scope=0, low=torch.tensor([0.0]), high=torch.tensor([1.0]))

        low = torch.tensor([[2.0]])
        high = torch.tensor([[3.0]])

        log_prob = log_likelihood_interval(leaf, low, high)
        # Should be -inf (log of 0)
        assert torch.isinf(log_prob).all() and (log_prob < 0).all()

    def test_uniform_interval_partial_overlap(self):
        """Interval partially overlapping support."""
        leaf = Uniform(scope=0, low=torch.tensor([0.0]), high=torch.tensor([1.0]))

        # Query interval [0.5, 1.5] overlaps [0.5, 1.0]
        low = torch.tensor([[0.5]])
        high = torch.tensor([[1.5]])

        log_prob = log_likelihood_interval(leaf, low, high)
        prob = torch.exp(log_prob)

        # Expected: (1.0 - 0.5) / (1.0 - 0.0) = 0.5
        expected = torch.tensor([[[[0.5]]]])
        torch.testing.assert_close(prob, expected, rtol=1e-5, atol=1e-5)


class TestNormalIntervalProb:
    """Tests for Normal distribution interval probability."""

    def test_normal_interval_prob_matches_cdf(self):
        """Compare against torch.distributions.Normal.cdf."""
        loc = torch.tensor([0.0])
        scale = torch.tensor([1.0])
        leaf = Normal(scope=0, loc=loc, scale=scale)

        low = torch.tensor([[-1.0]])
        high = torch.tensor([[1.0]])

        log_prob = log_likelihood_interval(leaf, low, high)
        prob = torch.exp(log_prob)

        # Use torch.distributions for reference
        dist = torch.distributions.Normal(loc, scale)
        expected_prob = dist.cdf(high.squeeze()) - dist.cdf(low.squeeze())

        # squeeze() removes all size-1 dims, so we compare scalars
        torch.testing.assert_close(prob.view(-1)[0], expected_prob.view(-1)[0], rtol=1e-5, atol=1e-5)

    def test_normal_symmetric_interval(self):
        """Symmetric interval around mean for standard normal."""
        leaf = Normal(scope=0, loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))

        low = torch.tensor([[-2.0]])
        high = torch.tensor([[2.0]])

        log_prob = log_likelihood_interval(leaf, low, high)
        prob = torch.exp(log_prob)

        # P(-2 <= X <= 2) ≈ 0.9545 for standard normal
        expected = torch.tensor(0.9545)
        torch.testing.assert_close(prob.view(-1)[0], expected, rtol=1e-3, atol=1e-3)

    def test_normal_one_sided_interval(self):
        """One-sided interval using NaN for unbounded side."""
        leaf = Normal(scope=0, loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))

        # P(X <= 0) for standard normal should be 0.5
        low = torch.tensor([[float("nan")]])
        high = torch.tensor([[0.0]])

        log_prob = log_likelihood_interval(leaf, low, high)
        prob = torch.exp(log_prob)

        expected = torch.tensor([[[[0.5]]]])
        torch.testing.assert_close(prob, expected, rtol=1e-5, atol=1e-5)


class TestCircuitIntervalProb:
    """Tests for interval probability through Sum/Product circuits."""

    def test_sum_circuit_interval_prob(self):
        """Test interval probability through Sum node."""
        # Two Uniform leaves with different supports
        leaf1 = Uniform(scope=0, low=torch.tensor([0.0]), high=torch.tensor([1.0]))
        leaf2 = Uniform(scope=0, low=torch.tensor([0.5]), high=torch.tensor([1.5]))

        # Equal weight mixture
        weights = torch.tensor([0.5, 0.5])
        circuit = Sum(inputs=[leaf1, leaf2], weights=weights)

        # Query interval [0.6, 0.9]
        low = torch.tensor([[0.6]])
        high = torch.tensor([[0.9]])

        log_prob = log_likelihood_interval(circuit, low, high)
        prob = torch.exp(log_prob)

        # leaf1: (0.9 - 0.6) / 1.0 = 0.3
        # leaf2: (0.9 - 0.6) / 1.0 = 0.3
        # Mixture: 0.5 * 0.3 + 0.5 * 0.3 = 0.3
        expected = torch.tensor([[[[0.3]]]])
        torch.testing.assert_close(prob, expected, rtol=1e-5, atol=1e-5)

    def test_product_circuit_interval_prob(self):
        """Test interval probability through Product node (independent variables)."""
        # Two independent Uniform leaves over different scopes
        leaf1 = Uniform(scope=0, low=torch.tensor([0.0]), high=torch.tensor([1.0]))
        leaf2 = Uniform(scope=1, low=torch.tensor([0.0]), high=torch.tensor([1.0]))

        circuit = Product(inputs=[leaf1, leaf2])

        # Query: X0 in [0.2, 0.8], X1 in [0.3, 0.7]
        low = torch.tensor([[0.2, 0.3]])
        high = torch.tensor([[0.8, 0.7]])

        log_prob = log_likelihood_interval(circuit, low, high)
        prob = torch.exp(log_prob)

        # P(X0 in [0.2, 0.8]) * P(X1 in [0.3, 0.7]) = 0.6 * 0.4 = 0.24
        expected = torch.tensor([[[[0.24]]]])
        torch.testing.assert_close(prob, expected, rtol=1e-5, atol=1e-5)


class TestNanBounds:
    """Tests for NaN bounds semantics."""

    def test_nan_low_means_no_lower_bound(self):
        """NaN in low should be treated as -inf."""
        leaf = Uniform(scope=0, low=torch.tensor([0.0]), high=torch.tensor([1.0]))

        # P(X <= 0.5) for Uniform(0, 1) = 0.5
        low = torch.tensor([[float("nan")]])
        high = torch.tensor([[0.5]])

        log_prob = log_likelihood_interval(leaf, low, high)
        prob = torch.exp(log_prob)

        expected = torch.tensor([[[[0.5]]]])
        torch.testing.assert_close(prob, expected, rtol=1e-5, atol=1e-5)

    def test_nan_high_means_no_upper_bound(self):
        """NaN in high should be treated as +inf."""
        leaf = Uniform(scope=0, low=torch.tensor([0.0]), high=torch.tensor([1.0]))

        # P(X >= 0.5) for Uniform(0, 1) = 0.5
        low = torch.tensor([[0.5]])
        high = torch.tensor([[float("nan")]])

        log_prob = log_likelihood_interval(leaf, low, high)
        prob = torch.exp(log_prob)

        expected = torch.tensor([[[[0.5]]]])
        torch.testing.assert_close(prob, expected, rtol=1e-5, atol=1e-5)

    def test_both_nan_means_full_support(self):
        """Both NaN should give probability 1 (full marginalization)."""
        leaf = Uniform(scope=0, low=torch.tensor([0.0]), high=torch.tensor([1.0]))

        low = torch.tensor([[float("nan")]])
        high = torch.tensor([[float("nan")]])

        log_prob = log_likelihood_interval(leaf, low, high)
        prob = torch.exp(log_prob)

        expected = torch.tensor([[[[1.0]]]])
        torch.testing.assert_close(prob, expected, rtol=1e-5, atol=1e-5)


class TestInvalidIntervals:
    """Tests for validation of invalid intervals."""

    def test_low_greater_than_high_raises(self):
        """low > high should raise ValueError."""
        low = torch.tensor([[0.8]])
        high = torch.tensor([[0.2]])

        with pytest.raises(ValueError, match="low > high"):
            IntervalEvidence(low=low, high=high)

    def test_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        low = torch.tensor([[0.2]])
        high = torch.tensor([[0.8, 0.9]])  # Wrong shape

        with pytest.raises(ValueError, match="same shape"):
            IntervalEvidence(low=low, high=high)

    def test_wrong_dimension_raises(self):
        """Non-2D tensors should raise ValueError."""
        low = torch.tensor([0.2])  # 1D instead of 2D
        high = torch.tensor([0.8])

        with pytest.raises(ValueError, match="2-dimensional"):
            IntervalEvidence(low=low, high=high)


class TestDirectIntervalEvidence:
    """Tests for using IntervalEvidence directly with log_likelihood."""

    def test_interval_evidence_direct_call(self):
        """Test calling log_likelihood directly with IntervalEvidence."""
        leaf = Uniform(scope=0, low=torch.tensor([0.0]), high=torch.tensor([1.0]))

        evidence = IntervalEvidence(low=torch.tensor([[0.2]]), high=torch.tensor([[0.8]]))
        log_prob = leaf.log_likelihood(evidence)
        prob = torch.exp(log_prob)

        expected = torch.tensor([[[[0.6]]]])
        torch.testing.assert_close(prob, expected, rtol=1e-5, atol=1e-5)


class TestBatchProcessing:
    """Tests for batch processing."""

    def test_batch_uniform(self):
        """Test batch processing for Uniform."""
        leaf = Uniform(scope=0, low=torch.tensor([0.0]), high=torch.tensor([1.0]))

        # 3 different queries
        low = torch.tensor([[0.0], [0.2], [0.4]])
        high = torch.tensor([[0.5], [0.7], [0.9]])

        log_prob = log_likelihood_interval(leaf, low, high)
        prob = torch.exp(log_prob)

        # Expected: [0.5, 0.5, 0.5]
        expected = torch.tensor([[[[0.5]]], [[[0.5]]], [[[0.5]]]])
        torch.testing.assert_close(prob, expected, rtol=1e-5, atol=1e-5)


class TestCategoricalIntervalProb:
    """Tests for Categorical distribution interval probability."""

    def test_categorical_interval_prob(self):
        """Verify interval sums probabilities of categories."""
        from spflow.modules.leaves.categorical import Categorical

        # Probs: [0.1, 0.2, 0.3, 0.4]
        probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
        probs = probs.view(1, 1, 1, 4)  # Add dimensions

        leaf = Categorical(scope=0, probs=probs)

        # Query: categories in [1, 2] -> sum(0.2, 0.3) = 0.5
        low = torch.tensor([[1.0]])
        high = torch.tensor([[2.0]])

        log_prob = log_likelihood_interval(leaf, low, high)
        prob = torch.exp(log_prob)

        expected = torch.tensor([[[[0.5]]]])
        torch.testing.assert_close(prob, expected, rtol=1e-5, atol=1e-5)

    def test_categorical_interval_nan_bounds(self):
        """Test NaN bounds for Categorical."""
        from spflow.modules.leaves.categorical import Categorical

        probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
        probs = probs.view(1, 1, 1, 4)
        leaf = Categorical(scope=0, probs=probs)

        # Query: low=NaN, high=1.0 -> categories [0, 1] -> 0.1 + 0.2 = 0.3
        low = torch.tensor([[float("nan")]])
        high = torch.tensor([[1.0]])

        log_prob = log_likelihood_interval(leaf, low, high)
        prob = torch.exp(log_prob)

        torch.testing.assert_close(prob, torch.tensor([[[[0.3]]]]), rtol=1e-5, atol=1e-5)


class TestBernoulliIntervalProb:
    """Tests for Bernoulli distribution interval probability."""

    def test_bernoulli_interval_prob(self):
        """Verify interval sums probabilities."""
        from spflow.modules.leaves.bernoulli import Bernoulli

        # p = 0.7
        probs = torch.tensor([0.7])
        probs = probs.view(1, 1, 1)
        leaf = Bernoulli(scope=0, probs=probs)

        # Query: low=0.5, high=1.5. Includes 1 (prob 0.7), excludes 0.
        low = torch.tensor([[0.5]])
        high = torch.tensor([[1.5]])

        log_prob = log_likelihood_interval(leaf, low, high)
        prob = torch.exp(log_prob)

        # Expected: 0.7
        torch.testing.assert_close(prob, torch.tensor([[[[0.7]]]]), rtol=1e-5, atol=1e-5)

        # Query: low=-0.5, high=0.5. Includes 0 (prob 0.3).
        low = torch.tensor([[-0.5]])
        high = torch.tensor([[0.5]])

        log_prob = log_likelihood_interval(leaf, low, high)
        prob = torch.exp(log_prob)

        torch.testing.assert_close(prob, torch.tensor([[[[0.3]]]]), rtol=1e-5, atol=1e-5)

    def test_bernoulli_full_interval(self):
        """Verify full interval [0, 1] gives probability 1."""
        from spflow.modules.leaves.bernoulli import Bernoulli

        leaf = Bernoulli(scope=0, probs=torch.tensor([[[0.7]]]))

        low = torch.tensor([[0.0]])
        high = torch.tensor([[1.0]])

        log_prob = log_likelihood_interval(leaf, low, high)
        prob = torch.exp(log_prob)

        torch.testing.assert_close(prob, torch.tensor([[[[1.0]]]]), rtol=1e-5, atol=1e-5)


class TestExponentialIntervalProb:
    """Tests for Exponential distribution (generic CDF implementation)."""

    def test_exponential_interval_prob(self):
        """Verify interval probability matches analytic formula."""
        from spflow.modules.leaves.exponential import Exponential

        # rate (lambda) = 2.0
        # CDF(x) = 1 - exp(-2x)
        leaf = Exponential(scope=0, rate=torch.tensor([2.0]))

        low = torch.tensor([[0.5]])
        high = torch.tensor([[1.0]])

        log_prob = log_likelihood_interval(leaf, low, high)
        prob = torch.exp(log_prob)

        # P(0.5 <= X <= 1.0) = (1 - exp(-2)) - (1 - exp(-1)) = exp(-1) - exp(-2)
        # exp(-1) ≈ 0.367879
        # exp(-2) ≈ 0.135335
        # diff ≈ 0.232544
        expected_val = torch.exp(torch.tensor(-1.0)) - torch.exp(torch.tensor(-2.0))
        expected = expected_val.view(1, 1, 1, 1)

        torch.testing.assert_close(prob.view(-1), expected.view(-1), rtol=1e-5, atol=1e-5)
