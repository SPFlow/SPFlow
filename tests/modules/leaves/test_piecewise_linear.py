"""Tests for PiecewiseLinear leaf distribution module."""

import pytest
import torch

from spflow.meta.data import Scope
from spflow.modules.leaves.piecewise_linear import PiecewiseLinear, PiecewiseLinearDist
from spflow.utils.domain import DataType, Domain


class TestPiecewiseLinearInitialization:
    """Test initialization of PiecewiseLinear leaf."""

    @pytest.mark.parametrize("num_repetitions", [1, 2])
    @pytest.mark.parametrize("out_channels", [1, 3])
    def test_initialization_continuous(self, num_repetitions, out_channels):
        """Test initialization with continuous data."""
        torch.manual_seed(42)
        scope = Scope([0, 1])
        leaf = PiecewiseLinear(scope=scope, out_channels=out_channels, num_repetitions=num_repetitions)

        # Generate synthetic data
        data = torch.randn(100, 2)
        domains = [
            Domain.continuous_inf_support(),
            Domain.continuous_inf_support(),
        ]

        leaf.initialize(data, domains)

        assert leaf.is_initialized
        assert leaf.xs is not None
        assert leaf.ys is not None
        assert len(leaf.xs) == num_repetitions
        assert len(leaf.xs[0]) == out_channels  # num_leaves (out_channels)
        assert len(leaf.xs[0][0]) == 2  # num_features

    def test_initialization_discrete(self):
        """Test initialization with discrete data."""
        torch.manual_seed(42)
        scope = Scope([0])
        leaf = PiecewiseLinear(scope=scope, out_channels=1, num_repetitions=1)

        # Generate synthetic discrete data (0-9)
        data = torch.randint(0, 10, (100, 1)).float()
        domains = [Domain.discrete_range(0, 9)]

        leaf.initialize(data, domains)

        assert leaf.is_initialized
        assert leaf.xs is not None

    def test_initialization_mixed(self):
        """Test initialization with mixed continuous and discrete data."""
        torch.manual_seed(42)
        scope = Scope([0, 1])
        leaf = PiecewiseLinear(scope=scope, out_channels=1, num_repetitions=1)

        # Generate mixed data
        continuous_data = torch.randn(100, 1)
        discrete_data = torch.randint(0, 5, (100, 1)).float()
        data = torch.cat([continuous_data, discrete_data], dim=1)

        domains = [
            Domain.continuous_inf_support(),
            Domain.discrete_range(0, 4),
        ]

        leaf.initialize(data, domains)

        assert leaf.is_initialized

    def test_uninitialized_raises(self):
        """Test that operations on uninitialized leaf raise errors."""
        scope = Scope([0])
        leaf = PiecewiseLinear(scope=scope)

        with pytest.raises(ValueError, match="not been initialized"):
            _ = leaf.distribution

        with pytest.raises(ValueError, match="not been initialized"):
            leaf.log_likelihood(torch.randn(10, 1))

        with pytest.raises(ValueError, match="not been initialized"):
            leaf.sample(num_samples=10, data=torch.full((10, 1), float("nan")))

    def test_reset(self):
        """Test reset functionality."""
        torch.manual_seed(42)
        scope = Scope([0])
        leaf = PiecewiseLinear(scope=scope, out_channels=1)

        data = torch.randn(100, 1)
        domains = [Domain.continuous_inf_support()]

        leaf.initialize(data, domains)
        assert leaf.is_initialized

        leaf.reset()
        assert not leaf.is_initialized
        assert leaf.xs is None
        assert leaf.ys is None


class TestPiecewiseLinearLogLikelihood:
    """Test log-likelihood computation."""

    @pytest.mark.parametrize("num_repetitions", [1, 2])
    @pytest.mark.parametrize("out_channels", [1, 3])
    def test_log_likelihood_shape(self, num_repetitions, out_channels):
        """Test that log-likelihood has correct output shape."""
        torch.manual_seed(42)
        scope = Scope([0, 1])
        leaf = PiecewiseLinear(scope=scope, out_channels=out_channels, num_repetitions=num_repetitions)

        data = torch.randn(100, 2)
        domains = [Domain.continuous_inf_support(), Domain.continuous_inf_support()]
        leaf.initialize(data, domains)

        test_data = torch.randn(20, 2)
        log_prob = leaf.log_likelihood(test_data)

        # Expected shape: (batch, features, channels, leaves, repetitions)
        assert log_prob.shape == (20, 1, 2, out_channels, num_repetitions)

    def test_log_likelihood_values(self):
        """Test that log-likelihoods are valid (finite)."""
        torch.manual_seed(42)
        scope = Scope([0])
        leaf = PiecewiseLinear(scope=scope, out_channels=1, num_repetitions=1)

        data = torch.randn(1000, 1)
        domains = [Domain.continuous_inf_support()]
        leaf.initialize(data, domains)

        # Test on data within the training range
        test_data = torch.randn(50, 1)
        log_prob = leaf.log_likelihood(test_data)

        # Log probs should be finite
        assert torch.isfinite(log_prob).all()

    def test_log_likelihood_marginalization(self):
        """Test that NaN values are marginalized correctly."""
        torch.manual_seed(42)
        scope = Scope([0, 1])
        leaf = PiecewiseLinear(scope=scope, out_channels=1, num_repetitions=1)

        data = torch.randn(100, 2)
        domains = [Domain.continuous_inf_support(), Domain.continuous_inf_support()]
        leaf.initialize(data, domains)

        # Create test data with NaN (marginalized) values
        test_data = torch.randn(10, 2)
        test_data[0, 0] = float("nan")
        test_data[5, 1] = float("nan")

        log_prob = leaf.log_likelihood(test_data)

        # Should still produce valid output
        assert not torch.isnan(log_prob).any()


class TestPiecewiseLinearDist:
    """Test PiecewiseLinearDist distribution class."""

    def test_log_prob(self):
        """Test log_prob computation."""
        torch.manual_seed(42)
        # Create a simple piecewise linear distribution
        xs = [[[[torch.tensor([-1.0, 0.0, 1.0, 2.0])]]]]  # [R][L][F][C]
        ys = [[[[torch.tensor([0.0, 0.5, 0.5, 0.0])]]]]
        domains = [Domain.continuous_inf_support()]

        dist = PiecewiseLinearDist(xs, ys, domains)

        x = torch.tensor([[0.5]]).unsqueeze(1)  # [N, C, F]
        log_prob = dist.log_prob(x)

        assert log_prob.shape == (1, 1, 1, 1, 1)
        assert torch.isfinite(log_prob).all()

    def test_mode(self):
        """Test mode computation."""
        xs = [[[[torch.tensor([-1.0, 0.0, 1.0, 2.0])]]]]
        ys = [[[[torch.tensor([0.0, 0.2, 0.8, 0.0])]]]]  # Mode should be at x=1.0
        domains = [Domain.continuous_inf_support()]

        dist = PiecewiseLinearDist(xs, ys, domains)
        mode = dist.mode

        assert mode.shape == (1, 1, 1, 1)  # [C, F, L, R]
        # Mode should be at x=1.0 (highest density)
        assert torch.isclose(mode[0, 0, 0, 0], torch.tensor(1.0))


class TestPiecewiseLinearSampling:
    """Test sampling functionality."""

    @pytest.mark.parametrize("num_repetitions", [1, 2])
    @pytest.mark.parametrize("out_channels", [1, 3])
    def test_sample_shape(self, num_repetitions, out_channels):
        """Test that sampled outputs have correct shape."""
        torch.manual_seed(42)
        scope = Scope([0, 1])
        leaf = PiecewiseLinear(scope=scope, out_channels=out_channels, num_repetitions=num_repetitions)

        data = torch.randn(100, 2)
        domains = [Domain.continuous_inf_support(), Domain.continuous_inf_support()]
        leaf.initialize(data, domains)
        
        # Manually set repetition index if needed (normally handled by SamplingContext)
        # However, for direct .sample() call on leaf, usually context is passed or default created
        # If repetitions > 1, we need to ensure repetition_idx is set in default context or manually
        # The sample method handles default context creation.
        # But if repetitions > 1, the sample method implementation in PiecewiseLinear 
        # raises ValueError if repetition_idx is None.
        
        from spflow.utils.sampling_context import SamplingContext
        sampling_ctx = None
        if num_repetitions > 1:
            sampling_ctx = SamplingContext(
                 num_samples=10, 
                 repetition_index=torch.zeros(10, dtype=torch.long)
            )

        # Create NaN tensor for sampling
        sample_data = torch.full((10, 2), float("nan"))
        samples = leaf.sample(num_samples=10, data=sample_data, sampling_ctx=sampling_ctx)

        assert samples.shape == (10, 2)
        assert not torch.isnan(samples).any()


class TestDomain:
    """Test Domain utility class."""

    def test_discrete_bins(self):
        """Test discrete_bins factory method."""
        domain = Domain.discrete_bins([1, 2, 3, 5, 8])

        assert domain.data_type == DataType.DISCRETE
        assert domain.values == [1, 2, 3, 5, 8]
        assert domain.min == 1
        assert domain.max == 8

    def test_discrete_range(self):
        """Test discrete_range factory method."""
        domain = Domain.discrete_range(0, 5)

        assert domain.data_type == DataType.DISCRETE
        assert domain.values == [0, 1, 2, 3, 4, 5]
        assert domain.min == 0
        assert domain.max == 5

    def test_continuous_range(self):
        """Test continuous_range factory method."""
        domain = Domain.continuous_range(-1.0, 1.0)

        assert domain.data_type == DataType.CONTINUOUS
        assert domain.min == -1.0
        assert domain.max == 1.0
        assert domain.values is None

    def test_continuous_inf_support(self):
        """Test continuous_inf_support factory method."""
        import numpy as np

        domain = Domain.continuous_inf_support()

        assert domain.data_type == DataType.CONTINUOUS
        assert domain.min == -np.inf
        assert domain.max == np.inf
