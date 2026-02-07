"""Tests for PiecewiseLinear leaf distribution module."""

import builtins
import sys
import types

import pytest
import torch

from spflow.exceptions import OptionalDependencyError
from spflow.meta.data import Scope
from spflow.modules.leaves.piecewise_linear import PiecewiseLinear, PiecewiseLinearDist, interp, pairwise
from spflow.utils.domain import DataType, Domain


def _randn(*size: int) -> torch.Tensor:
    return torch.randn(*size)


def _randint(low: int, high: int, size: tuple[int, ...]) -> torch.Tensor:
    return torch.randint(low=low, high=high, size=size)


class TestPiecewiseLinearInitialization:
    """Test initialization of PiecewiseLinear leaf."""

    @pytest.mark.parametrize("num_repetitions", [1, 2])
    @pytest.mark.parametrize("out_channels", [1, 3])
    def test_initialization_continuous(self, num_repetitions, out_channels):
        """Test initialization with continuous data."""
        scope = Scope([0, 1])
        leaf = PiecewiseLinear(scope=scope, out_channels=out_channels, num_repetitions=num_repetitions)

        # Generate synthetic data
        data = _randn(100, 2)
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
        scope = Scope([0])
        leaf = PiecewiseLinear(scope=scope, out_channels=1, num_repetitions=1)

        # Generate synthetic discrete data (0-9)
        data = _randint(0, 10, (100, 1)).float()
        domains = [Domain.discrete_range(0, 9)]

        leaf.initialize(data, domains)

        assert leaf.is_initialized
        assert leaf.xs is not None

    def test_initialization_mixed(self):
        """Test initialization with mixed continuous and discrete data."""
        scope = Scope([0, 1])
        leaf = PiecewiseLinear(scope=scope, out_channels=1, num_repetitions=1)

        # Generate mixed data
        continuous_data = _randn(100, 1)
        discrete_data = _randint(0, 5, (100, 1)).float()
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

        with pytest.raises(ValueError):
            _ = leaf.distribution

        with pytest.raises(ValueError):
            leaf.log_likelihood(_randn(10, 1))

        with pytest.raises(ValueError):
            leaf.sample(num_samples=10, data=torch.full((10, 1), float("nan")))

    def test_negative_alpha_raises(self):
        """Test alpha validation."""
        with pytest.raises(ValueError):
            _ = PiecewiseLinear(scope=Scope([0]), alpha=-0.1)

    def test_torch_distribution_class_is_none(self):
        """Test torch distribution class property."""
        leaf = PiecewiseLinear(scope=Scope([0]))
        assert leaf._torch_distribution_class is None

    def test_initialize_import_error_raises_optional_dependency_error(self, monkeypatch):
        """Test OptionalDependencyError branch when k-means dependency is missing."""
        real_import = builtins.__import__

        def _raising_import(name, *args, **kwargs):
            if name == "fast_pytorch_kmeans":
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _raising_import)

        leaf = PiecewiseLinear(scope=Scope([0]))
        with pytest.raises(OptionalDependencyError):
            leaf.initialize(_randn(10, 1), [Domain.continuous_inf_support()])

    def test_initialize_shape_validation_errors(self):
        """Test input validation branches in initialize."""
        leaf = PiecewiseLinear(scope=Scope([0, 1]))
        domains = [Domain.continuous_inf_support(), Domain.continuous_inf_support()]

        with pytest.raises(ValueError):
            leaf.initialize(_randn(8, 1), domains)

        with pytest.raises(ValueError):
            leaf.initialize(_randn(8, 2), domains[:1])

    def test_reset(self):
        """Test reset functionality."""
        scope = Scope([0])
        leaf = PiecewiseLinear(scope=scope, out_channels=1)

        data = _randn(100, 1)
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
        scope = Scope([0, 1])
        leaf = PiecewiseLinear(scope=scope, out_channels=out_channels, num_repetitions=num_repetitions)

        data = _randn(100, 2)
        domains = [Domain.continuous_inf_support(), Domain.continuous_inf_support()]
        leaf.initialize(data, domains)

        test_data = _randn(20, 2)
        log_prob = leaf.log_likelihood(test_data)

        # Expected shape: (batch, features, channels, leaves, repetitions)
        assert log_prob.shape == (20, 1, 2, out_channels, num_repetitions)

    def test_log_likelihood_values(self):
        """Test that log-likelihoods are valid (finite)."""
        scope = Scope([0])
        leaf = PiecewiseLinear(scope=scope, out_channels=1, num_repetitions=1)

        data = _randn(1000, 1)
        domains = [Domain.continuous_inf_support()]
        leaf.initialize(data, domains)

        # Test on data within the training range
        test_data = _randn(50, 1)
        log_prob = leaf.log_likelihood(test_data)

        # Log probs should be finite
        assert torch.isfinite(log_prob).all()

    def test_log_likelihood_marginalization(self):
        """Test that NaN values are marginalized correctly."""
        scope = Scope([0, 1])
        leaf = PiecewiseLinear(scope=scope, out_channels=1, num_repetitions=1)

        data = _randn(100, 2)
        domains = [Domain.continuous_inf_support(), Domain.continuous_inf_support()]
        leaf.initialize(data, domains)

        # Create test data with NaN (marginalized) values
        test_data = _randn(10, 2)
        test_data[0, 0] = float("nan")
        test_data[5, 1] = float("nan")

        log_prob = leaf.log_likelihood(test_data)

        # Should still produce valid output
        assert not torch.isnan(log_prob).any()

    def test_log_likelihood_requires_2d_data(self):
        """Test dimensionality validation branch."""
        scope = Scope([0])
        leaf = PiecewiseLinear(scope=scope, out_channels=1, num_repetitions=1)
        leaf.initialize(_randn(20, 1), [Domain.continuous_inf_support()])

        with pytest.raises(ValueError):
            _ = leaf.log_likelihood(_randn(20, 1, 1))


class TestPiecewiseLinearDist:
    """Test PiecewiseLinearDist distribution class."""

    def test_log_prob(self):
        """Test log_prob computation."""
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

    def test_log_prob_accepts_5d_input(self):
        """Test log_prob with [N, C, F, 1, 1] inputs."""
        xs = [[[[torch.tensor([-1.0, 0.0, 1.0, 2.0])]]]]
        ys = [[[[torch.tensor([0.0, 0.5, 0.5, 0.0])]]]]
        domains = [Domain.continuous_inf_support()]
        dist = PiecewiseLinearDist(xs, ys, domains)

        x = torch.tensor([[[[[0.25]]]], [[[[1.5]]]]])  # [N=2, C=1, F=1, 1, 1]
        log_prob = dist.log_prob(x)

        assert log_prob.shape == (2, 1, 1, 1, 1)
        assert torch.isfinite(log_prob).all()

    def test_sample_discrete_values(self):
        """Test sampling discrete values from categorical branch."""
        xs = [[[[torch.tensor([-1.0, 0.0, 1.0, 2.0])]]]]
        ys = [[[[torch.tensor([0.0, 0.1, 0.9, 0.0])]]]]
        domains = [Domain.discrete_range(0, 1)]
        dist = PiecewiseLinearDist(xs, ys, domains)

        samples = dist.sample((128,))

        assert samples.shape == (128, 1, 1, 1, 1)
        assert torch.isin(samples.unique(), torch.tensor([0.0, 1.0])).all()

    def test_sample_continuous_values_within_support(self):
        """Test continuous inverse-CDF sampling branch."""
        xs = [[[[torch.tensor([-1.0, 0.0, 1.0, 2.0])]]]]
        ys = [[[[torch.tensor([0.0, 1.0, 0.5, 0.0])]]]]
        domains = [Domain.continuous_inf_support()]
        dist = PiecewiseLinearDist(xs, ys, domains)

        samples = dist.sample((128,))

        assert samples.shape == (128, 1, 1, 1, 1)
        assert torch.all(samples >= -1.0)
        assert torch.all(samples <= 2.0)

    def test_sample_unknown_domain_type_raises(self):
        """Test error branch for unknown data types."""

        class _InvalidDomain:
            data_type = "invalid"

        xs = [[[[torch.tensor([-1.0, 0.0, 1.0, 2.0])]]]]
        ys = [[[[torch.tensor([0.0, 1.0, 0.5, 0.0])]]]]
        dist = PiecewiseLinearDist(xs, ys, [_InvalidDomain()])

        with pytest.raises(ValueError):
            _ = dist.sample((4,))


class TestPiecewiseLinearHelpers:
    """Test helper functions in piecewise_linear module."""

    def test_pairwise(self):
        """Test pairwise helper output."""
        assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]

    def test_interp_constant_extrapolation(self):
        """Test constant-value extrapolation."""
        xp = torch.tensor([0.0, 1.0, 2.0])
        fp = torch.tensor([0.0, 1.0, 0.0])
        x = torch.tensor([-1.0, 0.5, 3.0])

        y = interp(x=x, xp=xp, fp=fp, extrapolate="constant")

        assert torch.allclose(y, torch.tensor([0.0, 0.5, 0.0]))

    def test_interp_linear_extrapolation_with_dim(self):
        """Test linear extrapolation branch and non-default interpolation axis."""
        xp = torch.tensor([[0.0], [1.0], [2.0]])
        fp = torch.tensor([[1.0], [0.0], [1.0]])
        x = torch.tensor([[-1.0], [0.5], [3.0]])

        y = interp(x=x, xp=xp, fp=fp, dim=0, extrapolate="linear")

        assert y.shape == x.shape
        assert torch.all(y >= 0.0)


class TestPiecewiseLinearSampling:
    """Test sampling functionality."""

    @pytest.mark.parametrize("num_repetitions", [1, 2])
    @pytest.mark.parametrize("out_channels", [1, 3])
    def test_sample_shape(self, num_repetitions, out_channels):
        """Test that sampled outputs have correct shape."""
        scope = Scope([0, 1])
        leaf = PiecewiseLinear(scope=scope, out_channels=out_channels, num_repetitions=num_repetitions)

        data = _randn(100, 2)
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
            sampling_ctx = SamplingContext(num_samples=10, repetition_index=torch.zeros(10, dtype=torch.long))

        # Create NaN tensor for sampling
        sample_data = torch.full((10, 2), float("nan"))
        samples = leaf.sample(num_samples=10, data=sample_data, sampling_ctx=sampling_ctx)

        assert samples.shape == (10, 2)
        assert not torch.isnan(samples).any()

    def test_sample_requires_repetition_idx_for_multiple_repetitions(self):
        """Test repetition-index validation in sample."""
        scope = Scope([0])
        leaf = PiecewiseLinear(scope=scope, out_channels=1, num_repetitions=2)
        data = _randn(100, 1)
        domains = [Domain.continuous_inf_support()]
        leaf.initialize(data, domains)

        sample_data = torch.full((8, 1), float("nan"))
        with pytest.raises(ValueError):
            _ = leaf.sample(num_samples=8, data=sample_data)

    def test_sample_is_mpe_branch(self):
        """Test MPE sampling branch."""
        scope = Scope([0, 1])
        leaf = PiecewiseLinear(scope=scope, out_channels=1, num_repetitions=1)
        leaf.initialize(_randn(50, 2), [Domain.continuous_inf_support(), Domain.continuous_inf_support()])

        sample_data = torch.full((6, 2), float("nan"))
        samples = leaf.sample(num_samples=6, data=sample_data, is_mpe=True)

        assert samples.shape == (6, 2)
        assert torch.isfinite(samples).all()


class TestPiecewiseLinearParamsAndMode:
    """Test params/mode and unsupported MLE branches."""

    def test_params_and_mode(self):
        """Test params() and mode property accessors."""
        leaf = PiecewiseLinear(scope=Scope([0]), out_channels=1, num_repetitions=1)
        leaf.initialize(_randn(40, 1), [Domain.continuous_inf_support()])

        params = leaf.params()
        assert "xs" in params and "ys" in params
        mode = leaf.mode
        assert mode.shape == (1, 1, 1, 1)

    def test_compute_parameter_estimates_not_implemented(self):
        """Test MLE unsupported branch."""
        leaf = PiecewiseLinear(scope=Scope([0]))
        with pytest.raises(NotImplementedError):
            _ = leaf._compute_parameter_estimates(
                data=_randn(4, 1),
                weights=torch.ones(4, 1, 1),
                bias_correction=False,
            )


class TestPiecewiseLinearInitializeBranches:
    """Test additional initialize branch coverage with deterministic fake KMeans."""

    @staticmethod
    def _install_fake_kmeans(monkeypatch, assignments):
        """Install a fake fast_pytorch_kmeans module with deterministic assignments."""

        class FakeKMeans:
            def __init__(self, n_clusters, mode, verbose, init_method):
                self.n_clusters = n_clusters
                self.centroids = torch.zeros(n_clusters, 1)

            def fit(self, data):
                self.centroids = torch.zeros(self.n_clusters, data.shape[1], device=data.device)
                return self

            def max_sim(self, a, b):
                idx = torch.as_tensor(assignments, dtype=torch.long, device=a.device)
                return torch.zeros_like(idx, dtype=torch.float32), idx

        fake_mod = types.SimpleNamespace(KMeans=FakeKMeans)
        monkeypatch.setitem(sys.modules, "fast_pytorch_kmeans", fake_mod)

    def test_initialize_empty_cluster_discrete_uses_uniform(self, monkeypatch):
        """Test empty discrete cluster fallback densities."""
        self._install_fake_kmeans(monkeypatch, assignments=[0, 0, 0, 0])

        leaf = PiecewiseLinear(scope=Scope([0]), out_channels=2, num_repetitions=1)
        data = torch.tensor([[0.0], [1.0], [2.0], [1.0]])
        leaf.initialize(data, [Domain.discrete_range(0, 2)])

        # cluster_idx=1 gets no points; with tails this should be [0, 1/3, 1/3, 1/3, 0]
        y_empty_cluster = leaf.ys[0][1][0][0]
        assert torch.isclose(y_empty_cluster[1:-1].sum(), torch.tensor(1.0), atol=1e-5)

    def test_initialize_empty_cluster_continuous_and_alpha_smoothing(self, monkeypatch):
        """Test empty continuous cluster fallback + Laplace smoothing branch."""
        self._install_fake_kmeans(monkeypatch, assignments=[0, 0, 0, 0, 0])

        leaf = PiecewiseLinear(scope=Scope([0]), out_channels=2, num_repetitions=1, alpha=0.5)
        data = torch.tensor([[0.0], [0.2], [0.4], [0.6], [0.8]])
        leaf.initialize(data, [Domain.continuous_range(0.0, 1.0)])

        y_empty_cluster = leaf.ys[0][1][0][0]
        assert torch.isfinite(y_empty_cluster).all()
        assert torch.all(y_empty_cluster >= 0.0)

    def test_initialize_unknown_data_type_raises(self, monkeypatch):
        """Test error for unknown data type in histogram construction."""
        self._install_fake_kmeans(monkeypatch, assignments=[0, 0, 0])

        class _InvalidDomain:
            data_type = "invalid"
            min = 0.0
            max = 1.0
            values = None

        leaf = PiecewiseLinear(scope=Scope([0]), out_channels=2, num_repetitions=1)
        with pytest.raises(ValueError):
            leaf.initialize(torch.tensor([[0.0], [0.5], [1.0]]), [_InvalidDomain()])

    def test_initialize_tail_break_unknown_type_raises(self, monkeypatch):
        """Test tail-break error branch with changing data type value."""
        self._install_fake_kmeans(monkeypatch, assignments=[0, 0, 0])

        class _FlakyDomain:
            min = 0.0
            max = 1.0
            values = None

            def __init__(self):
                self.calls = 0

            @property
            def data_type(self):
                self.calls += 1
                if self.calls <= 2:
                    return DataType.CONTINUOUS
                return "invalid"

        leaf = PiecewiseLinear(scope=Scope([0]), out_channels=2, num_repetitions=1)
        with pytest.raises(ValueError):
            leaf.initialize(torch.tensor([[0.0], [0.5], [1.0]]), [_FlakyDomain()])


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
