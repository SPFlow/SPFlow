"""Unit tests for ProdConv module."""

from itertools import product

import pytest
import torch

from spflow.meta.data import Scope
from spflow.modules.conv import ProdConv
from spflow.modules.leaves import Normal
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext

# Test parameter values
out_channels_values = [1, 3]
height_width_kernel_values = [
    (4, 4, 2, 2),
    (8, 8, 2, 2),
    (8, 8, 4, 4),
]

# Combined parameter lists
construction_params = list(product(out_channels_values, height_width_kernel_values))
ll_params = list(product(out_channels_values, height_width_kernel_values))
sample_params = list(product(out_channels_values, height_width_kernel_values))


def make_normal_leaf(height: int, width: int, out_channels: int, num_repetitions: int = 1):
    """Create a Normal leaf for testing with spatial structure."""
    num_features = height * width
    scope = Scope(list(range(num_features)))
    return Normal(scope=scope, out_channels=out_channels, num_repetitions=num_repetitions)


class TestProdConvConstruction:
    """Test ProdConv construction and initialization."""

    @pytest.mark.parametrize("out_channels,hwk", construction_params)
    def test_basic_construction(self, out_channels, hwk):
        """Test that ProdConv can be constructed with valid parameters."""
        height, width, kernel_h, kernel_w = hwk
        leaf = make_normal_leaf(height, width, out_channels=out_channels)
        module = ProdConv(inputs=leaf, kernel_size_h=kernel_h, kernel_size_w=kernel_w)

        assert module.kernel_size_h == kernel_h
        assert module.kernel_size_w == kernel_w
        assert module.padding_h == 0
        assert module.padding_w == 0

    @pytest.mark.parametrize("out_channels,hwk", construction_params)
    def test_with_padding(self, out_channels, hwk):
        """Test construction with padding."""
        height, width, kernel_h, kernel_w = hwk
        leaf = make_normal_leaf(height, width, out_channels=out_channels)
        module = ProdConv(inputs=leaf, kernel_size_h=kernel_h, kernel_size_w=kernel_w, padding_h=1, padding_w=1)

        assert module.padding_h == 1
        assert module.padding_w == 1

    @pytest.mark.parametrize("out_channels", out_channels_values)
    def test_invalid_kernel_size(self, out_channels):
        """Test that kernel_size < 1 raises ValueError."""
        leaf = make_normal_leaf(4, 4, out_channels=out_channels)
        with pytest.raises(ValueError, match="kernel_size_h must be >= 1"):
            ProdConv(inputs=leaf, kernel_size_h=0, kernel_size_w=2)

    @pytest.mark.parametrize("out_channels,hwk", construction_params)
    @pytest.mark.parametrize("num_reps", [1, 2])
    def test_output_shape(self, out_channels, hwk, num_reps):
        """Test that output shape is correctly computed."""
        height, width, kernel_h, kernel_w = hwk
        leaf = make_normal_leaf(height, width, out_channels=out_channels, num_repetitions=num_reps)
        module = ProdConv(inputs=leaf, kernel_size_h=kernel_h, kernel_size_w=kernel_w)

        out_h = height // kernel_h
        out_w = width // kernel_w
        assert module.out_shape.features == out_h * out_w
        assert module.out_shape.channels == out_channels  # Channels preserved
        assert module.out_shape.repetitions == num_reps


class TestProdConvLogLikelihood:
    """Test ProdConv log_likelihood computation."""

    @pytest.mark.parametrize("out_channels,hwk", ll_params)
    @pytest.mark.parametrize("num_reps", [1, 2])
    def test_log_likelihood_shape(self, out_channels, hwk, num_reps):
        """Test that log_likelihood output has correct shape."""
        height, width, kernel_h, kernel_w = hwk
        leaf = make_normal_leaf(height, width, out_channels=out_channels, num_repetitions=num_reps)
        module = ProdConv(inputs=leaf, kernel_size_h=kernel_h, kernel_size_w=kernel_w)

        batch_size = 10
        data = torch.randn(batch_size, height * width)
        ll = module.log_likelihood(data)

        out_h = height // kernel_h
        out_w = width // kernel_w
        out_features = out_h * out_w

        assert ll.shape == (batch_size, out_features, out_channels, num_reps)

    @pytest.mark.parametrize("out_channels,hwk", ll_params)
    def test_log_likelihood_sum_in_log_space(self, out_channels, hwk):
        """Test that ProdConv sums log-likelihoods within patches."""
        # Use fixed parameters for this specific test
        leaf = make_normal_leaf(4, 4, out_channels=1)
        module = ProdConv(inputs=leaf, kernel_size_h=2, kernel_size_w=2)

        # Create data
        data = torch.randn(1, 16)
        cache = Cache()
        ll = module.log_likelihood(data, cache=cache)

        # Get input log-likelihoods
        input_ll = cache["log_likelihood"][leaf]  # (1, 16, 1, 1)

        # Reshape input to spatial
        input_ll_spatial = input_ll.view(1, 4, 4, 1, 1)

        # First output position should be sum of first 2x2 patch
        expected_first = input_ll_spatial[0, 0:2, 0:2, 0, 0].sum()
        actual_first = ll[0, 0, 0, 0]

        assert torch.allclose(actual_first, expected_first)

    @pytest.mark.parametrize("out_channels,hwk", ll_params)
    def test_log_likelihood_finite(self, out_channels, hwk):
        """Test that log_likelihood values are finite."""
        height, width, kernel_h, kernel_w = hwk
        leaf = make_normal_leaf(height, width, out_channels=out_channels)
        module = ProdConv(inputs=leaf, kernel_size_h=kernel_h, kernel_size_w=kernel_w)

        data = torch.randn(10, height * width)
        ll = module.log_likelihood(data)

        assert torch.isfinite(ll).all()


class TestProdConvFeatureToScope:
    """Test ProdConv feature_to_scope property (scope aggregation)."""

    @pytest.mark.parametrize("out_channels,hwk", construction_params)
    @pytest.mark.parametrize("num_reps", [1, 2])
    def test_scope_aggregation(self, out_channels, hwk, num_reps):
        """Test that patches aggregate scopes correctly."""
        height, width, kernel_h, kernel_w = hwk
        leaf = make_normal_leaf(height, width, out_channels=out_channels, num_repetitions=num_reps)
        module = ProdConv(inputs=leaf, kernel_size_h=kernel_h, kernel_size_w=kernel_w)

        f2s = module.feature_to_scope

        out_h = height // kernel_h
        out_w = width // kernel_w
        out_features = out_h * out_w

        # Shape should be (out_features, num_reps)
        assert f2s.shape == (out_features, num_reps)

        # First output position should have scope from first patch
        first_scope = f2s[0, 0]
        expected_indices = set()
        for i in range(kernel_h):
            for j in range(kernel_w):
                expected_indices.add(i * width + j)
        assert set(first_scope.query) == expected_indices

    @pytest.mark.parametrize("out_channels,hwk", construction_params)
    def test_all_scopes_are_scope_objects(self, out_channels, hwk):
        """Test that all feature_to_scope entries are Scope objects."""
        height, width, kernel_h, kernel_w = hwk
        leaf = make_normal_leaf(height, width, out_channels=out_channels)
        module = ProdConv(inputs=leaf, kernel_size_h=kernel_h, kernel_size_w=kernel_w)

        f2s = module.feature_to_scope
        assert all(isinstance(s, Scope) for s in f2s.flatten())


class TestProdConvSample:
    """Test ProdConv sampling (upsampling)."""

    @pytest.mark.parametrize("out_channels,hwk", sample_params)
    def test_sample_shape(self, out_channels, hwk):
        """Test that samples have correct shape."""
        height, width, kernel_h, kernel_w = hwk
        leaf = make_normal_leaf(height, width, out_channels=out_channels)
        module = ProdConv(inputs=leaf, kernel_size_h=kernel_h, kernel_size_w=kernel_w)

        num_samples = 20
        samples = module.sample(num_samples=num_samples)

        assert samples.shape == (num_samples, height * width)

    @pytest.mark.parametrize("out_channels,hwk", sample_params)
    def test_sample_finite(self, out_channels, hwk):
        """Test that samples are finite."""
        height, width, kernel_h, kernel_w = hwk
        leaf = make_normal_leaf(height, width, out_channels=out_channels)
        module = ProdConv(inputs=leaf, kernel_size_h=kernel_h, kernel_size_w=kernel_w)

        samples = module.sample(num_samples=10)
        assert torch.isfinite(samples).all()

    @pytest.mark.parametrize("out_channels,hwk", sample_params)
    def test_sample_channel_upsampling(self, out_channels, hwk):
        """Test that channel indices are properly upsampled."""
        height, width, kernel_h, kernel_w = hwk
        leaf = make_normal_leaf(height, width, out_channels=out_channels)
        module = ProdConv(inputs=leaf, kernel_size_h=kernel_h, kernel_size_w=kernel_w)

        batch_size = 5
        out_features = module.out_shape.features

        # Create sampling context with specific channel indices
        channel_idx = torch.zeros(batch_size, out_features, dtype=torch.long)
        for i in range(out_features):
            channel_idx[:, i] = i % out_channels

        mask = torch.ones(batch_size, out_features, dtype=torch.bool)
        sampling_ctx = SamplingContext(channel_index=channel_idx, mask=mask)

        data = torch.full((batch_size, height * width), float("nan"))
        module.sample(data=data, sampling_ctx=sampling_ctx)

        # After upsampling, samples should be finite
        assert torch.isfinite(data).all()
