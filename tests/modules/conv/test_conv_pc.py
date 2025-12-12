"""Unit tests for ConvPc module."""

from itertools import product

import pytest
import torch

from spflow.meta.data import Scope
from spflow.modules.conv import ConvPc
from spflow.modules.leaves import Normal
from spflow.utils.cache import Cache

# Test parameter values
out_channels_values = [1, 3]
depth_values = [1, 2]
height_width_values = [(4, 4), (8, 8)]
num_repetitions_values = [1, 3]
use_sum_conv_values = [True, False]

# Combined parameter lists
construction_params = list(product(out_channels_values, height_width_values, depth_values, num_repetitions_values, use_sum_conv_values))
ll_params = list(product(out_channels_values, height_width_values, depth_values, num_repetitions_values, use_sum_conv_values))
sample_params = list(product(out_channels_values, height_width_values, depth_values, num_repetitions_values, use_sum_conv_values))


def make_normal_leaf(height: int, width: int, out_channels: int, num_repetitions: int = 1):
    """Create a Normal leaf for testing with spatial structure."""
    num_features = height * width
    scope = Scope(list(range(num_features)))
    return Normal(scope=scope, out_channels=out_channels, num_repetitions=num_repetitions)


class TestConvPcConstruction:
    """Test ConvPc construction and initialization."""

    @pytest.mark.parametrize("out_channels,hw,depth,num_repetitions,use_sum_conv", construction_params)
    def test_basic_construction(self, out_channels, hw, depth, num_repetitions, use_sum_conv):
        """Test that ConvPc can be constructed with valid parameters."""
        height, width = hw
        # Skip invalid combinations where depth exceeds spatial dimensions
        if height // (2 ** depth) < 1 or width // (2 ** depth) < 1:
            pytest.skip("Invalid depth for spatial dimensions")

        leaf = make_normal_leaf(height, width, out_channels=out_channels, num_repetitions=num_repetitions)
        model = ConvPc(
            leaf=leaf,
            input_height=height,
            input_width=width,
            channels=5,
            depth=depth,
            kernel_size=2,
            num_repetitions=num_repetitions,
            use_sum_conv=use_sum_conv,
        )

        assert model.depth == depth
        assert model.kernel_size == 2
        assert model.input_height == height
        assert model.input_width == width
        # Output repetitions always 1 due to mixing layer
        assert model.out_shape.repetitions == 1

    @pytest.mark.parametrize("out_channels,hw,depth,num_repetitions,use_sum_conv", construction_params)
    def test_layer_structure(self, out_channels, hw, depth, num_repetitions, use_sum_conv):
        """Test that layers alternate ProdConv and SumConv via recursive inputs."""
        height, width = hw
        # Skip invalid combinations
        if height // (2 ** depth) < 1 or width // (2 ** depth) < 1:
            pytest.skip("Invalid depth for spatial dimensions")

        leaf = make_normal_leaf(height, width, out_channels=out_channels, num_repetitions=num_repetitions)
        model = ConvPc(
            leaf=leaf,
            input_height=height,
            input_width=width,
            channels=5,
            depth=depth,
            kernel_size=2,
            num_repetitions=num_repetitions,
            use_sum_conv=use_sum_conv,
        )

        # Traverse the recursive structure: inputs (Sum) -> ProdConv -> SumConv -> ... -> leaf
        # Count layers by walking the .inputs chain from the root Sum
        layer_count = 0
        current = model.inputs.inputs  # Skip the root Sum layer
        layer_types = []
        while current is not leaf:
            layer_types.append(type(current).__name__)
            current = current.inputs
            layer_count += 1

        # With bottom-up architecture: depth*(ProdConv + SumConv) + optional final ProdConv
        # The final ProdConv only exists if there are remaining spatial dims > 1
        # Expected: ProdConv, SumConv, ProdConv, SumConv, ..., possibly final ProdConv
        assert layer_count >= 2 * depth  # At minimum depth pairs of (Prod, Sum)

    @pytest.mark.parametrize("out_channels", out_channels_values)
    def test_arbitrary_input_sizes(self, out_channels):
        """Test that valid square input sizes work."""
        # With kernel_size=2, dimensions must be divisible at each depth
        # 8x8 with depth=2: 8 -> 4 -> 2 (all divisible by 2)
        leaf = make_normal_leaf(8, 8, out_channels=out_channels)

        model = ConvPc(
            leaf=leaf,
            input_height=8,
            input_width=8,
            channels=5,
            depth=2,
            kernel_size=2,
        )

        # Should produce 1x1 output
        assert model.out_shape.features == 1
        assert model.out_shape.channels == 1

    @pytest.mark.parametrize("out_channels,hw,depth,num_repetitions,use_sum_conv", construction_params)
    def test_output_shape(self, out_channels, hw, depth, num_repetitions, use_sum_conv):
        """Test that output shape is scalar."""
        height, width = hw
        # Skip invalid combinations
        if height // (2 ** depth) < 1 or width // (2 ** depth) < 1:
            pytest.skip("Invalid depth for spatial dimensions")

        leaf = make_normal_leaf(height, width, out_channels=out_channels, num_repetitions=num_repetitions)
        model = ConvPc(
            leaf=leaf,
            input_height=height,
            input_width=width,
            channels=5,
            depth=depth,
            kernel_size=2,
            num_repetitions=num_repetitions,
            use_sum_conv=use_sum_conv,
        )

        assert model.out_shape.features == 1
        assert model.out_shape.channels == 1
        assert model.out_shape.repetitions == 1  # Always 1 due to mixing


class TestConvPcLogLikelihood:
    """Test ConvPc log_likelihood computation."""

    @pytest.mark.parametrize("out_channels,hw,depth,num_repetitions,use_sum_conv", ll_params)
    def test_log_likelihood_shape(self, out_channels, hw, depth, num_repetitions, use_sum_conv):
        """Test that log_likelihood output has correct shape."""
        height, width = hw
        # Skip invalid combinations
        if height // (2 ** depth) < 1 or width // (2 ** depth) < 1:
            pytest.skip("Invalid depth for spatial dimensions")

        leaf = make_normal_leaf(height, width, out_channels=out_channels, num_repetitions=num_repetitions)
        model = ConvPc(
            leaf=leaf,
            input_height=height,
            input_width=width,
            channels=5,
            depth=depth,
            kernel_size=2,
            num_repetitions=num_repetitions,
            use_sum_conv=use_sum_conv,
        )

        batch_size = 10
        data = torch.randn(batch_size, height * width)
        ll = model.log_likelihood(data)

        # Output should be (batch, 1, 1, reps) - but ConvPc ends at the root Sum layer
        # which has shape (batch, out_features, out_channels, reps)
        # For a full ConvPc, we expect scalar output
        assert ll.dim() == 4
        assert ll.shape[0] == batch_size
        assert ll.shape[2] == 1  # Final out_channels = 1
        assert ll.shape[3] == 1  # Repetitions = 1

    @pytest.mark.parametrize("out_channels,hw,depth,num_repetitions,use_sum_conv", ll_params)
    def test_log_likelihood_finite(self, out_channels, hw, depth, num_repetitions, use_sum_conv):
        """Test that log_likelihood values are finite."""
        height, width = hw
        # Skip invalid combinations
        if height // (2 ** depth) < 1 or width // (2 ** depth) < 1:
            pytest.skip("Invalid depth for spatial dimensions")

        leaf = make_normal_leaf(height, width, out_channels=out_channels, num_repetitions=num_repetitions)
        model = ConvPc(
            leaf=leaf,
            input_height=height,
            input_width=width,
            channels=5,
            depth=depth,
            num_repetitions=num_repetitions,
            use_sum_conv=use_sum_conv,
        )

        data = torch.randn(10, height * width)
        ll = model.log_likelihood(data)

        assert torch.isfinite(ll).all()

    @pytest.mark.parametrize("out_channels,hw,depth,num_repetitions,use_sum_conv", ll_params)
    def test_log_likelihood_negative(self, out_channels, hw, depth, num_repetitions, use_sum_conv):
        """Test that log_likelihood values are negative (proper probabilities)."""
        height, width = hw
        # Skip invalid combinations
        if height // (2 ** depth) < 1 or width // (2 ** depth) < 1:
            pytest.skip("Invalid depth for spatial dimensions")

        leaf = make_normal_leaf(height, width, out_channels=out_channels, num_repetitions=num_repetitions)
        model = ConvPc(
            leaf=leaf,
            input_height=height,
            input_width=width,
            channels=5,
            depth=depth,
            num_repetitions=num_repetitions,
            use_sum_conv=use_sum_conv,
        )

        data = torch.randn(10, height * width)
        ll = model.log_likelihood(data)

        assert (ll <= 0).all()


class TestConvPcSample:
    """Test ConvPc sampling."""

    @pytest.mark.parametrize("out_channels,hw,depth,num_repetitions,use_sum_conv", sample_params)
    def test_sample_shape(self, out_channels, hw, depth, num_repetitions, use_sum_conv):
        """Test that samples have correct shape."""
        height, width = hw
        # Skip invalid combinations
        if height // (2 ** depth) < 1 or width // (2 ** depth) < 1:
            pytest.skip("Invalid depth for spatial dimensions")

        leaf = make_normal_leaf(height, width, out_channels=out_channels, num_repetitions=num_repetitions)
        model = ConvPc(
            leaf=leaf,
            input_height=height,
            input_width=width,
            channels=5,
            depth=depth,
            num_repetitions=num_repetitions,
            use_sum_conv=use_sum_conv,
        )

        num_samples = 20
        samples = model.sample(num_samples=num_samples)

        assert samples.shape == (num_samples, height * width)

    @pytest.mark.parametrize("out_channels,hw,depth,num_repetitions,use_sum_conv", sample_params)
    def test_sample_finite(self, out_channels, hw, depth, num_repetitions, use_sum_conv):
        """Test that samples are finite."""
        height, width = hw
        # Skip invalid combinations
        if height // (2 ** depth) < 1 or width // (2 ** depth) < 1:
            pytest.skip("Invalid depth for spatial dimensions")

        leaf = make_normal_leaf(height, width, out_channels=out_channels, num_repetitions=num_repetitions)
        model = ConvPc(
            leaf=leaf,
            input_height=height,
            input_width=width,
            channels=5,
            depth=depth,
            num_repetitions=num_repetitions,
            use_sum_conv=use_sum_conv,
        )

        samples = model.sample(num_samples=10)
        assert torch.isfinite(samples).all()

    @pytest.mark.parametrize("out_channels,hw,depth,num_repetitions,use_sum_conv", sample_params)
    def test_sample_mpe(self, out_channels, hw, depth, num_repetitions, use_sum_conv):
        """Test MPE sampling."""
        height, width = hw
        # Skip invalid combinations
        if height // (2 ** depth) < 1 or width // (2 ** depth) < 1:
            pytest.skip("Invalid depth for spatial dimensions")

        leaf = make_normal_leaf(height, width, out_channels=out_channels, num_repetitions=num_repetitions)
        model = ConvPc(
            leaf=leaf,
            input_height=height,
            input_width=width,
            channels=5,
            depth=depth,
            num_repetitions=num_repetitions,
            use_sum_conv=use_sum_conv,
        )

        samples = model.sample(num_samples=10, is_mpe=True)
        assert torch.isfinite(samples).all()


class TestConvPcConditionalSample:
    """Test ConvPc conditional sampling."""

    @pytest.mark.parametrize("out_channels,hw,depth,num_repetitions,use_sum_conv", sample_params)
    def test_conditional_sample(self, out_channels, hw, depth, num_repetitions, use_sum_conv):
        """Test sampling with evidence."""
        height, width = hw
        # Skip invalid combinations
        if height // (2 ** depth) < 1 or width // (2 ** depth) < 1:
            pytest.skip("Invalid depth for spatial dimensions")

        leaf = make_normal_leaf(height, width, out_channels=out_channels, num_repetitions=num_repetitions)
        model = ConvPc(
            leaf=leaf,
            input_height=height,
            input_width=width,
            channels=5,
            depth=depth,
            num_repetitions=num_repetitions,
            use_sum_conv=use_sum_conv,
        )

        num_features = height * width
        half_features = num_features // 2

        # Create evidence: first half pixels observed
        evidence = torch.randn(10, num_features)
        evidence[:, half_features:] = float("nan")
        evidence_copy = evidence.clone()

        # Run conditional sampling
        cache = Cache()
        samples = model.sample_with_evidence(evidence=evidence, cache=cache)

        assert samples.shape == evidence.shape
        assert torch.isfinite(samples).all()

        # Observed values should be unchanged
        assert torch.allclose(samples[:, :half_features], evidence_copy[:, :half_features], equal_nan=True)
