"""Unit tests for ConvPc module."""

from itertools import product

import pytest
import torch

from spflow.exceptions import UnsupportedOperationError
from spflow.meta.data import Scope
from spflow.zoo.conv import ConvPc
from spflow.modules.leaves import Normal
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import DifferentiableSamplingContext

# Test parameter values
out_channels_values = [1, 3]
depth_values = [1, 2]
height_width_values = [(4, 4), (8, 8)]
num_repetitions_values = [1]
num_repetitions_unsupported_values = [3]
use_sum_conv_values = [True, False]


def _is_valid_spatial_depth(hw: tuple[int, int], depth: int) -> bool:
    height, width = hw
    return height // (2**depth) >= 1 and width // (2**depth) >= 1


all_params = list(
    product(
        out_channels_values, height_width_values, depth_values, num_repetitions_values, use_sum_conv_values
    )
)
valid_params = [p for p in all_params if _is_valid_spatial_depth(p[1], p[2])]

# Combined parameter lists (only valid combinations).
construction_params = valid_params
ll_params = valid_params
sample_params = valid_params


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

    @pytest.mark.parametrize("num_repetitions", num_repetitions_unsupported_values)
    def test_construction_rejects_multiple_repetitions(self, num_repetitions):
        leaf = make_normal_leaf(4, 4, out_channels=2, num_repetitions=1)
        with pytest.raises(UnsupportedOperationError):
            ConvPc(
                leaf=leaf,
                input_height=4,
                input_width=4,
                channels=5,
                depth=1,
                kernel_size=2,
                num_repetitions=num_repetitions,
                use_sum_conv=False,
            )

    @pytest.mark.parametrize("out_channels,hw,depth,num_repetitions,use_sum_conv", construction_params)
    def test_layer_structure(self, out_channels, hw, depth, num_repetitions, use_sum_conv):
        """Test that layers alternate ProdConv and SumConv via recursive inputs."""
        height, width = hw

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

    @pytest.mark.parametrize("out_channels,hw,depth,num_repetitions,use_sum_conv", sample_params)
    def test_rsample(self, out_channels, hw, depth, num_repetitions, use_sum_conv):
        height, width = hw
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

        num_samples = 8
        root_features = int(model.inputs.out_shape.features)
        root_channels = int(model.inputs.out_shape.channels)
        channel_probs = torch.rand((num_samples, root_features, root_channels))
        channel_probs = channel_probs / channel_probs.sum(dim=-1, keepdim=True)
        sampling_ctx = DifferentiableSamplingContext(
            channel_probs=channel_probs,
            mask=torch.ones((num_samples, root_features), dtype=torch.bool),
        )

        samples = model.rsample(num_samples=num_samples, diff_method="gumbel")
        assert samples.shape == (num_samples, height * width)
        assert torch.isfinite(samples).all()


class TestConvPcConditionalSample:
    """Test ConvPc conditional sampling."""

    @pytest.mark.parametrize("out_channels,hw,depth,num_repetitions,use_sum_conv", sample_params)
    def test_conditional_sample(self, out_channels, hw, depth, num_repetitions, use_sum_conv):
        """Test sampling with evidence."""
        height, width = hw

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
        torch.testing.assert_close(
            samples[:, :half_features],
            evidence_copy[:, :half_features],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )


# Additional branch-focused conv_pc tests
from spflow.zoo.conv.conv_pc import compute_non_overlapping_kernel_and_padding


def _leaf(h: int, w: int, c: int = 1, r: int = 1) -> Normal:
    return Normal(scope=Scope(list(range(h * w))), out_channels=c, num_repetitions=r)


def test_kernel_padding_validation_and_constructor_guards():
    with pytest.raises(ValueError):
        compute_non_overlapping_kernel_and_padding(0, 1, 1, 1)

    with pytest.raises(ValueError):
        ConvPc(leaf=_leaf(4, 4), input_height=4, input_width=4, channels=2, depth=0)

    with pytest.raises(ValueError):
        ConvPc(leaf=_leaf(4, 4), input_height=4, input_width=4, channels=0, depth=1)


def test_feature_scope_repr_cache_and_delegate_paths(monkeypatch):
    model = ConvPc(leaf=_leaf(4, 4), input_height=4, input_width=4, channels=2, depth=1, num_repetitions=1)
    assert model.feature_to_scope.shape[0] == 1
    assert "depth=1" in model.extra_repr()

    # log_likelihood cache=None branch
    x = torch.randn(3, 16)
    ll = model.log_likelihood(x, cache=None)
    assert ll.shape[0] == 3

    # sample data=None/num_samples=None branch
    s = model.sample(num_samples=None, data=None)
    assert s.shape == (1, 16)

    em_called = {"n": 0}

    def _fake_em(data, cache=None, bias_correction=True):
        em_called["n"] += 1

    monkeypatch.setattr(model.inputs, "_expectation_maximization_step", _fake_em)
    model._expectation_maximization_step(x, cache=Cache())
    assert em_called["n"] == 1

    sentinel = object()

    def _fake_marginalize(marg_rvs, prune=True, cache=None):
        return sentinel

    monkeypatch.setattr(model.inputs, "marginalize", _fake_marginalize)
    assert model.marginalize([0]) is sentinel
