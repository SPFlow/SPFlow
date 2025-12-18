"""Tests for LinsumLayer module."""

from itertools import product

import pytest
import torch

from spflow.meta import Scope
from spflow.modules.einsum import LinsumLayer
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext
from tests.utils.leaves import make_normal_leaf, make_normal_data, DummyLeaf, make_leaf


# Test parameter combinations
in_channels_values = [1, 3]
out_channels_values = [1, 4]
in_features_values = [2, 4, 8]  # Must be even for LinsumLayer
num_repetitions_values = [1, 2]

params = list(
    product(in_channels_values, out_channels_values, in_features_values, num_repetitions_values)
)


def make_linsum_single_input(
    in_channels: int, out_channels: int, in_features: int, num_repetitions: int
) -> LinsumLayer:
    """Create LinsumLayer with single input module (split internally)."""
    inputs = make_normal_leaf(
        out_features=in_features,
        out_channels=in_channels,
        num_repetitions=num_repetitions,
    )
    return LinsumLayer(
        inputs=inputs,
        out_channels=out_channels,
        num_repetitions=num_repetitions,
    )


def make_linsum_two_inputs(
    in_channels: int, out_channels: int, in_features: int, num_repetitions: int
) -> LinsumLayer:
    """Create LinsumLayer with two separate input modules."""
    # Create left and right inputs with disjoint scopes
    left_scope = Scope(list(range(0, in_features)))
    right_scope = Scope(list(range(in_features, in_features * 2)))

    left_input = make_leaf(
        cls=DummyLeaf,
        out_channels=in_channels,
        scope=left_scope,
        num_repetitions=num_repetitions,
    )
    right_input = make_leaf(
        cls=DummyLeaf,
        out_channels=in_channels,
        scope=right_scope,
        num_repetitions=num_repetitions,
    )

    return LinsumLayer(
        inputs=[left_input, right_input],
        out_channels=out_channels,
        num_repetitions=num_repetitions,
    )


class TestLinsumLayerConstruction:
    """Test LinsumLayer construction and initialization."""

    @pytest.mark.parametrize("in_channels,out_channels,in_features,num_reps", params)
    def test_single_input_construction(
        self, in_channels: int, out_channels: int, in_features: int, num_reps: int
    ):
        """Test constructing LinsumLayer with single input."""
        module = make_linsum_single_input(in_channels, out_channels, in_features, num_reps)

        # Check output shape
        assert module.out_shape.features == in_features // 2
        assert module.out_shape.channels == out_channels
        assert module.out_shape.repetitions == num_reps

        # Check weights shape - linear, not cross-product
        expected_weight_shape = (
            in_features // 2,  # out_features
            out_channels,
            num_reps,
            in_channels,  # Only single channel dim (not in_channels × in_channels)
        )
        assert module.weights_shape == expected_weight_shape
        assert module.weights.shape == expected_weight_shape

    @pytest.mark.parametrize("in_channels,out_channels,in_features,num_reps", params)
    def test_two_input_construction(
        self, in_channels: int, out_channels: int, in_features: int, num_reps: int
    ):
        """Test constructing LinsumLayer with two inputs."""
        module = make_linsum_two_inputs(in_channels, out_channels, in_features, num_reps)

        # Check output shape matches input features (no halving for two inputs)
        assert module.out_shape.features == in_features
        assert module.out_shape.channels == out_channels
        assert module.out_shape.repetitions == num_reps

    def test_invalid_odd_features(self):
        """Test that odd number of features raises error for single input."""
        inputs = make_normal_leaf(out_features=3, out_channels=2, num_repetitions=1)
        with pytest.raises(ValueError, match="even number"):
            LinsumLayer(inputs=inputs, out_channels=2)

    def test_invalid_single_feature(self):
        """Test that single feature raises error."""
        inputs = make_normal_leaf(out_features=1, out_channels=2, num_repetitions=1)
        with pytest.raises(ValueError, match="at least 2"):
            LinsumLayer(inputs=inputs, out_channels=2)

    def test_invalid_two_inputs_channel_mismatch(self):
        """Test that mismatched channels raises error for two inputs (unlike EinsumLayer)."""
        left = make_leaf(cls=DummyLeaf, out_channels=2, scope=Scope([0, 1]), num_repetitions=1)
        right = make_leaf(cls=DummyLeaf, out_channels=3, scope=Scope([2, 3]), num_repetitions=1)
        with pytest.raises(ValueError, match="same number of channels"):
            LinsumLayer(inputs=[left, right], out_channels=4)

    def test_invalid_two_inputs_overlapping_scope(self):
        """Test that overlapping scopes raises error."""
        left = make_leaf(cls=DummyLeaf, out_channels=2, scope=Scope([0, 1]), num_repetitions=1)
        right = make_leaf(cls=DummyLeaf, out_channels=2, scope=Scope([1, 2]), num_repetitions=1)
        with pytest.raises(ValueError, match="disjoint"):
            LinsumLayer(inputs=[left, right], out_channels=2)


class TestLinsumLayerLogLikelihood:
    """Test LinsumLayer log-likelihood computation."""

    @pytest.mark.parametrize("in_channels,out_channels,in_features,num_reps", params)
    def test_log_likelihood_single_input(
        self, in_channels: int, out_channels: int, in_features: int, num_reps: int
    ):
        """Test log-likelihood with single input."""
        module = make_linsum_single_input(in_channels, out_channels, in_features, num_reps)
        data = make_normal_data(out_features=in_features)

        lls = module.log_likelihood(data)

        # Check output shape
        expected_shape = (
            data.shape[0],
            module.out_shape.features,
            module.out_shape.channels,
            num_reps,
        )
        assert lls.shape == expected_shape
        assert torch.isfinite(lls).all()

    @pytest.mark.parametrize("in_channels,out_channels,in_features,num_reps", params)
    def test_log_likelihood_two_inputs(
        self, in_channels: int, out_channels: int, in_features: int, num_reps: int
    ):
        """Test log-likelihood with two inputs."""
        module = make_linsum_two_inputs(in_channels, out_channels, in_features, num_reps)
        # Need data for combined scope (left + right = 2 * in_features)
        data = make_normal_data(out_features=in_features * 2)

        lls = module.log_likelihood(data)

        expected_shape = (
            data.shape[0],
            module.out_shape.features,
            module.out_shape.channels,
            num_reps,
        )
        assert lls.shape == expected_shape
        assert torch.isfinite(lls).all()

    def test_log_likelihood_cached(self):
        """Test that log-likelihood is cached correctly."""
        module = make_linsum_single_input(2, 3, 4, 1)
        data = make_normal_data(out_features=4)
        cache = Cache()

        lls = module.log_likelihood(data, cache=cache)

        # Check cache contains our result
        assert "log_likelihood" in cache
        assert cache["log_likelihood"].get(module) is not None
        assert torch.allclose(cache["log_likelihood"][module], lls)


class TestLinsumLayerSampling:
    """Test LinsumLayer sampling."""

    @pytest.mark.parametrize(
        "in_channels,out_channels,in_features,num_reps",
        product([2], [3], [4], [1, 2]),
    )
    def test_sample_single_input(
        self, in_channels: int, out_channels: int, in_features: int, num_reps: int
    ):
        """Test sampling with single input."""
        num_samples = 50
        module = make_linsum_single_input(in_channels, out_channels, in_features, num_reps)

        data = torch.full((num_samples, in_features), torch.nan)
        channel_index = torch.randint(
            low=0, high=out_channels, size=(num_samples, module.out_shape.features)
        )
        mask = torch.ones((num_samples, module.out_shape.features), dtype=torch.bool)
        repetition_index = torch.randint(low=0, high=num_reps, size=(num_samples,))

        sampling_ctx = SamplingContext(
            channel_index=channel_index, mask=mask, repetition_index=repetition_index
        )

        samples = module.sample(data=data, sampling_ctx=sampling_ctx)

        assert samples.shape == (num_samples, in_features)
        assert torch.isfinite(samples[:, module.scope.query]).all()

    @pytest.mark.parametrize(
        "in_channels,out_channels,in_features,num_reps",
        product([2], [3], [4], [1, 2]),
    )
    def test_sample_two_inputs(
        self, in_channels: int, out_channels: int, in_features: int, num_reps: int
    ):
        """Test sampling with two inputs."""
        num_samples = 50
        module = make_linsum_two_inputs(in_channels, out_channels, in_features, num_reps)
        total_features = in_features * 2

        data = torch.full((num_samples, total_features), torch.nan)
        channel_index = torch.randint(
            low=0, high=out_channels, size=(num_samples, module.out_shape.features)
        )
        mask = torch.ones((num_samples, module.out_shape.features), dtype=torch.bool)
        repetition_index = torch.randint(low=0, high=num_reps, size=(num_samples,))

        sampling_ctx = SamplingContext(
            channel_index=channel_index, mask=mask, repetition_index=repetition_index
        )

        samples = module.sample(data=data, sampling_ctx=sampling_ctx)

        assert samples.shape == (num_samples, total_features)
        assert torch.isfinite(samples[:, module.scope.query]).all()

    def test_mpe_sampling(self):
        """Test MPE (most probable explanation) sampling."""
        num_samples = 20
        module = make_linsum_single_input(2, 3, 4, 1)

        data = torch.full((num_samples, 4), torch.nan)
        channel_index = torch.zeros((num_samples, 2), dtype=torch.long)
        mask = torch.ones((num_samples, 2), dtype=torch.bool)
        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask)

        samples = module.sample(data=data, is_mpe=True, sampling_ctx=sampling_ctx)

        assert samples.shape == (num_samples, 4)
        assert torch.isfinite(samples).all()


class TestLinsumLayerWeights:
    """Test LinsumLayer weight properties."""

    def test_weights_normalized(self):
        """Test that weights sum to 1 over input channels."""
        module = make_linsum_single_input(3, 4, 6, 2)

        weights = module.weights
        sums = weights.sum(dim=-1)

        assert torch.allclose(sums, torch.ones_like(sums))

    def test_log_weights_consistent(self):
        """Test that log_weights equals log of weights."""
        module = make_linsum_single_input(3, 4, 6, 2)

        expected = torch.log(module.weights)
        actual = module.log_weights

        assert torch.allclose(expected, actual, atol=1e-6)

    def test_set_weights(self):
        """Test setting new weights."""
        module = make_linsum_single_input(2, 3, 4, 1)

        # Create valid weights
        new_weights = torch.rand(module.weights_shape) + 1e-8
        new_weights = new_weights / new_weights.sum(dim=-1, keepdim=True)

        module.weights = new_weights

        assert torch.allclose(module.weights, new_weights, atol=1e-5)

    def test_set_invalid_weights_shape(self):
        """Test that invalid weight shape raises error."""
        module = make_linsum_single_input(2, 3, 4, 1)

        with pytest.raises(ValueError, match="shape"):
            module.weights = torch.rand(1, 2, 3)

    def test_set_invalid_weights_not_normalized(self):
        """Test that unnormalized weights raise error."""
        module = make_linsum_single_input(2, 3, 4, 1)

        with pytest.raises(ValueError, match="sum to 1"):
            module.weights = torch.rand(module.weights_shape) + 0.1


class TestLinsumLayerMarginalization:
    """Test LinsumLayer marginalization."""

    def test_marginalize_partial_single_input(self):
        """Test partial marginalization with single input."""
        module = make_linsum_single_input(2, 3, 4, 1)

        # Marginalize first variable
        marg_module = module.marginalize([0])

        # Should return something (not fully marginalized)
        assert marg_module is not None

    def test_marginalize_full_single_input(self):
        """Test full marginalization returns None."""
        module = make_linsum_single_input(2, 3, 4, 1)

        # Marginalize all variables
        all_vars = list(module.scope.query)
        marg_module = module.marginalize(all_vars)

        assert marg_module is None

    def test_marginalize_no_overlap(self):
        """Test marginalizing variables not in scope returns self unchanged."""
        module = make_linsum_single_input(2, 3, 4, 1)

        # Marginalize variables not in scope
        marg_module = module.marginalize([100, 101])

        # Should return a LinsumLayer
        assert isinstance(marg_module, LinsumLayer)


class TestLinsumLayerGradientDescent:
    """Test gradient descent optimization."""

    def test_gradient_flow(self):
        """Test that gradients flow through log-likelihood."""
        module = make_linsum_single_input(2, 3, 4, 1)
        data = make_normal_data(out_features=4, num_samples=20)

        lls = module.log_likelihood(data)
        loss = -lls.mean()
        loss.backward()

        # Check that logits have gradients
        assert module.logits.grad is not None
        assert torch.isfinite(module.logits.grad).all()

    def test_optimization_changes_weights(self):
        """Test that optimization updates weights."""
        module = make_linsum_single_input(2, 3, 4, 1)
        data = make_normal_data(out_features=4, num_samples=50)

        weights_before = module.weights.clone()

        optimizer = torch.optim.Adam(module.parameters(), lr=0.1)
        for _ in range(5):
            optimizer.zero_grad()
            lls = module.log_likelihood(data)
            loss = -lls.mean()
            loss.backward()
            optimizer.step()

        # Weights should have changed
        assert not torch.allclose(module.weights, weights_before)


class TestLinsumLayerExtraRepr:
    """Test string representation."""

    def test_extra_repr(self):
        """Test extra_repr includes weight shape."""
        module = make_linsum_single_input(2, 3, 4, 1)
        repr_str = module.extra_repr()

        assert "weights=" in repr_str
        assert str(module.weights_shape) in repr_str


class TestLinsumLayerVsEinsumLayer:
    """Compare LinsumLayer and EinsumLayer behavior."""

    def test_weight_shape_difference(self):
        """Verify LinsumLayer has fewer parameters than EinsumLayer."""
        from spflow.modules.einsum import EinsumLayer

        in_channels = 3
        out_channels = 4
        in_features = 6
        num_reps = 2

        linsum = make_linsum_single_input(in_channels, out_channels, in_features, num_reps)

        einsum_input = make_normal_leaf(
            out_features=in_features, out_channels=in_channels, num_repetitions=num_reps
        )
        einsum = EinsumLayer(inputs=einsum_input, out_channels=out_channels)

        # LinsumLayer: linear combination = (D, O, R, C)
        # EinsumLayer: cross-product = (D, O, R, C, C)
        assert len(linsum.weights_shape) == 4
        assert len(einsum.weights_shape) == 5

        # Verify parameter count difference
        linsum_params = linsum.logits.numel()
        einsum_params = einsum.logits.numel()

        # LinsumLayer should have fewer parameters (no C×C cross-product)
        assert linsum_params < einsum_params
