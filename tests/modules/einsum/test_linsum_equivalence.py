"""Tests for LinsumLayer equivalence with SumLayer(ElementwiseProductLayer)."""

from itertools import product

import pytest
import torch

from spflow.modules.einsum import LinsumLayer
from spflow.modules.products import ElementwiseProduct
from spflow.modules.sums import Sum
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, to_one_hot
from tests.utils.leaves import make_normal_leaf, make_normal_data
from tests.utils.sampling_context_helpers import patch_simple_as_categorical_one_hot

# Sweep dimensions/repetitions to catch layout-dependent regressions.
in_channels_values = [1, 3]
out_channels_values = [1, 4]
in_features_values = [2, 4]  # Inputs are split in half in the equivalence construction.
num_repetitions_values = [1, 2]

params = list(product(in_channels_values, out_channels_values, in_features_values, num_repetitions_values))


class TestLinsumLayerEquivalence:
    """Test LinsumLayer equivalence with SumLayer(ElementwiseProductLayer)."""

    def _create_models(self, in_channels, out_channels, in_features, num_reps):
        """Helper to create equivalent models and sync weights."""
        half_features = in_features // 2

        left_input = make_normal_leaf(
            scope=list(range(0, half_features)),
            out_features=half_features,
            out_channels=in_channels,
            num_repetitions=num_reps,
        )
        right_input = make_normal_leaf(
            scope=list(range(half_features, in_features)),
            out_features=half_features,
            out_channels=in_channels,
            num_repetitions=num_reps,
        )

        linsum = LinsumLayer(
            inputs=[left_input, right_input], out_channels=out_channels, num_repetitions=num_reps
        )

        prod_layer = ElementwiseProduct(inputs=[left_input, right_input], num_splits=2)

        sum_layer = Sum(inputs=prod_layer, out_channels=out_channels, num_repetitions=num_reps)

        # Align parameter tensors so any output mismatch reflects layer logic, not indexing layout.
        # Linsum stores (D, O, R, C) while Sum expects (D, C, O, R).
        w_linsum = linsum.weights
        w_sum = w_linsum.permute(0, 3, 1, 2)
        sum_layer.weights = w_sum

        return linsum, sum_layer

    @pytest.mark.parametrize("in_channels,out_channels,in_features,num_reps", params)
    def test_log_likelihood_equivalence(self, in_channels, out_channels, in_features, num_reps):
        """Test log-likelihood equivalence."""
        linsum, sum_layer = self._create_models(in_channels, out_channels, in_features, num_reps)
        batch_size = 10
        data = make_normal_data(out_features=in_features, num_samples=batch_size)

        ll_linsum = linsum.log_likelihood(data)
        ll_sum = sum_layer.log_likelihood(data)

        torch.testing.assert_close(ll_linsum, ll_sum, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("in_channels,out_channels,in_features,num_reps", params)
    def test_sampling_equivalence(self, in_channels, out_channels, in_features, num_reps):
        """Test sampling equivalence with fixed seed."""
        linsum, sum_layer = self._create_models(in_channels, out_channels, in_features, num_reps)
        batch_size = 10

        torch.manual_seed(42)
        channel_indices = torch.randint(0, out_channels, (batch_size, linsum.out_shape.features))
        repetition_indices = torch.randint(0, num_reps, (batch_size,))
        mask = torch.ones((batch_size, linsum.out_shape.features), dtype=torch.bool)
        ctx_common = SamplingContext(
            channel_index=channel_indices,
            repetition_index=repetition_indices,
            mask=mask,
        )

        torch.manual_seed(42)
        sample_linsum = linsum._sample(
            data=torch.full((batch_size, in_features), torch.nan),
            sampling_ctx=ctx_common.copy(),
            cache=Cache(),
        )

        torch.manual_seed(42)
        sample_sum = sum_layer._sample(
            data=torch.full((batch_size, in_features), torch.nan),
            sampling_ctx=ctx_common.copy(),
            cache=Cache(),
        )

        torch.testing.assert_close(sample_linsum, sample_sum, rtol=1e-5, atol=1e-8)

    def test_diff_sampling_equals_non_diff_sampling_two_inputs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        in_channels = 2
        out_channels = 3
        in_features = 4
        num_reps = 2
        batch_size = 10
        linsum, _ = self._create_models(in_channels, out_channels, in_features, num_reps)

        channel_indices = torch.randint(0, out_channels, (batch_size, linsum.out_shape.features))
        repetition_indices = torch.randint(0, num_reps, (batch_size,))
        mask = torch.ones((batch_size, linsum.out_shape.features), dtype=torch.bool)
        sampling_ctx = SamplingContext(
            channel_index=channel_indices.clone(),
            repetition_index=repetition_indices.clone(),
            mask=mask.clone(),
            is_mpe=False,
        )
        sampling_ctx_diff = SamplingContext(
            channel_index=to_one_hot(channel_indices, dim=-1, dim_size=out_channels),
            repetition_index=to_one_hot(repetition_indices, dim=-1, dim_size=num_reps),
            mask=mask.clone(),
            is_mpe=False,
            is_differentiable=True,
        )

        patch_simple_as_categorical_one_hot(monkeypatch)

        torch.manual_seed(1337)
        samples_a = linsum._sample(
            data=torch.full((batch_size, in_features), torch.nan),
            sampling_ctx=sampling_ctx,
            cache=Cache(),
        )
        torch.manual_seed(1337)
        samples_b = linsum._sample(
            data=torch.full((batch_size, in_features), torch.nan),
            sampling_ctx=sampling_ctx_diff,
            cache=Cache(),
        )

        torch.testing.assert_close(samples_a, samples_b, rtol=1e-6, atol=1e-6)


class TestLinsumLayerSingleInputEquivalence:
    """Test LinsumLayer single-input (SplitConsecutive) equivalence with Sum(ElementwiseProduct(SplitConsecutive))."""

    # Keep only valid explicit-composition shapes so failures point to Linsum semantics.
    # Larger channel counts fail ElementwiseProduct validation for unrelated reasons.
    single_input_params = [(ic, oc, if_, nr) for ic, oc, if_, nr in params if ic <= if_ // 2]

    def _create_single_input_models(self, in_channels, out_channels, in_features, num_reps):
        """Helper to create equivalent models with single input and sync weights."""
        # Single-leaf input exercises Linsum's internal split path against an explicit split graph.
        from spflow.modules.ops.split_consecutive import SplitConsecutive

        leaf = make_normal_leaf(
            scope=list(range(in_features)),
            out_features=in_features,
            out_channels=in_channels,
            num_repetitions=num_reps,
        )

        linsum = LinsumLayer(inputs=leaf, out_channels=out_channels, num_repetitions=num_reps)

        split_layer = SplitConsecutive(leaf, dim=1, num_splits=2)
        prod_layer = ElementwiseProduct(inputs=split_layer, num_splits=2)
        sum_layer = Sum(inputs=prod_layer, out_channels=out_channels, num_repetitions=num_reps)

        # Match parameter layout between implementations to isolate structural differences.
        w_linsum = linsum.weights
        w_sum = w_linsum.permute(0, 3, 1, 2)
        sum_layer.weights = w_sum

        return linsum, sum_layer

    @pytest.mark.parametrize("in_channels,out_channels,in_features,num_reps", single_input_params)
    def test_log_likelihood_single_input_equivalence(self, in_channels, out_channels, in_features, num_reps):
        """Test log-likelihood equivalence for single-input LinsumLayer."""
        linsum, sum_layer = self._create_single_input_models(in_channels, out_channels, in_features, num_reps)
        batch_size = 10
        data = make_normal_data(out_features=in_features, num_samples=batch_size)

        ll_linsum = linsum.log_likelihood(data)
        ll_sum = sum_layer.log_likelihood(data)

        torch.testing.assert_close(ll_linsum, ll_sum, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("in_channels,out_channels,in_features,num_reps", single_input_params)
    def test_sampling_single_input_equivalence(self, in_channels, out_channels, in_features, num_reps):
        """Test sampling equivalence for single-input LinsumLayer with fixed seed."""
        linsum, sum_layer = self._create_single_input_models(in_channels, out_channels, in_features, num_reps)
        batch_size = 10

        torch.manual_seed(42)
        channel_indices = torch.randint(0, out_channels, (batch_size, linsum.out_shape.features))
        repetition_indices = torch.randint(0, num_reps, (batch_size,))
        mask = torch.ones((batch_size, linsum.out_shape.features), dtype=torch.bool)
        ctx_common = SamplingContext(
            channel_index=channel_indices,
            repetition_index=repetition_indices,
            mask=mask,
        )

        torch.manual_seed(42)
        sample_linsum = linsum._sample(
            data=torch.full((batch_size, in_features), torch.nan),
            sampling_ctx=ctx_common.copy(),
            cache=Cache(),
        )

        torch.manual_seed(42)
        sample_sum = sum_layer._sample(
            data=torch.full((batch_size, in_features), torch.nan),
            sampling_ctx=ctx_common.copy(),
            cache=Cache(),
        )

        torch.testing.assert_close(sample_linsum, sample_sum, rtol=1e-5, atol=1e-8)

    def test_diff_sampling_equals_non_diff_sampling_single_input(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        in_channels = 2
        out_channels = 3
        in_features = 4
        num_reps = 2
        batch_size = 10
        linsum, _ = self._create_single_input_models(in_channels, out_channels, in_features, num_reps)

        channel_indices = torch.randint(0, out_channels, (batch_size, linsum.out_shape.features))
        repetition_indices = torch.randint(0, num_reps, (batch_size,))
        mask = torch.ones((batch_size, linsum.out_shape.features), dtype=torch.bool)
        sampling_ctx_a = SamplingContext(
            channel_index=channel_indices.clone(),
            repetition_index=repetition_indices.clone(),
            mask=mask.clone(),
            is_mpe=False,
        )
        sampling_ctx_b = SamplingContext(
            channel_index=to_one_hot(channel_indices, dim=-1, dim_size=out_channels),
            repetition_index=to_one_hot(repetition_indices, dim=-1, dim_size=num_reps),
            mask=mask.clone(),
            is_mpe=False,
            is_differentiable=True,
        )

        patch_simple_as_categorical_one_hot(monkeypatch)

        torch.manual_seed(1337)
        samples_a = linsum._sample(
            data=torch.full((batch_size, in_features), torch.nan),
            sampling_ctx=sampling_ctx_a,
            cache=Cache(),
        )
        torch.manual_seed(1337)
        samples_b = linsum._sample(
            data=torch.full((batch_size, in_features), torch.nan),
            sampling_ctx=sampling_ctx_b,
            cache=Cache(),
        )

        torch.testing.assert_close(samples_a, samples_b, rtol=1e-6, atol=1e-6)
