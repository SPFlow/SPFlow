"""Tests for EinsumLayer equivalence with SumLayer(OuterProductLayer)."""

from itertools import product

import pytest
import torch

from spflow.modules.einsum import EinsumLayer
from spflow.modules.products import OuterProduct
from spflow.modules.sums import Sum
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, to_one_hot
from tests.utils.leaves import make_normal_leaf, make_normal_data
from tests.utils.sampling_context_helpers import patch_simple_as_categorical_one_hot

# Sweep dimensions/repetitions to catch shape-dependent equivalence regressions.
in_channels_values = [1, 3]
out_channels_values = [1, 4]
in_features_values = [2, 4]  # The reference graph splits features into two equal halves.
num_repetitions_values = [1, 2]

params = list(product(in_channels_values, out_channels_values, in_features_values, num_repetitions_values))


class TestEinsumLayerEquivalence:
    """Test EinsumLayer equivalence with SumLayer(OuterProductLayer)."""

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

        einsum = EinsumLayer(
            inputs=[left_input, right_input], out_channels=out_channels, num_repetitions=num_reps
        )

        prod_layer = OuterProduct(inputs=[left_input, right_input], num_splits=2)

        sum_layer = Sum(inputs=prod_layer, out_channels=out_channels, num_repetitions=num_reps)

        # Align parameter layout so this test compares operator behavior, not weight indexing.
        # Einsum stores pairwise channels separately, Sum flattens them into one channel axis.
        w_einsum = einsum.weights
        w_permuted = w_einsum.permute(0, 3, 4, 1, 2)
        w_sum = w_permuted.reshape(half_features, in_channels * in_channels, out_channels, num_reps)
        sum_layer.weights = w_sum

        return einsum, sum_layer

    @pytest.mark.parametrize("in_channels,out_channels,in_features,num_reps", params)
    def test_log_likelihood_equivalence(self, in_channels, out_channels, in_features, num_reps):
        """Test log-likelihood equivalence."""
        einsum, sum_layer = self._create_models(in_channels, out_channels, in_features, num_reps)
        batch_size = 10
        data = make_normal_data(out_features=in_features, num_samples=batch_size)

        ll_einsum = einsum.log_likelihood(data)
        ll_sum = sum_layer.log_likelihood(data)

        torch.testing.assert_close(ll_einsum, ll_sum, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("in_channels,out_channels,in_features,num_reps", params)
    def test_sampling_equivalence(self, in_channels, out_channels, in_features, num_reps):
        """Test sampling equivalence with fixed seed."""
        einsum, sum_layer = self._create_models(in_channels, out_channels, in_features, num_reps)
        batch_size = 10

        torch.manual_seed(42)
        out_features = einsum.out_shape.features
        channel_indices = torch.randint(0, out_channels, (batch_size, out_features))
        repetition_indices = torch.randint(0, num_reps, (batch_size,))
        mask = torch.ones((batch_size, out_features), dtype=torch.bool)
        ctx_common = SamplingContext(
            channel_index=channel_indices,
            repetition_index=repetition_indices,
            mask=mask,
        )

        torch.manual_seed(42)
        sample_einsum = einsum._sample(
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

        torch.testing.assert_close(sample_einsum, sample_sum, rtol=1e-5, atol=1e-8)

    def test_diff_sampling_equals_non_diff_sampling(self, monkeypatch: pytest.MonkeyPatch):
        in_channels = 2
        out_channels = 3
        in_features = 4
        num_reps = 2
        batch_size = 10
        einsum, _ = self._create_models(in_channels, out_channels, in_features, num_reps)

        channel_indices = torch.randint(0, out_channels, (batch_size, einsum.out_shape.features))
        repetition_indices = torch.randint(0, num_reps, (batch_size,))
        mask = torch.ones((batch_size, einsum.out_shape.features), dtype=torch.bool)
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
        samples_a = einsum._sample(
            data=torch.full((batch_size, in_features), torch.nan),
            sampling_ctx=sampling_ctx_a,
            cache=Cache(),
        )
        torch.manual_seed(1337)
        samples_b = einsum._sample(
            data=torch.full((batch_size, in_features), torch.nan),
            sampling_ctx=sampling_ctx_b,
            cache=Cache(),
        )

        torch.testing.assert_close(samples_a, samples_b, rtol=1e-6, atol=1e-6)
